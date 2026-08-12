"""在 execution 的 World step 边界采样相机帧，并编排输出资源的 observer。

采样发生在 physics step 完成后，时间戳使用该步结束时刻；observer 只负责调度采样并把
frame 交给 publisher，不在仿真线程内执行文件或网络 I/O。输出启动分成
``prepare -> apply paths -> open sinks`` 三阶段：先验证全部相机和 MCAP 目标，再统一修改
文件系统，最后打开资源，避免后发现的一处配置错误留下前面已打开的半套输出。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path

from linkerbot_sim.utils.output_paths import (
    OutputPathPlan,
    apply_output_path_plans,
    plan_output_file,
    timestamped_run_name,
)

from .foxglove import FoxgloveCameraFrameSink
from .frame import sample_camera_frames
from .recorder import (
    CameraFramePublisher,
    CameraFrameSink,
    CompositeCameraFrameSink,
    OfflineCameraFrameSink,
    OfflineCameraFrameSinkPlan,
    validate_camera_frame_publisher_settings,
)
from .runtime import SensorCameraRuntime


PathResolver = Callable[[str], Path]


@dataclass(frozen=True)
class CameraPublisherSettings:
    """产品无关的 camera publisher 资源策略。

    Mirror 的 strict ``CameraOutputSettings`` 具有相同字段，可以直接以结构化方式传入；
    本地默认对象只服务独立 sensor API。sensor 层因此不再反向依赖全局 runtime 配置，
    也不会复制 Mirror 的路径或 endpoint 事实。
    """

    queue_size: int = 128
    overflow_policy: str = "block"
    worker_poll_interval_s: float = 0.1
    existing_data_policy: str = "error"
    shutdown_policy: str = "drain"
    rgb_format: str = "ppm"
    depth_format: str = "npy"
    metadata_flush_interval_frames: int = 1
    max_bytes_per_camera: int = 1_073_741_824


class CameraFrameObserver:
    """按每台相机的 frequency 在 world step 后采样。

    采样游标按相机名称独立维护，frame index 则按 ``(camera, modality)`` 独立递增；因此
    RGB 与 depth 可以拥有各自连续的序号，而不会因为另一 modality 缺帧产生假空洞。
    """

    def __init__(
        self,
        *,
        cameras: Sequence[SensorCameraRuntime],
        publisher: CameraFramePublisher,
    ) -> None:
        """记录需要观察的 camera，并初始化每个 camera 的采样时间游标。"""

        self.cameras = tuple(cameras)
        self.publisher = publisher
        self._next_sample_time: dict[str, float] = {}
        self._frame_indices: dict[tuple[str, str], int] = {}

    def observe(self, world, *, step: int, phase: str | None = None) -> None:
        """在 physics step 后采样当前到期的相机并把 frame 提交给 publisher。

        ``step`` 是从零开始的已完成 step 索引，所以物理时间为 ``(step + 1) * dt``。
        ``phase`` 当前仅满足通用 execution observer 接口，不参与采样或 metadata。
        """

        del phase
        time_s = (step + 1) * float(world.get_physics_dt())
        for camera in self.cameras:
            if not self._should_sample(camera, time_s):
                continue
            for frame in sample_camera_frames(
                camera,
                frame_indices=self._frame_indices,
                simulation_step=step,
                time_s=time_s,
            ):
                self.publisher.publish(frame)

    def _should_sample(self, camera: SensorCameraRuntime, time_s: float) -> bool:
        """按仿真时间判断当前相机是否到达下一采样点。

        ``1e-9`` 容差吸收浮点累积误差，避免理论上恰好到期的帧因极小舍入误差被推迟一个
        physics step。游标从实际采样时刻向后推进一个周期，不追补跳过的历史帧。
        """

        next_time = self._next_sample_time.get(camera.name)
        if next_time is not None and time_s + 1.0e-9 < next_time:
            return False
        self._next_sample_time[camera.name] = time_s + 1.0 / camera.settings.frequency
        return True

    def reset(self) -> None:
        """清理 reset 前的采样节奏和 frame index，使新 episode 从零开始。"""

        self._next_sample_time.clear()
        self._frame_indices.clear()


@dataclass
class CameraOutputHandle:
    """同时持有采样 observer 与后台 publisher 的相机输出运行时句柄。"""

    observer: CameraFrameObserver
    publisher: CameraFramePublisher

    def close(self) -> bool:
        """关闭后台 publisher 和其持有的所有输出 sink。"""

        return self.publisher.close()


@dataclass(frozen=True)
class PreparedCameraOutput:
    """只读的相机输出预检结果；尚未打开 sink，也未修改目标路径。

    ``path_plans`` 汇总所有离线目录和 MCAP 文件，供上层一次性检查重叠目标后应用；
    live/MCAP group 已按共享 endpoint/path 合并相机 topic，确保一个资源只打开一次。
    """

    observed_cameras: tuple[SensorCameraRuntime, ...]
    offline_plans: tuple[OfflineCameraFrameSinkPlan, ...]
    live_groups: tuple[tuple[str, int, Mapping[str, str]], ...]
    mcap_groups: tuple[tuple[OutputPathPlan, Mapping[str, str]], ...]
    path_plans: tuple[OutputPathPlan, ...]
    settings: CameraPublisherSettings
    shutdown_timeout_s: float


def start_camera_output(
    cameras: Sequence[SensorCameraRuntime],
    *,
    path_resolver: PathResolver | None = None,
    settings: CameraPublisherSettings | None = None,
    shutdown_timeout_s: float = 2.0,
) -> CameraOutputHandle | None:
    """根据 camera output 配置统一启动离线、Foxglove live 和 MCAP 输出。"""

    return _start_camera_output(
        cameras,
        path_resolver=path_resolver,
        include_foxglove=True,
        settings=settings or CameraPublisherSettings(),
        shutdown_timeout_s=shutdown_timeout_s,
    )


def _start_camera_output(
    cameras: Sequence[SensorCameraRuntime],
    *,
    path_resolver: PathResolver | None,
    include_foxglove: bool,
    settings: CameraPublisherSettings,
    shutdown_timeout_s: float,
) -> CameraOutputHandle | None:
    """为独立调用方顺序执行预检、路径应用和 sink 打开。"""

    prepared = prepare_camera_output(
        cameras,
        path_resolver=path_resolver,
        include_foxglove=include_foxglove,
        settings=settings,
        shutdown_timeout_s=shutdown_timeout_s,
    )
    apply_output_path_plans(prepared.path_plans)
    return open_prepared_camera_output(prepared)


def prepare_camera_output(
    cameras: Sequence[SensorCameraRuntime],
    *,
    path_resolver: PathResolver | None = None,
    include_foxglove: bool = True,
    settings: CameraPublisherSettings,
    shutdown_timeout_s: float,
) -> PreparedCameraOutput:
    """预检全部相机输出，不打开资源或修改任何目标。

    离线目录必须每相机唯一；相同 live endpoint 或 MCAP path 会合并为一个 sink。包含文件
    输出时禁止有损 queue 策略，MCAP 也不支持无法证明安全续写的 ``resume``。使用
    ``timestamped_dir`` 时，所有 plan 共享一次生成的 UTC run name，使同次启动的目录落在
    同一 run namespace。
    """

    timeout = validate_camera_frame_publisher_settings(
        max_queue_size=settings.queue_size,
        overflow_policy=settings.overflow_policy,
        worker_poll_interval_s=settings.worker_poll_interval_s,
        shutdown_policy=settings.shutdown_policy,
        shutdown_timeout_s=shutdown_timeout_s,
    )

    offline_dirs = _resolved_offline_dirs(cameras, path_resolver=path_resolver)
    observed_cameras: list[SensorCameraRuntime] = []
    live_groups: dict[tuple[str, int], dict[str, str]] = {}
    mcap_groups: dict[Path, dict[str, str]] = {}
    for camera in cameras:
        output = camera.settings.output
        save_dir = output.save_dir
        topic_prefix = output.foxglove_topic_prefix or f"/cameras/{camera.name}"
        if include_foxglove and output.foxglove_live_port is not None:
            key = (output.foxglove_live_host, output.foxglove_live_port)
            live_groups.setdefault(key, {})[camera.name] = topic_prefix
        if include_foxglove and output.foxglove_mcap_path is not None:
            mcap_path = (
                Path(output.foxglove_mcap_path)
                if path_resolver is None
                else path_resolver(output.foxglove_mcap_path)
            )
            mcap_path = _lexical_absolute_path(mcap_path)
            mcap_groups.setdefault(mcap_path, {})[camera.name] = topic_prefix
        if (
            save_dir is not None
            or (include_foxglove and output.foxglove_live_port is not None)
            or (include_foxglove and output.foxglove_mcap_path is not None)
        ):
            observed_cameras.append(camera)
    if (offline_dirs or mcap_groups) and settings.overflow_policy in {
        "drop_oldest",
        "drop_newest",
    }:
        raise ValueError(
            "offline camera output requires overflow_policy 'block' or 'error'; "
            "lossy policies may only be used for realtime-only output"
        )
    if mcap_groups and settings.existing_data_policy == "resume":
        raise ValueError(
            "camera MCAP does not support existing_data_policy='resume'; "
            "use error, truncate, or timestamped_dir"
        )

    # 同一启动批次只生成一次时间戳，否则跨秒/微秒创建的相机目录无法归属于同一 run。
    run_name = (
        timestamped_run_name()
        if settings.existing_data_policy == "timestamped_dir"
        else None
    )
    offline_plans = {
        camera.name: OfflineCameraFrameSink.prepare(
            camera_name=camera.name,
            save_dir=offline_dirs[camera.name],
            existing_data_policy=settings.existing_data_policy,
            timestamped_run_name=run_name,
            rgb_format=settings.rgb_format,
            depth_format=settings.depth_format,
            metadata_flush_interval_frames=(settings.metadata_flush_interval_frames),
            max_bytes_per_camera=settings.max_bytes_per_camera,
        )
        for camera in cameras
        if camera.name in offline_dirs
    }
    mcap_plans = {
        path: plan_output_file(
            path,
            policy=settings.existing_data_policy,
            run_name=run_name,
        )
        for path in mcap_groups
    }
    path_plans: list[OutputPathPlan] = [
        plan.path_plan for plan in offline_plans.values()
    ]
    path_plans.extend(mcap_plans.values())
    return PreparedCameraOutput(
        observed_cameras=tuple(observed_cameras),
        offline_plans=tuple(offline_plans.values()),
        live_groups=tuple(
            (host, port, dict(prefixes))
            for (host, port), prefixes in live_groups.items()
        ),
        mcap_groups=tuple(
            (mcap_plans[path], dict(prefixes)) for path, prefixes in mcap_groups.items()
        ),
        path_plans=tuple(path_plans),
        settings=settings,
        shutdown_timeout_s=timeout,
    )


def open_prepared_camera_output(
    prepared: PreparedCameraOutput,
) -> CameraOutputHandle | None:
    """在调用方应用完整路径计划后打开 sink 并启动 publisher。

    任一 sink 或线程启动失败都会尽力关闭此前已打开资源，再保留原启动异常向上抛出。
    空计划不会创建线程，直接返回 ``None``。
    """

    if (
        not prepared.offline_plans
        and not prepared.live_groups
        and not prepared.mcap_groups
    ):
        return None

    sinks: list[CameraFrameSink] = []
    publisher: CameraFramePublisher | None = None
    try:
        sinks.extend(
            OfflineCameraFrameSink.open_prepared(plan)
            for plan in prepared.offline_plans
        )
        for host, port, prefixes in prepared.live_groups:
            sinks.append(
                FoxgloveCameraFrameSink.open_live(
                    host=host,
                    port=port,
                    topic_prefix_by_camera=prefixes,
                )
            )
        for mcap_plan, prefixes in prepared.mcap_groups:
            sinks.append(
                FoxgloveCameraFrameSink.open_mcap(
                    mcap_plan.resolved_path,
                    topic_prefix_by_camera=prefixes,
                )
            )
        sink: CameraFrameSink = (
            sinks[0] if len(sinks) == 1 else CompositeCameraFrameSink(sinks)
        )
        publisher = CameraFramePublisher(
            sink=sink,
            name="camera-offline-writer",
            max_queue_size=prepared.settings.queue_size,
            overflow_policy=prepared.settings.overflow_policy,
            worker_poll_interval_s=prepared.settings.worker_poll_interval_s,
            shutdown_policy=prepared.settings.shutdown_policy,
            shutdown_timeout_s=prepared.shutdown_timeout_s,
        )
        observer = CameraFrameObserver(
            cameras=prepared.observed_cameras,
            publisher=publisher,
        )
        publisher.start()
    except BaseException:
        _rollback_camera_output_start(publisher=publisher, sinks=sinks)
        raise
    return CameraOutputHandle(observer=observer, publisher=publisher)


def _rollback_camera_output_start(
    *,
    publisher: CameraFramePublisher | None,
    sinks: Sequence[CameraFrameSink],
) -> None:
    """尽力回滚已打开的输出资源，同时不替换原始启动异常。"""

    if publisher is not None:
        try:
            publisher.close()
            return
        except BaseException:
            pass
    if sinks:
        try:
            CompositeCameraFrameSink(sinks).close()
        except BaseException:
            pass


def _resolved_offline_dirs(
    cameras: Sequence[SensorCameraRuntime],
    *,
    path_resolver: PathResolver | None,
) -> dict[str, Path]:
    """解析并拒绝多 camera 共享 recorder namespace。"""

    result: dict[str, Path] = {}
    owners: dict[Path, str] = {}
    for camera in cameras:
        save_dir = camera.settings.output.save_dir
        if save_dir is None:
            continue
        output_dir = (
            Path(save_dir) if path_resolver is None else path_resolver(save_dir)
        )
        output_dir = _lexical_absolute_path(output_dir)
        previous = owners.get(output_dir)
        if previous is not None and previous != camera.name:
            raise ValueError(
                "camera output save_dir must be unique per camera: "
                f"{previous!r} and {camera.name!r} both resolve to {output_dir}"
            )
        owners[output_dir] = camera.name
        result[camera.name] = output_dir
    return result


def _lexical_absolute_path(path: str | Path) -> Path:
    """规范化 ``.``/``..`` 并转为绝对路径，但不解析最终输出符号链接。

    这里需要稳定比较多个配置路径是否指向同一 lexical target；符号链接安全由后续路径
    预检单独拒绝，不能用 ``resolve`` 提前跟随到工作区之外。
    """

    return Path(os.path.abspath(Path(path).expanduser()))

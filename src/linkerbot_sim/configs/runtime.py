"""运行时 profile 的严格解析、合并与解析结果追踪。

运行时 profile 负责进程级设置，包括所选业务 profile、Isaac 应用参数、交互传输、
规划资源、回放、相机输出、遥测及关闭超时。本模块刻意保持为纯 Python 边界，不导入
Isaac、torch 或 cuRobo，因此调用方可以在仿真应用启动前完成类型、枚举、范围、跨字段
约束和依赖文件检查。

配置生命周期分为两步：``RuntimeProfileConfig.from_mapping`` 严格解析单份 YAML，
``resolve_runtime_config`` 再按“代码默认值 < profile < 显式 CLI 覆盖”的优先级生成最终
不可变配置，并记录每个叶子字段的来源和内容指纹。本模块只解析配置，不启动仿真、不创建
网络端点，也不写入输出文件。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from math import isfinite
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Literal, TypeAlias

from linkerbot_sim.sensors.camera.limits import DEFAULT_MAX_BYTES_PER_CAMERA
from linkerbot_sim.utils.config import load_yaml, require_loopback_host
from linkerbot_sim.utils.output_paths import EXISTING_DATA_POLICIES


RUNTIME_MODES = frozenset({"single_scene", "tiled_scene"})
MaxBatchProblems: TypeAlias = int | Literal["auto"]


@dataclass(frozen=True)
class RuntimeProfileSelection:
    """一次运行引用的命名 profile 集合。

    字段均为 ``configs`` 下对应分组中的稳定文件名，不包含目录或扩展名。解析时会拒绝
    路径分隔符，最终解析阶段还会确认环境、cuRobo 和日志 profile 确实存在；控制器
    bundle 则在完整依赖图校验时加载。
    """

    env: str = "scene1"
    curobo: str = "default"
    logging: str = "default_logger"
    controller_bundle: str = "default"


@dataclass(frozen=True)
class SimulationGpuSettings:
    """Isaac 应用和物理仿真使用的 GPU 索引约束。

    ``max_gpu_count`` 是进程允许使用的设备数量上限，``active_gpu`` 与
    ``physics_gpu`` 必须落在 ``[0, max_gpu_count)``。``multi_gpu`` 只表达启动意图，
    本模块不会探测主机实际 GPU 数量。
    """

    multi_gpu: bool = False
    max_gpu_count: int = 1
    active_gpu: int = 0
    physics_gpu: int = 0


@dataclass(frozen=True)
class SimulationRenderSettings:
    """Isaac 应用窗口、渲染器和无头模式的渲染参数。

    三个尺寸字段均为正整数 ``(宽, 高)``；抗锯齿级别和逐像素采样数受严格范围校验。
    ``hide_ui``、``disable_viewport_updates`` 与 ``fast_shutdown`` 允许为 ``None``，表示
    由启动层按 GUI/无头上下文决定。``headless_dt_policy`` 控制无头运行时是否为相机
    保留渲染节奏，本对象本身不修改 Isaac settings。
    """

    gui_size: tuple[int, int] = (1280, 720)
    headless_size: tuple[int, int] = (640, 480)
    window_size: tuple[int, int] = (1440, 900)
    renderer: str = "RaytracedLighting"
    anti_aliasing_gui: int = 3
    anti_aliasing_headless: int = 0
    samples_per_pixel_per_frame: int = 1
    denoiser: bool = False
    hide_ui: bool | None = None
    disable_viewport_updates: bool | None = None
    fast_shutdown: bool | None = None
    material_sync_loads: bool = False
    hydra_material_sync_loads: bool = False
    headless_dt_policy: str = "camera_aware"


@dataclass(frozen=True)
class SimulationAppSettings:
    """仿真应用启动参数的聚合配置。

    ``gui`` 选择交互窗口或无头运行；``gpu`` 和 ``render`` 分别承载设备与渲染设置。
    实例在运行 profile 解析后保持不可变，真正创建 ``SimulationApp`` 由应用层负责。
    """

    gui: bool = False
    gpu: SimulationGpuSettings = field(default_factory=SimulationGpuSettings)
    render: SimulationRenderSettings = field(default_factory=SimulationRenderSettings)


@dataclass(frozen=True)
class RuntimeCommandDefaults:
    """交互命令未显式给值时采用的运动语义。

    ``joint_interpolation`` 指定关节轨迹插值，``pose_frame`` 指定末端位姿参考系，
    ``orientation_mode`` 指定缺省姿态处理方式。三个字段都在解析时限制为当前运行时
    已实现的枚举值。
    """

    joint_interpolation: str = "smoothstep"
    pose_frame: str = "env"
    orientation_mode: str = "current"


@dataclass(frozen=True)
class RuntimeExecutionSettings:
    """物理步进和控制命令执行策略。

    ``control_mode`` 选择 articulation action 类型；``idle_physics_policy`` 与
    ``idle_step_duration_s`` 决定没有待执行命令时是否及如何继续推进物理；
    ``default_decimation`` 是命令未指定降频值时的默认 physics-step 倍数。
    ``tiled_scene`` 模式目前只接受位置控制，该不变量在跨字段校验中保证。
    """

    control_mode: str = "position"
    idle_physics_policy: str = "hold_step"
    idle_step_duration_s: float = 0.05
    default_decimation: int = 2
    command_defaults: RuntimeCommandDefaults = field(
        default_factory=RuntimeCommandDefaults
    )


@dataclass(frozen=True)
class RuntimeTransportEndpoint:
    """单个交互监听端点的启用状态和地址。

    ``host`` 必须是回环地址，避免配置意外扩大服务暴露面；端点启用时 ``port`` 必须为
    ``1..65535``。该对象仅描述监听参数，不持有 socket 生命周期。
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int | None = None


@dataclass(frozen=True)
class RuntimeTransportSettings:
    """交互协议端点、队列容量与轮询时限。

    ``tcp_jsonl`` 和 ``websocket`` 可独立启用；消息大小、连接数和请求/事件队列容量
    都是严格正整数。``overflow_policy`` 决定队列满时的处理方式，三个时间字段分别约束
    服务启动、服务端轮询和响应轮询。资源实际创建与关闭由交互运行时负责。
    """

    tcp_jsonl: RuntimeTransportEndpoint = field(
        default_factory=RuntimeTransportEndpoint
    )
    websocket: RuntimeTransportEndpoint = field(
        default_factory=RuntimeTransportEndpoint
    )
    max_message_bytes: int = 1_048_576
    max_connections: int = 16
    request_queue_capacity: int = 256
    event_queue_capacity: int = 256
    overflow_policy: str = "reject"
    startup_timeout_s: float = 5.0
    server_poll_interval_s: float = 0.1
    response_poll_interval_s: float = 0.5


@dataclass(frozen=True)
class InteractiveRuntimeSettings:
    """标准输入、命令队列、快照请求与外部传输的交互配置。

    ``stdin_eof_policy`` 决定输入流结束后退出还是继续运行；两个 timeout 均以秒计。
    历史和快照容量限制常驻内存，``transport`` 进一步限制网络侧资源。配置在进程启动时
    解析一次，运行期间不就地修改。
    """

    stdin_enabled: bool = True
    stdin_eof_policy: str = "exit"
    queue_poll_timeout_s: float = 0.05
    snapshot_timeout_s: float = 30.0
    command_history_capacity: int = 256
    snapshot_request_capacity: int = 32
    transport: RuntimeTransportSettings = field(
        default_factory=RuntimeTransportSettings
    )


@dataclass(frozen=True)
class PlannerRequestDefaults:
    """规划请求省略字段时采用的行为。

    ``duration_s`` 是目标运动时长；碰撞相关开关控制世界刷新与避障；``coordination``
    描述多机器人请求的协调模式；``load_on_success`` 和 ``replace`` 决定成功轨迹是否立即
    装载以及是否替换既有队列。后端能力相关组合会在跨字段校验中被拒绝。
    """

    duration_s: float = 1.0
    avoid_collisions: bool = False
    force_collision_refresh: bool = False
    coordination: str = "independent"
    load_on_success: bool = True
    replace: bool = True


@dataclass(frozen=True)
class PlannerResourceSettings:
    """异步规划器的线程、队列、结果缓存与批处理上限。

    ``max_batch_problems`` 可在输入阶段为正整数或 ``"auto"``；最终解析时会结合环境数和
    所选 cuRobo profile 的容量解析为整数。``shutdown_timeout_s`` 是等待规划工作线程退出
    的秒数，其他容量字段均为严格正整数。
    """

    max_workers: int = 2
    max_pending_requests: int = 64
    max_completed_results: int = 256
    max_batch_problems: MaxBatchProblems = 64
    shutdown_timeout_s: float = 30.0


@dataclass(frozen=True)
class RuntimePlannerSettings:
    """规划后端、请求策略和资源限制的完整运行时配置。

    ``joint_batch_mode`` 控制关节请求批处理，``oversize_request_policy`` 处理超出批容量的
    请求，``failure_policy`` 决定部分环境失败后的响应。``request_defaults`` 与
    ``resources`` 分别负责请求语义和进程资源；对象不创建实际 planner。
    """

    backend: str = "curobo"
    joint_batch_mode: str = "auto"
    request_defaults: PlannerRequestDefaults = field(
        default_factory=PlannerRequestDefaults
    )
    oversize_request_policy: str = "split"
    failure_policy: str = "hold_failed_env"
    resources: PlannerResourceSettings = field(default_factory=PlannerResourceSettings)


@dataclass(frozen=True)
class PlaybackResourceSettings:
    """逐环境轨迹回放队列的防御性资源上限。

    深度、样本数和累计时长分别限制单个环境中尚未消费的数据，``overflow_policy`` 决定
    超限时拒绝还是采用其它当前支持策略。限制在请求进入回放系统时使用。
    """

    max_queue_depth_per_env: int = 32
    max_samples_per_env: int = 100_000
    max_duration_s_per_env: float = 3600.0
    overflow_policy: str = "reject"


@dataclass(frozen=True)
class CameraOutputRuntimeSettings:
    """相机异步落盘队列、编码格式和数据保留策略。

    ``queue_size`` 和 ``overflow_policy`` 控制生产者背压；``existing_data_policy`` 决定
    目标已存在时的处理；``shutdown_policy`` 决定退出时排空或丢弃队列。RGB/深度格式、
    元数据刷盘帧间隔和每相机目录字节配额均在启动前验证。实际目录和工作线程不由本类创建。
    """

    queue_size: int = 128
    overflow_policy: str = "block"
    worker_poll_interval_s: float = 0.1
    existing_data_policy: str = "error"
    shutdown_policy: str = "drain"
    rgb_format: str = "ppm"
    depth_format: str = "npy"
    metadata_flush_interval_frames: int = 1
    max_bytes_per_camera: int = DEFAULT_MAX_BYTES_PER_CAMERA


@dataclass(frozen=True)
class TelemetryTopicSettings:
    """标准关节、场景和状态消息对应的绝对 topic 路径。

    路径必须以 ``/`` 开头、不得包含非法父级片段，并且三者必须互不相同，防止不同消息
    schema 写入同一 topic。
    """

    joint_states: str = "/joint_states"
    scene: str = "/scene"
    state: str = "/linkerbot/state"


@dataclass(frozen=True)
class McapRuntimeSettings:
    """可选 MCAP 文件输出位置；``None`` 表示不创建 MCAP sink。"""

    path: str | None = None


@dataclass(frozen=True)
class FoxgloveLiveRuntimeSettings:
    """Foxglove 实时服务的回环监听配置。

    启用时必须提供合法端口；``host`` 与交互传输相同，只允许回环地址。本对象不启动
    Foxglove 服务。
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int | None = None


@dataclass(frozen=True)
class TelemetryRuntimeSettings:
    """遥测采样范围、频率、消息内容和输出 sink 配置。

    ``primary_env_id`` 是标准单环境 topic 的数据源，``selected_env_ids`` 与
    ``publish_decimation`` 控制 tiled 采样范围和降频；两者必须与环境数量一致。
    ``rate_hz``、缓冲容量、丢弃和错误策略控制发布节奏；``include_*`` 字段决定是否采集
    关节、状态、场景、effort 与对象数据。``joint_effort_field`` 只适用于
    ``single_scene`` 模式。
    ``topics``、``mcap``、``foxglove_live`` 描述输出，但 sink 生命周期由遥测层管理。
    """

    primary_env_id: int = 0
    selected_env_ids: tuple[int, ...] = (0,)
    publish_decimation: int = 1
    rate_hz: float = 60.0
    buffer_size: int = 1
    drop_policy: str = "latest"
    on_error: str = "stop"
    include_joint_states: bool = True
    include_state_json: bool = True
    include_scene_markers: bool = False
    include_efforts: bool = False
    include_objects: bool = False
    joint_effort_field: str = "none"
    topics: TelemetryTopicSettings = field(default_factory=TelemetryTopicSettings)
    mcap: McapRuntimeSettings = field(default_factory=McapRuntimeSettings)
    foxglove_live: FoxgloveLiveRuntimeSettings = field(
        default_factory=FoxgloveLiveRuntimeSettings
    )


@dataclass(frozen=True)
class OutputPolicySettings:
    """CSV 与 MCAP 目标已存在时采用的文件处理策略。

    两个字段均限制为输出路径模块支持的策略集合；该对象不检查具体目标，也不执行删除、
    续写或目录创建。
    """

    csv_existing_file_policy: str = "error"
    mcap_existing_file_policy: str = "error"


@dataclass(frozen=True)
class RuntimePathSettings:
    """进程级可选路径覆盖。

    ``cache_root`` 为 ``None`` 时由消费方使用项目默认缓存目录；非空路径禁止 ``..`` 与
    NUL 字节。本配置只保留路径文本，不提前创建目录。
    """

    cache_root: str | None = None


@dataclass(frozen=True)
class ShutdownSettings:
    """关闭阶段等待各异步子系统退出的独立超时。

    三个值分别约束状态发布器、相机发布器和交互传输，单位均为秒且不得为负。拆分超时可让
    关闭协调器准确报告具体未退出的子系统。
    """

    state_publisher_timeout_s: float = 2.0
    camera_publisher_timeout_s: float = 2.0
    transport_timeout_s: float = 2.0


@dataclass(frozen=True)
class RuntimeProfileConfig:
    """严格解析后的单份运行时 profile。

    公开字段与 ``runtime`` YAML 树一一对应，实例冻结后可安全地跨启动组件共享。
    ``_provided_paths`` 记录 profile 显式提供的叶子路径，``_raw_runtime`` 保存脱离原输入的
    只读副本，另外两个私有字段仅用于诊断和来源追踪。直接构造实例表示使用代码默认值；
    外部 YAML 应始终通过 :meth:`from_mapping` 进入严格 schema 边界。
    """

    mode: str = "single_scene"
    profiles: RuntimeProfileSelection = field(default_factory=RuntimeProfileSelection)
    simulation_app: SimulationAppSettings = field(default_factory=SimulationAppSettings)
    execution: RuntimeExecutionSettings = field(
        default_factory=RuntimeExecutionSettings
    )
    interactive: InteractiveRuntimeSettings = field(
        default_factory=InteractiveRuntimeSettings
    )
    planner: RuntimePlannerSettings = field(default_factory=RuntimePlannerSettings)
    playback: PlaybackResourceSettings = field(default_factory=PlaybackResourceSettings)
    camera_output: CameraOutputRuntimeSettings = field(
        default_factory=CameraOutputRuntimeSettings
    )
    telemetry: TelemetryRuntimeSettings = field(
        default_factory=TelemetryRuntimeSettings
    )
    output: OutputPolicySettings = field(default_factory=OutputPolicySettings)
    paths: RuntimePathSettings = field(default_factory=RuntimePathSettings)
    shutdown: ShutdownSettings = field(default_factory=ShutdownSettings)
    _provided_paths: frozenset[str] = field(
        default_factory=frozenset, repr=False, compare=False
    )
    _raw_runtime: Mapping[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )
    _profile_name: str | None = field(default=None, repr=False, compare=False)
    _source_path: str | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
        *,
        profile_name: str | None = None,
        source_path: str | Path | None = None,
    ) -> "RuntimeProfileConfig":
        """从完整 YAML mapping 构造严格的运行时 profile。

        参数:
            data: 顶层只允许包含 ``runtime`` 的 mapping。
            profile_name: 可选稳定 profile 名，用于最终字段来源标记。
            source_path: 可选源文件路径，只参与错误信息和诊断元数据。
        返回:
            与输入数据脱离、字段已规范化且冻结的配置对象。
        异常:
            ValueError: 输入结构、字段类型、枚举、范围或跨字段组合不合法。
        副作用:
            无；不会读取依赖 profile，也不会启动任何运行时资源。
        """

        source = "<mapping>" if source_path is None else str(source_path)
        if not isinstance(data, Mapping):
            raise ValueError(f"{source}: runtime profile must be a mapping")
        canonical = dict(data)
        _reject_keys(canonical, {"runtime"}, "runtime profile")
        runtime = _copy_mapping(_mapping(canonical, "runtime", "runtime profile"))
        parsed = _parse_runtime(runtime)
        _require_tiled_scene_telemetry_selection(runtime, parsed)
        return cls(
            **{
                field_info.name: getattr(parsed, field_info.name)
                for field_info in fields(cls)
                if not field_info.name.startswith("_")
            },
            _provided_paths=frozenset(_flatten(runtime)),
            _raw_runtime=MappingProxyType(_copy_mapping(runtime)),
            _profile_name=profile_name,
            _source_path=None if source_path is None else str(source_path),
        )

    def as_dict(self) -> dict[str, object]:
        """返回可序列化的 ``{"runtime": ...}`` 深拷贝。

        私有来源元数据不会出现在结果中，tuple 会转换为 list，因此调用方修改返回值不会
        影响本配置实例。
        """

        return {"runtime": _dataclass_dict(self)}


@dataclass(frozen=True)
class ResolvedRuntimeConfig:
    """完成默认值、profile 和 CLI 合并后的最终运行时配置。

    ``config`` 是可供应用层消费的不可变配置；``sources`` 把每个
    ``runtime.<叶子路径>`` 映射到 ``default``、具体 profile 或 ``cli``；
    ``fingerprint`` 是最终有效配置规范 JSON 的 SHA-256，用于日志关联和复现实验。
    该对象在启动解析完成后生成一次，不应在运行期间重算或修改。
    """

    config: RuntimeProfileConfig
    sources: Mapping[str, str]
    fingerprint: str

    def as_dict(self) -> dict[str, object]:
        """返回最终有效配置的可序列化深拷贝，不包含来源和指纹。"""

        return self.config.as_dict()

    @property
    def mode(self) -> str:
        """返回当前入口模式，即 ``scene`` 或 ``tiled``。"""

        return self.config.mode

    @property
    def profiles(self) -> RuntimeProfileSelection:
        """返回最终选中的依赖 profile 名称集合。"""

        return self.config.profiles

    @property
    def simulation_app(self) -> SimulationAppSettings:
        """返回仿真应用启动配置。"""

        return self.config.simulation_app

    @property
    def execution(self) -> RuntimeExecutionSettings:
        """返回控制与物理步进配置。"""

        return self.config.execution

    @property
    def interactive(self) -> InteractiveRuntimeSettings:
        """返回标准输入、队列与传输配置。"""

        return self.config.interactive

    @property
    def planner(self) -> RuntimePlannerSettings:
        """返回已解析批容量的规划器配置。"""

        return self.config.planner

    @property
    def playback(self) -> PlaybackResourceSettings:
        """返回逐环境轨迹回放资源上限。"""

        return self.config.playback

    @property
    def camera_output(self) -> CameraOutputRuntimeSettings:
        """返回相机异步输出配置。"""

        return self.config.camera_output

    @property
    def telemetry(self) -> TelemetryRuntimeSettings:
        """返回遥测采样与 sink 配置。"""

        return self.config.telemetry

    @property
    def output(self) -> OutputPolicySettings:
        """返回通用输出文件的已存在数据策略。"""

        return self.config.output

    @property
    def paths(self) -> RuntimePathSettings:
        """返回进程级路径覆盖。"""

        return self.config.paths

    @property
    def shutdown(self) -> ShutdownSettings:
        """返回各异步子系统的关闭等待超时。"""

        return self.config.shutdown


def load_runtime_profile(name: str) -> RuntimeProfileConfig:
    """按稳定名称加载并严格解析项目内置运行时 profile。

    参数:
        name: ``configs/runtime`` 下不含扩展名的 profile 名称。
    返回:
        冻结的 :class:`RuntimeProfileConfig`。
    异常:
        FileNotFoundError: 对应 YAML 文件不存在。
        ValueError: profile 名称或 YAML 内容不符合当前 schema。
    副作用:
        只读取一个 YAML 文件，不加载 Isaac 或创建运行时资源。
    """

    from linkerbot_sim.configs.profiles import profile_path

    path = profile_path("runtime", name)
    return RuntimeProfileConfig.from_mapping(
        load_yaml(path),
        profile_name=name,
        source_path=path,
    )


def resolve_runtime_config(
    profile: RuntimeProfileConfig,
    *,
    cli_overrides: Mapping[str, object],
    env_config: Mapping[str, object],
    expected_mode: str | None = None,
) -> ResolvedRuntimeConfig:
    """合并默认值、严格 profile 与显式 CLI 覆盖并完成最终校验。

    参数:
        profile: 已经由严格 schema 解析的运行时 profile。
        cli_overrides: 点路径或嵌套 mapping 形式的显式覆盖；值为 ``None`` 的项被忽略。
        env_config: 已选择环境的配置，用于环境数量、遥测范围和规划批容量解析。
        expected_mode: 可选入口模式约束，用于阻止 ``single_scene`` / ``tiled_scene``
            profile 走错入口。
    返回:
        包含最终配置、逐字段来源和稳定 SHA-256 指纹的解析结果。
    异常:
        TypeError: 三个主要输入不是声明的配置或 mapping 类型。
        ValueError: 覆盖字段、交叉约束、环境范围或批容量不合法。
        FileNotFoundError: profile 引用的依赖配置不存在。
    副作用:
        会读取所选依赖 profile 以校验文件和 cuRobo 批容量；不启动仿真或写文件。
    """

    if not isinstance(profile, RuntimeProfileConfig):
        raise TypeError("profile must be RuntimeProfileConfig")
    if not isinstance(cli_overrides, Mapping):
        raise TypeError("cli_overrides must be a mapping")
    if not isinstance(env_config, Mapping):
        raise TypeError("env_config must be a mapping")

    defaults = _dataclass_dict(RuntimeProfileConfig())
    merged = _merge_declared(defaults, profile._raw_runtime)
    cli_overlay = _normalize_cli_overrides(cli_overrides)
    merged = _merge_declared(merged, cli_overlay)
    parsed = _parse_runtime(merged)

    if expected_mode is not None:
        normalized_expected = _enum(expected_mode, RUNTIME_MODES, "expected_mode")
        if parsed.mode != normalized_expected:
            raise ValueError(
                f"runtime.mode={parsed.mode!r} is incompatible with "
                f"{normalized_expected!r} entrypoint"
            )

    _validate_selected_profiles(parsed)
    parsed = _resolve_planner_batch_limit(parsed, env_config=env_config)
    _validate_telemetry_env(parsed, env_config=env_config)

    effective = parsed.as_dict()["runtime"]
    assert isinstance(effective, Mapping)
    sources = {f"runtime.{path}": "default" for path in _flatten(effective)}
    runtime_source = (
        "runtime"
        if profile._profile_name is None
        else f"runtime:{profile._profile_name}"
    )
    for path in profile._provided_paths:
        _mark_source_tree(sources, effective, path, runtime_source)
    for path in _flatten(cli_overlay):
        _mark_source_tree(sources, effective, path, "cli")

    payload = {"runtime": effective}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return ResolvedRuntimeConfig(
        config=parsed,
        sources=MappingProxyType(dict(sorted(sources.items()))),
        fingerprint=hashlib.sha256(encoded).hexdigest(),
    )


def _parse_runtime(
    data: Mapping[str, object], *, validate_cross_fields: bool = True
) -> RuntimeProfileConfig:
    """把完整 ``runtime`` mapping 严格转换为嵌套 dataclass。

    每一层先拒绝未知键，再用不做隐式强制转换的标量解析器读取字段。仅在校验 CLI 的局部
    overlay 时暂时关闭跨字段检查，因为该 overlay 尚未与 profile 和环境上下文合并。
    """

    _reject_keys(data, _RUNTIME_KEYS, "runtime")
    defaults = RuntimeProfileConfig()

    profiles = _section(data, "profiles", "runtime")
    _reject_keys(
        profiles, {"env", "curobo", "logging", "controller_bundle"}, "runtime.profiles"
    )
    profile_defaults = defaults.profiles

    simulation = _section(data, "simulation_app", "runtime")
    _reject_keys(simulation, {"gui", "gpu", "render"}, "runtime.simulation_app")
    gpu = _section(simulation, "gpu", "runtime.simulation_app")
    _reject_keys(
        gpu,
        {"multi_gpu", "max_gpu_count", "active_gpu", "physics_gpu"},
        "runtime.simulation_app.gpu",
    )
    gpu_defaults = defaults.simulation_app.gpu
    render = _section(simulation, "render", "runtime.simulation_app")
    _reject_keys(render, _RENDER_KEYS, "runtime.simulation_app.render")
    render_defaults = defaults.simulation_app.render

    execution = _section(data, "execution", "runtime")
    _reject_keys(
        execution,
        {
            "control_mode",
            "idle_physics_policy",
            "idle_step_duration_s",
            "default_decimation",
            "command_defaults",
        },
        "runtime.execution",
    )
    commands = _section(execution, "command_defaults", "runtime.execution")
    _reject_keys(
        commands,
        {"joint_interpolation", "pose_frame", "orientation_mode"},
        "runtime.execution.command_defaults",
    )
    execution_defaults = defaults.execution

    interactive = _section(data, "interactive", "runtime")
    _reject_keys(
        interactive,
        {
            "stdin_enabled",
            "stdin_eof_policy",
            "queue_poll_timeout_s",
            "snapshot_timeout_s",
            "command_history_capacity",
            "snapshot_request_capacity",
            "transport",
        },
        "runtime.interactive",
    )
    transport = _section(interactive, "transport", "runtime.interactive")
    _reject_keys(transport, _TRANSPORT_KEYS, "runtime.interactive.transport")
    tcp = _section(transport, "tcp_jsonl", "runtime.interactive.transport")
    ws = _section(transport, "websocket", "runtime.interactive.transport")
    _reject_keys(
        tcp, {"enabled", "host", "port"}, "runtime.interactive.transport.tcp_jsonl"
    )
    _reject_keys(
        ws, {"enabled", "host", "port"}, "runtime.interactive.transport.websocket"
    )
    interactive_defaults = defaults.interactive
    transport_defaults = interactive_defaults.transport

    planner = _section(data, "planner", "runtime")
    _reject_keys(
        planner,
        {
            "backend",
            "joint_batch_mode",
            "request_defaults",
            "oversize_request_policy",
            "failure_policy",
            "resources",
        },
        "runtime.planner",
    )
    requests = _section(planner, "request_defaults", "runtime.planner")
    _reject_keys(
        requests,
        {
            "duration_s",
            "avoid_collisions",
            "force_collision_refresh",
            "coordination",
            "load_on_success",
            "replace",
        },
        "runtime.planner.request_defaults",
    )
    resources = _section(planner, "resources", "runtime.planner")
    _reject_keys(
        resources,
        {
            "max_workers",
            "max_pending_requests",
            "max_completed_results",
            "max_batch_problems",
            "shutdown_timeout_s",
        },
        "runtime.planner.resources",
    )
    planner_defaults = defaults.planner

    playback = _section(data, "playback", "runtime")
    _reject_keys(
        playback,
        {
            "max_queue_depth_per_env",
            "max_samples_per_env",
            "max_duration_s_per_env",
            "overflow_policy",
        },
        "runtime.playback",
    )
    camera = _section(data, "camera_output", "runtime")
    if "shutdown_timeout_s" in camera:
        raise ValueError(
            "runtime.camera_output.shutdown_timeout_s moved to "
            "runtime.shutdown.camera_publisher_timeout_s; configure the single "
            "publisher join timeout owner there"
        )
    _reject_keys(
        camera,
        {
            "queue_size",
            "overflow_policy",
            "worker_poll_interval_s",
            "existing_data_policy",
            "shutdown_policy",
            "rgb_format",
            "depth_format",
            "metadata_flush_interval_frames",
            "max_bytes_per_camera",
        },
        "runtime.camera_output",
    )

    telemetry = _section(data, "telemetry", "runtime")
    _reject_keys(
        telemetry,
        {
            "primary_env_id",
            "selected_env_ids",
            "publish_decimation",
            "rate_hz",
            "buffer_size",
            "drop_policy",
            "on_error",
            "include_joint_states",
            "include_state_json",
            "include_scene_markers",
            "include_efforts",
            "include_objects",
            "joint_effort_field",
            "topics",
            "mcap",
            "foxglove_live",
        },
        "runtime.telemetry",
    )
    topics = _section(telemetry, "topics", "runtime.telemetry")
    mcap = _section(telemetry, "mcap", "runtime.telemetry")
    live = _section(telemetry, "foxglove_live", "runtime.telemetry")
    _reject_keys(topics, {"joint_states", "scene", "state"}, "runtime.telemetry.topics")
    _reject_keys(mcap, {"path"}, "runtime.telemetry.mcap")
    _reject_keys(live, {"enabled", "host", "port"}, "runtime.telemetry.foxglove_live")
    telemetry_defaults = defaults.telemetry

    output = _section(data, "output", "runtime")
    _reject_keys(
        output,
        {"csv_existing_file_policy", "mcap_existing_file_policy"},
        "runtime.output",
    )
    paths = _section(data, "paths", "runtime")
    _reject_keys(paths, {"cache_root"}, "runtime.paths")
    shutdown = _section(data, "shutdown", "runtime")
    _reject_keys(
        shutdown,
        {
            "state_publisher_timeout_s",
            "camera_publisher_timeout_s",
            "transport_timeout_s",
        },
        "runtime.shutdown",
    )

    result = RuntimeProfileConfig(
        mode=_enum(data.get("mode", defaults.mode), RUNTIME_MODES, "runtime.mode"),
        profiles=RuntimeProfileSelection(
            env=_profile_name(
                profiles.get("env", profile_defaults.env), "runtime.profiles.env"
            ),
            curobo=_profile_name(
                profiles.get("curobo", profile_defaults.curobo),
                "runtime.profiles.curobo",
            ),
            logging=_profile_name(
                profiles.get("logging", profile_defaults.logging),
                "runtime.profiles.logging",
            ),
            controller_bundle=_profile_name(
                profiles.get("controller_bundle", profile_defaults.controller_bundle),
                "runtime.profiles.controller_bundle",
            ),
        ),
        simulation_app=SimulationAppSettings(
            gui=_bool(
                simulation.get("gui", defaults.simulation_app.gui),
                "runtime.simulation_app.gui",
            ),
            gpu=SimulationGpuSettings(
                multi_gpu=_bool(
                    gpu.get("multi_gpu", gpu_defaults.multi_gpu),
                    "runtime.simulation_app.gpu.multi_gpu",
                ),
                max_gpu_count=_positive_int(
                    gpu.get("max_gpu_count", gpu_defaults.max_gpu_count),
                    "runtime.simulation_app.gpu.max_gpu_count",
                ),
                active_gpu=_nonnegative_int(
                    gpu.get("active_gpu", gpu_defaults.active_gpu),
                    "runtime.simulation_app.gpu.active_gpu",
                ),
                physics_gpu=_nonnegative_int(
                    gpu.get("physics_gpu", gpu_defaults.physics_gpu),
                    "runtime.simulation_app.gpu.physics_gpu",
                ),
            ),
            render=SimulationRenderSettings(
                gui_size=_size(
                    render.get("gui_size", render_defaults.gui_size),
                    "runtime.simulation_app.render.gui_size",
                ),
                headless_size=_size(
                    render.get("headless_size", render_defaults.headless_size),
                    "runtime.simulation_app.render.headless_size",
                ),
                window_size=_size(
                    render.get("window_size", render_defaults.window_size),
                    "runtime.simulation_app.render.window_size",
                ),
                renderer=_nonempty_str(
                    render.get("renderer", render_defaults.renderer),
                    "runtime.simulation_app.render.renderer",
                ),
                anti_aliasing_gui=_nonnegative_int(
                    render.get("anti_aliasing_gui", render_defaults.anti_aliasing_gui),
                    "runtime.simulation_app.render.anti_aliasing_gui",
                ),
                anti_aliasing_headless=_nonnegative_int(
                    render.get(
                        "anti_aliasing_headless", render_defaults.anti_aliasing_headless
                    ),
                    "runtime.simulation_app.render.anti_aliasing_headless",
                ),
                samples_per_pixel_per_frame=_positive_int(
                    render.get(
                        "samples_per_pixel_per_frame",
                        render_defaults.samples_per_pixel_per_frame,
                    ),
                    "runtime.simulation_app.render.samples_per_pixel_per_frame",
                ),
                denoiser=_bool(
                    render.get("denoiser", render_defaults.denoiser),
                    "runtime.simulation_app.render.denoiser",
                ),
                hide_ui=_nullable_bool(
                    render.get("hide_ui", render_defaults.hide_ui),
                    "runtime.simulation_app.render.hide_ui",
                ),
                disable_viewport_updates=_nullable_bool(
                    render.get(
                        "disable_viewport_updates",
                        render_defaults.disable_viewport_updates,
                    ),
                    "runtime.simulation_app.render.disable_viewport_updates",
                ),
                fast_shutdown=_nullable_bool(
                    render.get("fast_shutdown", render_defaults.fast_shutdown),
                    "runtime.simulation_app.render.fast_shutdown",
                ),
                material_sync_loads=_bool(
                    render.get(
                        "material_sync_loads", render_defaults.material_sync_loads
                    ),
                    "runtime.simulation_app.render.material_sync_loads",
                ),
                hydra_material_sync_loads=_bool(
                    render.get(
                        "hydra_material_sync_loads",
                        render_defaults.hydra_material_sync_loads,
                    ),
                    "runtime.simulation_app.render.hydra_material_sync_loads",
                ),
                headless_dt_policy=_enum(
                    render.get(
                        "headless_dt_policy", render_defaults.headless_dt_policy
                    ),
                    {"camera_aware", "physics"},
                    "runtime.simulation_app.render.headless_dt_policy",
                ),
            ),
        ),
        execution=RuntimeExecutionSettings(
            control_mode=_enum(
                execution.get("control_mode", execution_defaults.control_mode),
                {"position", "velocity", "effort"},
                "runtime.execution.control_mode",
            ),
            idle_physics_policy=_enum(
                execution.get(
                    "idle_physics_policy", execution_defaults.idle_physics_policy
                ),
                {"pause", "hold_step"},
                "runtime.execution.idle_physics_policy",
            ),
            idle_step_duration_s=_positive_float(
                execution.get(
                    "idle_step_duration_s", execution_defaults.idle_step_duration_s
                ),
                "runtime.execution.idle_step_duration_s",
            ),
            default_decimation=_positive_int(
                execution.get(
                    "default_decimation", execution_defaults.default_decimation
                ),
                "runtime.execution.default_decimation",
            ),
            command_defaults=RuntimeCommandDefaults(
                joint_interpolation=_enum(
                    commands.get(
                        "joint_interpolation",
                        execution_defaults.command_defaults.joint_interpolation,
                    ),
                    {"linear", "smoothstep"},
                    "runtime.execution.command_defaults.joint_interpolation",
                ),
                pose_frame=_enum(
                    commands.get(
                        "pose_frame", execution_defaults.command_defaults.pose_frame
                    ),
                    {"env", "world"},
                    "runtime.execution.command_defaults.pose_frame",
                ),
                orientation_mode=_enum(
                    commands.get(
                        "orientation_mode",
                        execution_defaults.command_defaults.orientation_mode,
                    ),
                    {"free", "current", "target"},
                    "runtime.execution.command_defaults.orientation_mode",
                ),
            ),
        ),
        interactive=InteractiveRuntimeSettings(
            stdin_enabled=_bool(
                interactive.get("stdin_enabled", interactive_defaults.stdin_enabled),
                "runtime.interactive.stdin_enabled",
            ),
            stdin_eof_policy=_enum(
                interactive.get(
                    "stdin_eof_policy", interactive_defaults.stdin_eof_policy
                ),
                {"exit", "keep_alive"},
                "runtime.interactive.stdin_eof_policy",
            ),
            queue_poll_timeout_s=_positive_float(
                interactive.get(
                    "queue_poll_timeout_s", interactive_defaults.queue_poll_timeout_s
                ),
                "runtime.interactive.queue_poll_timeout_s",
            ),
            snapshot_timeout_s=_positive_float(
                interactive.get(
                    "snapshot_timeout_s", interactive_defaults.snapshot_timeout_s
                ),
                "runtime.interactive.snapshot_timeout_s",
            ),
            command_history_capacity=_nonnegative_int(
                interactive.get(
                    "command_history_capacity",
                    interactive_defaults.command_history_capacity,
                ),
                "runtime.interactive.command_history_capacity",
            ),
            snapshot_request_capacity=_positive_int(
                interactive.get(
                    "snapshot_request_capacity",
                    interactive_defaults.snapshot_request_capacity,
                ),
                "runtime.interactive.snapshot_request_capacity",
            ),
            transport=RuntimeTransportSettings(
                tcp_jsonl=_parse_endpoint(
                    tcp,
                    transport_defaults.tcp_jsonl,
                    "runtime.interactive.transport.tcp_jsonl",
                ),
                websocket=_parse_endpoint(
                    ws,
                    transport_defaults.websocket,
                    "runtime.interactive.transport.websocket",
                ),
                max_message_bytes=_positive_int(
                    transport.get(
                        "max_message_bytes", transport_defaults.max_message_bytes
                    ),
                    "runtime.interactive.transport.max_message_bytes",
                ),
                max_connections=_positive_int(
                    transport.get(
                        "max_connections", transport_defaults.max_connections
                    ),
                    "runtime.interactive.transport.max_connections",
                ),
                request_queue_capacity=_positive_int(
                    transport.get(
                        "request_queue_capacity",
                        transport_defaults.request_queue_capacity,
                    ),
                    "runtime.interactive.transport.request_queue_capacity",
                ),
                event_queue_capacity=_positive_int(
                    transport.get(
                        "event_queue_capacity", transport_defaults.event_queue_capacity
                    ),
                    "runtime.interactive.transport.event_queue_capacity",
                ),
                overflow_policy=_enum(
                    transport.get(
                        "overflow_policy", transport_defaults.overflow_policy
                    ),
                    {"reject"},
                    "runtime.interactive.transport.overflow_policy",
                ),
                startup_timeout_s=_positive_float(
                    transport.get(
                        "startup_timeout_s", transport_defaults.startup_timeout_s
                    ),
                    "runtime.interactive.transport.startup_timeout_s",
                ),
                server_poll_interval_s=_positive_float(
                    transport.get(
                        "server_poll_interval_s",
                        transport_defaults.server_poll_interval_s,
                    ),
                    "runtime.interactive.transport.server_poll_interval_s",
                ),
                response_poll_interval_s=_positive_float(
                    transport.get(
                        "response_poll_interval_s",
                        transport_defaults.response_poll_interval_s,
                    ),
                    "runtime.interactive.transport.response_poll_interval_s",
                ),
            ),
        ),
        planner=RuntimePlannerSettings(
            backend=_enum(
                planner.get("backend", planner_defaults.backend),
                {"curobo", "linear"},
                "runtime.planner.backend",
            ),
            joint_batch_mode=_enum(
                planner.get("joint_batch_mode", planner_defaults.joint_batch_mode),
                {"auto", "per_env", "batch_only"},
                "runtime.planner.joint_batch_mode",
            ),
            request_defaults=PlannerRequestDefaults(
                duration_s=_positive_float(
                    requests.get(
                        "duration_s", planner_defaults.request_defaults.duration_s
                    ),
                    "runtime.planner.request_defaults.duration_s",
                ),
                avoid_collisions=_bool(
                    requests.get(
                        "avoid_collisions",
                        planner_defaults.request_defaults.avoid_collisions,
                    ),
                    "runtime.planner.request_defaults.avoid_collisions",
                ),
                force_collision_refresh=_bool(
                    requests.get(
                        "force_collision_refresh",
                        planner_defaults.request_defaults.force_collision_refresh,
                    ),
                    "runtime.planner.request_defaults.force_collision_refresh",
                ),
                coordination=_enum(
                    requests.get(
                        "coordination", planner_defaults.request_defaults.coordination
                    ),
                    {"independent", "static_others", "coupled"},
                    "runtime.planner.request_defaults.coordination",
                ),
                load_on_success=_bool(
                    requests.get(
                        "load_on_success",
                        planner_defaults.request_defaults.load_on_success,
                    ),
                    "runtime.planner.request_defaults.load_on_success",
                ),
                replace=_bool(
                    requests.get("replace", planner_defaults.request_defaults.replace),
                    "runtime.planner.request_defaults.replace",
                ),
            ),
            oversize_request_policy=_enum(
                planner.get(
                    "oversize_request_policy", planner_defaults.oversize_request_policy
                ),
                {"split", "reject"},
                "runtime.planner.oversize_request_policy",
            ),
            failure_policy=_planner_failure_policy(
                planner.get("failure_policy", planner_defaults.failure_policy),
                "runtime.planner.failure_policy",
            ),
            resources=PlannerResourceSettings(
                max_workers=_positive_int(
                    resources.get(
                        "max_workers", planner_defaults.resources.max_workers
                    ),
                    "runtime.planner.resources.max_workers",
                ),
                max_pending_requests=_positive_int(
                    resources.get(
                        "max_pending_requests",
                        planner_defaults.resources.max_pending_requests,
                    ),
                    "runtime.planner.resources.max_pending_requests",
                ),
                max_completed_results=_nonnegative_int(
                    resources.get(
                        "max_completed_results",
                        planner_defaults.resources.max_completed_results,
                    ),
                    "runtime.planner.resources.max_completed_results",
                ),
                max_batch_problems=_positive_int_or_auto(
                    resources.get(
                        "max_batch_problems",
                        planner_defaults.resources.max_batch_problems,
                    ),
                    "runtime.planner.resources.max_batch_problems",
                ),
                shutdown_timeout_s=_positive_float(
                    resources.get(
                        "shutdown_timeout_s",
                        planner_defaults.resources.shutdown_timeout_s,
                    ),
                    "runtime.planner.resources.shutdown_timeout_s",
                ),
            ),
        ),
        playback=PlaybackResourceSettings(
            max_queue_depth_per_env=_positive_int(
                playback.get(
                    "max_queue_depth_per_env", defaults.playback.max_queue_depth_per_env
                ),
                "runtime.playback.max_queue_depth_per_env",
            ),
            max_samples_per_env=_positive_int(
                playback.get(
                    "max_samples_per_env", defaults.playback.max_samples_per_env
                ),
                "runtime.playback.max_samples_per_env",
            ),
            max_duration_s_per_env=_positive_float(
                playback.get(
                    "max_duration_s_per_env",
                    defaults.playback.max_duration_s_per_env,
                ),
                "runtime.playback.max_duration_s_per_env",
            ),
            overflow_policy=_enum(
                playback.get("overflow_policy", defaults.playback.overflow_policy),
                {"reject"},
                "runtime.playback.overflow_policy",
            ),
        ),
        camera_output=CameraOutputRuntimeSettings(
            queue_size=_positive_int(
                camera.get("queue_size", defaults.camera_output.queue_size),
                "runtime.camera_output.queue_size",
            ),
            overflow_policy=_enum(
                camera.get("overflow_policy", defaults.camera_output.overflow_policy),
                {"drop_oldest", "drop_newest", "block", "error"},
                "runtime.camera_output.overflow_policy",
            ),
            worker_poll_interval_s=_positive_float(
                camera.get(
                    "worker_poll_interval_s",
                    defaults.camera_output.worker_poll_interval_s,
                ),
                "runtime.camera_output.worker_poll_interval_s",
            ),
            existing_data_policy=_enum(
                camera.get(
                    "existing_data_policy", defaults.camera_output.existing_data_policy
                ),
                EXISTING_DATA_POLICIES,
                "runtime.camera_output.existing_data_policy",
            ),
            shutdown_policy=_enum(
                camera.get("shutdown_policy", defaults.camera_output.shutdown_policy),
                {"drain", "abort"},
                "runtime.camera_output.shutdown_policy",
            ),
            rgb_format=_enum(
                camera.get("rgb_format", defaults.camera_output.rgb_format),
                {"ppm", "png", "npy"},
                "runtime.camera_output.rgb_format",
            ),
            depth_format=_enum(
                camera.get("depth_format", defaults.camera_output.depth_format),
                {"npy", "npz"},
                "runtime.camera_output.depth_format",
            ),
            metadata_flush_interval_frames=_positive_int(
                camera.get(
                    "metadata_flush_interval_frames",
                    defaults.camera_output.metadata_flush_interval_frames,
                ),
                "runtime.camera_output.metadata_flush_interval_frames",
            ),
            max_bytes_per_camera=_positive_int(
                camera.get(
                    "max_bytes_per_camera",
                    defaults.camera_output.max_bytes_per_camera,
                ),
                "runtime.camera_output.max_bytes_per_camera",
            ),
        ),
        telemetry=TelemetryRuntimeSettings(
            primary_env_id=_nonnegative_int(
                telemetry.get("primary_env_id", telemetry_defaults.primary_env_id),
                "runtime.telemetry.primary_env_id",
            ),
            selected_env_ids=_env_id_tuple(
                telemetry.get("selected_env_ids", telemetry_defaults.selected_env_ids),
                "runtime.telemetry.selected_env_ids",
            ),
            publish_decimation=_positive_int(
                telemetry.get(
                    "publish_decimation", telemetry_defaults.publish_decimation
                ),
                "runtime.telemetry.publish_decimation",
            ),
            rate_hz=_nonnegative_float(
                telemetry.get("rate_hz", telemetry_defaults.rate_hz),
                "runtime.telemetry.rate_hz",
            ),
            buffer_size=_positive_int(
                telemetry.get("buffer_size", telemetry_defaults.buffer_size),
                "runtime.telemetry.buffer_size",
            ),
            drop_policy=_enum(
                telemetry.get("drop_policy", telemetry_defaults.drop_policy),
                {"latest", "drop_oldest", "drop_newest"},
                "runtime.telemetry.drop_policy",
            ),
            on_error=_enum(
                telemetry.get("on_error", telemetry_defaults.on_error),
                {"stop", "continue"},
                "runtime.telemetry.on_error",
            ),
            include_joint_states=_bool(
                telemetry.get(
                    "include_joint_states", telemetry_defaults.include_joint_states
                ),
                "runtime.telemetry.include_joint_states",
            ),
            include_state_json=_bool(
                telemetry.get(
                    "include_state_json", telemetry_defaults.include_state_json
                ),
                "runtime.telemetry.include_state_json",
            ),
            include_scene_markers=_bool(
                telemetry.get(
                    "include_scene_markers",
                    telemetry_defaults.include_scene_markers,
                ),
                "runtime.telemetry.include_scene_markers",
            ),
            include_efforts=_bool(
                telemetry.get("include_efforts", telemetry_defaults.include_efforts),
                "runtime.telemetry.include_efforts",
            ),
            include_objects=_bool(
                telemetry.get("include_objects", telemetry_defaults.include_objects),
                "runtime.telemetry.include_objects",
            ),
            joint_effort_field=_enum(
                telemetry.get(
                    "joint_effort_field", telemetry_defaults.joint_effort_field
                ),
                {"none", "commanded", "measured", "applied"},
                "runtime.telemetry.joint_effort_field",
            ),
            topics=TelemetryTopicSettings(
                joint_states=_topic(
                    topics.get("joint_states", telemetry_defaults.topics.joint_states),
                    "runtime.telemetry.topics.joint_states",
                ),
                scene=_topic(
                    topics.get("scene", telemetry_defaults.topics.scene),
                    "runtime.telemetry.topics.scene",
                ),
                state=_topic(
                    topics.get("state", telemetry_defaults.topics.state),
                    "runtime.telemetry.topics.state",
                ),
            ),
            mcap=McapRuntimeSettings(
                path=_nullable_path(
                    mcap.get("path", telemetry_defaults.mcap.path),
                    "runtime.telemetry.mcap.path",
                ),
            ),
            foxglove_live=FoxgloveLiveRuntimeSettings(
                enabled=_bool(
                    live.get("enabled", telemetry_defaults.foxglove_live.enabled),
                    "runtime.telemetry.foxglove_live.enabled",
                ),
                host=_loopback_host(
                    live.get("host", telemetry_defaults.foxglove_live.host),
                    "runtime.telemetry.foxglove_live.host",
                ),
                port=_nullable_port(
                    live.get("port", telemetry_defaults.foxglove_live.port),
                    "runtime.telemetry.foxglove_live.port",
                ),
            ),
        ),
        output=OutputPolicySettings(
            csv_existing_file_policy=_enum(
                output.get(
                    "csv_existing_file_policy", defaults.output.csv_existing_file_policy
                ),
                EXISTING_DATA_POLICIES,
                "runtime.output.csv_existing_file_policy",
            ),
            mcap_existing_file_policy=_enum(
                output.get(
                    "mcap_existing_file_policy",
                    defaults.output.mcap_existing_file_policy,
                ),
                EXISTING_DATA_POLICIES,
                "runtime.output.mcap_existing_file_policy",
            ),
        ),
        paths=RuntimePathSettings(
            cache_root=_nullable_path(
                paths.get("cache_root", defaults.paths.cache_root),
                "runtime.paths.cache_root",
            ),
        ),
        shutdown=ShutdownSettings(
            state_publisher_timeout_s=_positive_float(
                shutdown.get(
                    "state_publisher_timeout_s",
                    defaults.shutdown.state_publisher_timeout_s,
                ),
                "runtime.shutdown.state_publisher_timeout_s",
            ),
            camera_publisher_timeout_s=_positive_float(
                shutdown.get(
                    "camera_publisher_timeout_s",
                    defaults.shutdown.camera_publisher_timeout_s,
                ),
                "runtime.shutdown.camera_publisher_timeout_s",
            ),
            transport_timeout_s=_positive_float(
                shutdown.get(
                    "transport_timeout_s", defaults.shutdown.transport_timeout_s
                ),
                "runtime.shutdown.transport_timeout_s",
            ),
        ),
    )
    if validate_cross_fields:
        _validate_cross_fields(result)
    return result


def _require_tiled_scene_telemetry_selection(
    runtime: Mapping[str, object], config: RuntimeProfileConfig
) -> None:
    """要求 ``tiled_scene`` profile 显式声明遥测环境范围。

    这些字段不能依赖 ``single_scene`` 模式的单环境默认值，否则增加 ``num_envs`` 后会
    静默改变数据覆盖范围；因此检查原始 profile 是否真正提供了键，而不只检查解析后的值。
    """

    if config.mode != "tiled_scene":
        return
    telemetry = runtime.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError("runtime.telemetry must be a mapping in tiled_scene mode")
    for key in ("primary_env_id", "selected_env_ids"):
        if key not in telemetry:
            raise ValueError(f"runtime.telemetry.{key} is required in tiled_scene mode")


def _validate_cross_fields(config: RuntimeProfileConfig) -> None:
    """校验单字段解析无法表达的模式、后端和输出组合不变量。"""

    if config.mode == "tiled_scene" and config.execution.control_mode != "position":
        raise ValueError(
            "runtime.execution.control_mode must be 'position' in tiled_scene mode; "
            "the tiled articulation runtime does not implement velocity or effort actions"
        )
    request_defaults = config.planner.request_defaults
    if request_defaults.coordination == "coupled":
        raise ValueError(
            "runtime.planner.request_defaults.coordination='coupled' is unsupported "
            "because no coupled planner backend is configured"
        )
    if config.planner.backend == "linear" and request_defaults.avoid_collisions:
        raise ValueError(
            "runtime.planner.backend='linear' cannot be combined with "
            "runtime.planner.request_defaults.avoid_collisions=true because the "
            "linear backend has no collision checker"
        )
    if config.mode == "tiled_scene" and request_defaults.coordination != "independent":
        raise ValueError(
            "runtime.planner.request_defaults.coordination must be 'independent' in "
            "tiled_scene mode; async tiled planning is request-atomic and does not build "
            "cross-robot coordination snapshots"
        )
    if config.mode == "tiled_scene" and request_defaults.force_collision_refresh:
        raise ValueError(
            "runtime.planner.request_defaults.force_collision_refresh is unsupported "
            "in tiled_scene mode because each async planner request creates an isolated "
            "context"
        )
    gpu = config.simulation_app.gpu
    if gpu.active_gpu >= gpu.max_gpu_count:
        raise ValueError(
            "runtime.simulation_app.gpu.active_gpu must be below max_gpu_count"
        )
    if gpu.physics_gpu >= gpu.max_gpu_count:
        raise ValueError(
            "runtime.simulation_app.gpu.physics_gpu must be below max_gpu_count"
        )
    for label, endpoint in (
        (
            "runtime.interactive.transport.tcp_jsonl",
            config.interactive.transport.tcp_jsonl,
        ),
        (
            "runtime.interactive.transport.websocket",
            config.interactive.transport.websocket,
        ),
        ("runtime.telemetry.foxglove_live", config.telemetry.foxglove_live),
    ):
        if endpoint.enabled and endpoint.port is None:
            raise ValueError(f"{label}.port is required when enabled is true")
    telemetry = config.telemetry
    if config.mode == "single_scene" and (
        telemetry.selected_env_ids != (0,) or telemetry.publish_decimation != 1
    ):
        raise ValueError(
            "runtime.telemetry.selected_env_ids and publish_decimation are only "
            "supported in tiled_scene mode"
        )
    if config.mode == "tiled_scene" and telemetry.joint_effort_field != "none":
        raise ValueError(
            "runtime.telemetry.joint_effort_field is only supported in single_scene mode"
        )
    if telemetry.joint_effort_field != "none" and not telemetry.include_efforts:
        raise ValueError(
            "runtime.telemetry.joint_effort_field requires include_efforts=true"
        )
    topic_values = (
        telemetry.topics.joint_states,
        telemetry.topics.scene,
        telemetry.topics.state,
    )
    if len(set(topic_values)) != len(topic_values):
        raise ValueError("runtime.telemetry.topics must use distinct topic paths")
    if (telemetry.foxglove_live.enabled or telemetry.mcap.path is not None) and not (
        telemetry.include_joint_states
        or telemetry.include_state_json
        or telemetry.include_scene_markers
    ):
        raise ValueError(
            "runtime.telemetry must enable at least one output modality when an "
            "output sink is configured"
        )
    if (
        config.mode == "single_scene"
        and (telemetry.foxglove_live.enabled or telemetry.mcap.path is not None)
        and telemetry.include_scene_markers
        and not telemetry.include_objects
    ):
        raise ValueError(
            "runtime.telemetry.include_scene_markers requires include_objects=true in "
            "single_scene mode because Scene markers are sampled from runtime objects"
        )


def _validate_selected_profiles(config: RuntimeProfileConfig) -> None:
    """确认运行时引用的项目 profile 存在，并兼容目录式环境布局。"""

    from linkerbot_sim.configs.profiles import profile_path

    for group, name in (
        ("env", config.profiles.env),
        ("curobo", config.profiles.curobo),
        ("logging", config.profiles.logging),
    ):
        path = profile_path(group, name)
        if group == "env" and not path.is_file():
            directory_base = path.parent / name / "base.yaml"
            if directory_base.is_file():
                continue
        if not path.is_file():
            raise FileNotFoundError(f"selected {group} profile not found: {name!r}")


def _validate_telemetry_env(
    config: RuntimeProfileConfig,
    *,
    env_config: Mapping[str, object],
) -> None:
    """校验 ``tiled_scene`` standard topic 的单一 primary env 数据源。"""

    if config.mode != "tiled_scene":
        return
    num_envs = _configured_num_envs(env_config)
    if config.telemetry.primary_env_id not in config.telemetry.selected_env_ids:
        raise ValueError(
            "runtime.telemetry.primary_env_id must be included in selected_env_ids"
        )
    if config.telemetry.primary_env_id >= num_envs:
        raise ValueError(
            "runtime.telemetry.primary_env_id must be below tiled.num_envs "
            f"({num_envs})"
        )
    if any(env_id >= num_envs for env_id in config.telemetry.selected_env_ids):
        raise ValueError(
            "runtime.telemetry.selected_env_ids must be below tiled.num_envs "
            f"({num_envs})"
        )


def _resolve_planner_batch_limit(
    config: RuntimeProfileConfig,
    *,
    env_config: Mapping[str, object],
) -> RuntimeProfileConfig:
    """把 auto 推导为 int，并按所选 cuRobo profile 的最小相关容量校验。"""

    configured = config.planner.resources.max_batch_problems
    num_envs = _configured_num_envs(env_config)
    capacities: list[int] = []
    if config.planner.backend == "curobo":
        capacities = _selected_curobo_batch_capacities(config)
    capacity = min(capacities) if capacities else None
    if configured == "auto":
        resolved = min(num_envs, capacity) if capacity is not None else num_envs
    else:
        resolved = _positive_int(
            configured, "runtime.planner.resources.max_batch_problems"
        )
        if capacity is not None and resolved > capacity:
            raise ValueError(
                "runtime.planner.resources.max_batch_problems cannot exceed "
                f"selected cuRobo max_batch_size ({capacity})"
            )
    resources = replace(config.planner.resources, max_batch_problems=resolved)
    return replace(config, planner=replace(config.planner, resources=resources))


def _selected_curobo_batch_capacities(
    config: RuntimeProfileConfig,
) -> list[int]:
    """读取会约束 tiled 同步 IK/异步 motion batch 的全部 cuRobo caps。"""

    from linkerbot_sim.backends.curobo.config import (
        CuroboIkConfig,
        CuroboMotionPlannerConfig,
    )
    from linkerbot_sim.configs.profiles import load_profile_yaml

    data = load_profile_yaml("curobo", config.profiles.curobo)
    curobo = data.get("curobo")
    if not isinstance(curobo, Mapping):
        raise ValueError("selected curobo profile must contain curobo mapping")
    kinematics = curobo.get("kinematics")
    if kinematics is not None and not isinstance(kinematics, Mapping):
        raise ValueError("curobo.kinematics must be a mapping")
    ik_data = kinematics.get("ik") if isinstance(kinematics, Mapping) else None
    motion_data = curobo.get("motion_planner")
    if ik_data is not None and not isinstance(ik_data, Mapping):
        raise ValueError("curobo.kinematics.ik must be a mapping")
    if motion_data is not None and not isinstance(motion_data, Mapping):
        raise ValueError("curobo.motion_planner must be a mapping")
    ik = CuroboIkConfig.from_mapping(ik_data if isinstance(ik_data, Mapping) else None)
    motion = CuroboMotionPlannerConfig.from_mapping(
        motion_data if isinstance(motion_data, Mapping) else None
    )
    return [int(ik.max_batch_size), int(motion.max_batch_size)]


def _configured_num_envs(env_config: Mapping[str, object]) -> int:
    """返回 runtime 实际 tiled env 数；非 tiled scene 按单 env 处理。"""

    tiled = env_config.get("tiled")
    if not isinstance(tiled, Mapping):
        return 1
    return _positive_int(tiled.get("num_envs", 1), "tiled.num_envs")


def _normalize_cli_overrides(values: Mapping[str, object]) -> dict[str, object]:
    """把嵌套或点路径 CLI 覆盖统一为 ``runtime`` 子树。

    规范化后立即与代码默认值合并并走一次字段级解析，以便在启动读取环境前就报告拼写、
    类型和路径冲突；涉及其它字段的组合约束留给最终合并阶段。
    """

    overlay: dict[str, object] = {}
    for key, value in values.items():
        if value is None:
            continue
        path = str(key)
        if path.startswith("runtime."):
            path = path.removeprefix("runtime.")
        if isinstance(value, Mapping) and "." not in path:
            current = overlay.get(path)
            if current is not None and not isinstance(current, Mapping):
                raise ValueError(f"conflicting CLI override for runtime.{path}")
            overlay[path] = _merge_declared(
                {} if current is None else current,
                value,
            )
        else:
            _set_dotted(overlay, path, _copy_value(value))
    _parse_runtime(
        _merge_declared(_dataclass_dict(RuntimeProfileConfig()), overlay),
        validate_cross_fields=False,
    )
    return overlay


def _parse_endpoint(
    data: Mapping[str, object], defaults: RuntimeTransportEndpoint, label: str
) -> RuntimeTransportEndpoint:
    return RuntimeTransportEndpoint(
        enabled=_bool(data.get("enabled", defaults.enabled), f"{label}.enabled"),
        host=_loopback_host(data.get("host", defaults.host), f"{label}.host"),
        port=_nullable_port(data.get("port", defaults.port), f"{label}.port"),
    )


def _dataclass_dict(value: object) -> dict[str, object]:
    """递归导出公开 dataclass 字段，并把 tuple 转成 JSON 友好的 list。"""

    if not is_dataclass(value):
        raise TypeError("expected dataclass instance")
    result: dict[str, object] = {}
    for field_info in fields(value):
        if field_info.name.startswith("_"):
            continue
        item = getattr(value, field_info.name)
        if is_dataclass(item):
            result[field_info.name] = _dataclass_dict(item)
        elif isinstance(item, tuple):
            result[field_info.name] = list(item)
        else:
            result[field_info.name] = _copy_value(item)
    return result


def _section(data: Mapping[str, object], key: str, label: str) -> Mapping[str, object]:
    if key not in data:
        return {}
    value = data[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.{key} must be a mapping")
    return value


def _mapping(data: Mapping[str, object], key: str, label: str) -> Mapping[str, object]:
    if key not in data or not isinstance(data[key], Mapping):
        raise ValueError(f"{label}.{key} must be a mapping")
    return data[key]  # type: ignore[return-value]


def _reject_keys(
    data: Mapping[str, object], allowed: set[str] | frozenset[str], label: str
) -> None:
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        paths = [f"{label}.{key}" for key in unknown]
        raise ValueError(f"unsupported configuration field(s): {paths}")


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _nullable_bool(value: object, label: str) -> bool | None:
    if value is None:
        return None
    return _bool(value, label)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _env_id_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a non-empty sequence of integers")
    result = tuple(
        _nonnegative_int(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    if not result:
        raise ValueError(f"{label} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} cannot contain duplicates")
    return result


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _positive_int_or_auto(value: object, label: str) -> MaxBatchProblems:
    if value == "auto":
        return "auto"
    return _positive_int(value, label)


def _planner_failure_policy(value: object, label: str) -> str:
    return _enum(value, {"hold_failed_env", "reject_request"}, label)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_float(value: object, label: str) -> float:
    result = _number(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_float(value: object, label: str) -> float:
    result = _number(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _nonempty_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _enum(value: object, choices: set[str] | frozenset[str], label: str) -> str:
    result = _nonempty_str(value, label)
    if result not in choices:
        raise ValueError(f"{label} must be one of {sorted(choices)}")
    return result


def _profile_name(value: object, label: str) -> str:
    result = _nonempty_str(value, label)
    if result in {".", ".."} or "/" in result or "\\" in result:
        raise ValueError(f"{label} must be a simple profile name")
    return result


def _loopback_host(value: object, label: str) -> str:
    """限制进程监听端点为无需外部网络暴露的 loopback 地址。"""

    return require_loopback_host(value, label=label)


def _nullable_port(value: object, label: str) -> int | None:
    if value is None:
        return None
    result = _positive_int(value, label)
    if result > 65535:
        raise ValueError(f"{label} must be between 1 and 65535")
    return result


def _size(value: object, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{label} must contain exactly two positive integers")
    return (
        _positive_int(value[0], f"{label}[0]"),
        _positive_int(value[1], f"{label}[1]"),
    )


def _topic(value: object, label: str) -> str:
    result = _nonempty_str(value, label)
    if not result.startswith("/") or "//" in result or ".." in result.split("/"):
        raise ValueError(f"{label} must be an absolute topic path")
    return result


def _nullable_path(value: object, label: str) -> str | None:
    if value is None:
        return None
    result = _nonempty_str(value, label)
    if "\x00" in result:
        raise ValueError(f"{label} contains a NUL byte")
    if any(part == ".." for part in PurePath(result).parts):
        raise ValueError(f"{label} must not contain '..'")
    return result


def _merge_declared(
    base: Mapping[str, object], override: Mapping[str, object]
) -> dict[str, object]:
    """递归合并已声明的配置树，同时复制容器以隔离调用方后续修改。"""

    merged = _copy_mapping(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _merge_declared(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = _copy_value(value)
    return merged


def _copy_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _copy_value(item) for key, item in value.items()}


def _copy_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _copy_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_copy_value(item) for item in value]
    return value


def _flatten(data: Mapping[str, object], prefix: str = "") -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_flatten(value, path))
        else:
            result[path] = value
    return result


def _set_dotted(data: dict[str, object], path: str, value: object) -> None:
    """设置点路径覆盖，并拒绝标量与子树占用同一路径的歧义。"""

    parts = path.split(".")
    if not path or any(not part for part in parts):
        raise ValueError(f"invalid configuration path: {path!r}")
    current = data
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            nested: dict[str, object] = {}
            current[part] = nested
            current = nested
        elif isinstance(existing, dict):
            current = existing
        else:
            raise ValueError(f"configuration path conflicts at {part!r}")
    current[parts[-1]] = value


def _mark_source_tree(
    sources: dict[str, str], effective: Mapping[str, object], path: str, source: str
) -> None:
    """把显式提供的字段及其所有有效叶子统一标记为同一来源。"""

    prefix = f"runtime.{path}"
    matching = [key for key in sources if key == prefix or key.startswith(f"{prefix}.")]
    for key in matching:
        sources[key] = source


_RUNTIME_KEYS = frozenset(
    {
        "mode",
        "profiles",
        "simulation_app",
        "execution",
        "interactive",
        "planner",
        "playback",
        "camera_output",
        "telemetry",
        "output",
        "paths",
        "shutdown",
    }
)
_RENDER_KEYS = frozenset(
    {
        "gui_size",
        "headless_size",
        "window_size",
        "renderer",
        "anti_aliasing_gui",
        "anti_aliasing_headless",
        "samples_per_pixel_per_frame",
        "denoiser",
        "hide_ui",
        "disable_viewport_updates",
        "fast_shutdown",
        "material_sync_loads",
        "hydra_material_sync_loads",
        "headless_dt_policy",
    }
)
_TRANSPORT_KEYS = frozenset(
    {
        "tcp_jsonl",
        "websocket",
        "max_message_bytes",
        "max_connections",
        "request_queue_capacity",
        "event_queue_capacity",
        "overflow_policy",
        "startup_timeout_s",
        "server_poll_interval_s",
        "response_poll_interval_s",
    }
)
__all__ = [
    "CameraOutputRuntimeSettings",
    "FoxgloveLiveRuntimeSettings",
    "InteractiveRuntimeSettings",
    "McapRuntimeSettings",
    "OutputPolicySettings",
    "PlaybackResourceSettings",
    "PlannerRequestDefaults",
    "PlannerResourceSettings",
    "ResolvedRuntimeConfig",
    "RuntimeExecutionSettings",
    "RuntimePathSettings",
    "RuntimePlannerSettings",
    "RuntimeProfileConfig",
    "RuntimeProfileSelection",
    "RuntimeTransportEndpoint",
    "RuntimeTransportSettings",
    "ShutdownSettings",
    "SimulationAppSettings",
    "SimulationGpuSettings",
    "SimulationRenderSettings",
    "TelemetryRuntimeSettings",
    "TelemetryTopicSettings",
    "load_runtime_profile",
    "resolve_runtime_config",
]

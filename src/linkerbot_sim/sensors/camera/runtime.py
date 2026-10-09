"""Isaac Experimental RTX sensor camera 的创建与兼容 facade。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from linkerbot_sim.isaac.physics.backend import normalize_physics_backend
from linkerbot_sim.sensors.config import SceneSensorSettings
from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz
from linkerbot_sim.visualization.viewport import (
    set_viewport_camera_navigation_enabled,
)

from .config import (
    SensorCameraIntrinsicsSettings,
    SensorCameraSettings,
)


ArrayFactory = Callable[[object], object]
QuatFactory = Callable[[tuple[float, float, float]], object]

_MODALITY_ANNOTATORS = {
    "rgb": "rgb",
    "depth": "distance_to_image_plane",
    "semantic_segmentation": "semantic_segmentation",
    "instance_segmentation": "instance_segmentation",
}

# Legacy Camera interpreted configured RPY in its world-camera axes (+X forward,
# +Z up) and converted it to USD camera axes (+Y up, -Z forward). RtxCamera
# accepts USD transforms directly, so preserve the established configuration
# contract explicitly.
_WORLD_CAMERA_TO_USD_QUAT_WXYZ = (0.5, 0.5, -0.5, -0.5)
_USD_TO_WORLD_CAMERA_QUAT_WXYZ = (0.5, -0.5, 0.5, 0.5)


@dataclass
class SensorCameraRuntime:
    """已创建的 RTX CameraSensor 及项目稳定访问接口。"""

    settings: SensorCameraSettings
    camera: object
    _frame_tracker: _NativeFrameTracker | None = field(
        default=None, init=False, repr=False
    )
    _capture_metadata: dict[str, object] = field(
        default_factory=dict, init=False, repr=False
    )
    _render_active: bool | None = field(default=None, init=False, repr=False)

    @property
    def render_update_count(self) -> int:
        return getattr(self.camera, "render_update_count", 1)

    def set_render_active(self, active: bool) -> None:
        """Enable only this owned product, without recreating sensors or advancing physics."""

        if type(active) is not bool:
            raise TypeError("camera render activation must be a boolean")
        if self._render_active is active:
            return
        if isinstance(self.camera, _NewtonSyntheticDataCamera):
            self.camera.set_render_active(active)
        else:
            # Isaac 6.1 CameraSensor exposes the USD product but no public pause
            # method. Replicator owns this Hydra handle and uses the same switch;
            # keep the version-specific access here, alongside native cleanup.
            self.camera._hydra_texture.hydra_texture.set_updates_enabled(active)
        self._render_active = active

    def begin_render_capture(self) -> None:
        """Invalidate the last capture before publishing another frozen snapshot."""

        self._capture_metadata.clear()
        if self._frame_tracker is not None:
            self._frame_tracker.begin()

    def invalidate_render_capture(self) -> None:
        """A paused state edit invalidates pixels even if physical time is unchanged."""

        self._capture_metadata.clear()

    def capture_matches_time(self, time_s: float) -> bool:
        if self._frame_tracker is None:
            return True
        captured_time = self._capture_metadata.get("physics_time_s")
        return (
            captured_time is not None and abs(float(captured_time) - time_s) <= 1.0e-9
        )

    def render_capture_ready(self) -> bool:
        if self._frame_tracker is None:
            return True
        if not self._frame_tracker.ready:
            return False
        for modality in self.settings.modalities:
            data = self.get_modality_data(modality, device=None)
            shape = getattr(data, "shape", ())
            if len(shape) < 2 or not all(shape):
                return False
        return True

    def render_capture_status(self) -> dict[str, object]:
        tracker = self._frame_tracker
        if tracker is None:
            return {"native_tracking": False}
        return {
            "render_product_path": tracker.product_path,
            "native_frame_id": tracker.frame_id,
            "minimum_frame_id": tracker._minimum_frame,
            "native_frame_source": tracker.frame_source,
            "render_time": tracker._render_time,
            "minimum_render_time": tracker._minimum_render_time,
            "native_ready": tracker.ready,
            "modality_shapes": {
                modality: tuple(
                    getattr(self.get_modality_data(modality, device=None), "shape", ())
                )
                for modality in self.settings.modalities
            },
        }

    def finish_render_capture(
        self, *, snapshot_index: int, physics_time_s: float | None
    ) -> None:
        if self._frame_tracker is not None:
            self._capture_metadata.update(
                native_frame_id=self._frame_tracker.frame_id,
                native_frame_source=self._frame_tracker.frame_source,
                render_product_path=self._frame_tracker.product_path,
                render_snapshot_index=snapshot_index,
                physics_time_s=physics_time_s,
            )

    def get_capture_metadata(self) -> dict[str, object]:
        """Native render identity; empty until a complete owner render transaction."""

        if not self._capture_metadata:
            return {}
        return {
            **self._capture_metadata,
            "camera_profile": self.settings.camera_profile,
            "sensor_model": self.settings.sensor_model,
            "calibration_source": self.settings.calibration_source,
            "parent_prim_path": self.settings.parent_prim_path,
            "pose_axes": self.settings.pose_axes,
        }

    @property
    def name(self) -> str:
        """配置中的摄像机名称。"""

        return self.settings.name

    @property
    def prim_path(self) -> str:
        """摄像机 USD prim path。"""

        return self.settings.prim_path

    def get_modality_data(
        self,
        modality: str,
        *,
        device: str | None = "cpu",
        clone: bool = False,
    ) -> object:
        """读取一个 modality，并在请求 CPU 时把 Warp 数据显式复制到主机。"""

        self._require_modality(modality)
        annotator = _MODALITY_ANNOTATORS[modality]
        get_data = getattr(self.camera, "get_data", None)
        if callable(get_data):
            data, _info = get_data(annotator)
            return _array_for_consumer(data, device=device, clone=clone)

        # Lightweight test doubles and downstream adapters written against the
        # previous facade remain usable; production Isaac 6 always takes the
        # get_data branch above.
        if modality == "rgb":
            data = self.camera.get_rgb(device=device)
        elif modality == "depth":
            data = self.camera.get_depth(device=device)
        else:
            frame = self.camera.get_current_frame(clone=clone)
            data = frame.get(modality) if isinstance(frame, dict) else None
        return _array_for_consumer(data, device=device, clone=clone)

    def get_current_frame(self, *, clone: bool = False) -> dict[str, object]:
        """按项目旧 facade 契约返回已配置 annotator 的 frame mapping。"""

        return {
            modality: self.get_modality_data(modality, clone=clone)
            for modality in self.settings.modalities
        }

    def get_rgb(self, *, device: str | None = None) -> object:
        """读取 RGB 图像。"""

        return self.get_modality_data("rgb", device=device)

    def get_depth(self, *, device: str | None = None) -> object:
        """读取 distance-to-image-plane depth 图像。"""

        return self.get_modality_data("depth", device=device)

    def get_intrinsics_matrix(self, *, device: str | None = None) -> object:
        """读取 3x3 pinhole 内参矩阵；配置值优先。"""

        del device
        if self.settings.intrinsics is not None:
            return self.settings.intrinsics.matrix()

        legacy_method = getattr(self.camera, "get_intrinsics_matrix", None)
        if callable(legacy_method):
            return legacy_method(device="cpu")

        optical_camera = self.camera.camera
        focal_length = _first_scalar(optical_camera.get_focal_lengths())
        horizontal, vertical = optical_camera.get_apertures()
        width, height = self.settings.resolution
        fx = width * focal_length / _first_scalar(horizontal)
        fy = height * focal_length / _first_scalar(vertical)
        return (
            (fx, 0.0, width * 0.5),
            (0.0, fy, height * 0.5),
            (0.0, 0.0, 1.0),
        )

    def get_world_pose(self) -> tuple[object, object]:
        """以旧 facade 的 world-camera axes 返回无 batch 维的世界 pose。"""

        legacy_method = getattr(self.camera, "get_world_pose", None)
        if callable(legacy_method):
            return legacy_method()

        positions, orientations = self.camera.authoring_object.get_world_poses()
        position = _first_row(_array_for_consumer(positions, device="cpu"))
        orientation_usd = _first_row(_array_for_consumer(orientations, device="cpu"))
        orientation = _quat_multiply(
            orientation_usd,
            _USD_TO_WORLD_CAMERA_QUAT_WXYZ,
        )
        return position, orientation

    def close(self) -> None:
        """Release renderer resources owned by this camera, if any."""

        if self._frame_tracker is not None:
            self._frame_tracker.close()
            self._frame_tracker = None
        self._capture_metadata.clear()
        close = getattr(self.camera, "close", None)
        if callable(close):
            close()
        else:
            # Isaac 6 CameraSensor has no public close; release its annotators and
            # Hydra product using the same idempotent hook as its destructor,
            # while Kit is still alive. Keep this version-specific fallback here.
            invalidate = getattr(self.camera, "_invalidate_sensor", None)
            if callable(invalidate):
                invalidate()

    def _require_modality(self, modality: str) -> None:
        """确认 camera 配置中启用了指定 modality。"""

        if modality not in self.settings.modalities:
            raise ValueError(
                f"camera {self.name!r} was not configured with modality {modality!r}"
            )


def create_sensor_camera_runtimes(
    *,
    stage: object,
    sensors: SceneSensorSettings,
    physics_backend: object,
) -> tuple[SensorCameraRuntime, ...]:
    """为 enabled=true 的 camera 配置创建 RTX runtime。"""

    sensors.validate_mirror_camera_scope()
    if not sensors.enabled_cameras:
        return ()
    created: list[SensorCameraRuntime] = []
    try:
        for settings in sensors.enabled_cameras:
            created.append(
                create_sensor_camera_runtime(
                    stage=stage,
                    settings=settings,
                    physics_backend=physics_backend,
                )
            )
    except BaseException as creation_error:
        # 每台相机都独占 native viewport/render product。批量构造失败时必须在
        # Session/App 仍存活的边界逆序释放已完成项，同时保留真正的构造异常。
        for camera in reversed(created):
            try:
                camera.close()
            except BaseException as cleanup_error:
                creation_error.add_note(
                    "previous camera cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        raise
    return tuple(created)


def create_sensor_camera_runtime(
    *,
    stage: object,
    settings: SensorCameraSettings,
    physics_backend: object,
    rtx_camera_type: type | None = None,
    camera_sensor_type: type | None = None,
    array_factory: ArrayFactory | None = None,
    quat_from_rpy: QuatFactory | None = None,
) -> SensorCameraRuntime:
    """创建 ``RtxCamera`` authoring object 和 ``CameraSensor`` runtime。"""

    backend = normalize_physics_backend(physics_backend)
    if (
        rtx_camera_type is None
        and camera_sensor_type is None
        and array_factory is None
        and quat_from_rpy is None
        and backend == "newton"
    ):
        return _create_newton_camera_runtime(stage=stage, settings=settings)
    settings = _resolve_parent_link(stage, settings)
    _validate_parent_prim(stage=stage, settings=settings)
    if (
        rtx_camera_type is None
        or camera_sensor_type is None
        or array_factory is None
        or quat_from_rpy is None
    ):
        (
            default_rtx_camera_type,
            default_camera_sensor_type,
            default_array_factory,
            default_quat_from_rpy,
        ) = _load_camera_dependencies()
        rtx_camera_type = rtx_camera_type or default_rtx_camera_type
        camera_sensor_type = camera_sensor_type or default_camera_sensor_type
        array_factory = array_factory or default_array_factory
        quat_from_rpy = quat_from_rpy or default_quat_from_rpy

    orientation = _quat_multiply(
        quat_from_rpy(settings.pose_rpy),
        _camera_to_usd_quaternion(settings),
    )
    schemas, attributes = _opencv_pinhole_schema_settings(settings)
    authoring_camera = rtx_camera_type(
        settings.prim_path,
        tick_rate=settings.frequency,
        schemas=schemas,
        attributes=attributes,
        translations=array_factory((settings.pose_xyz,)),
        orientations=array_factory((orientation,)),
    )
    optical_camera = authoring_camera.camera
    near, far = settings.clipping_range
    optical_camera.set_clipping_ranges(near, far)

    width, height = settings.resolution
    camera_sensor = camera_sensor_type(
        authoring_camera,
        resolution=(height, width),
        annotators=[_MODALITY_ANNOTATORS[item] for item in settings.modalities],
    )
    _apply_camera_intrinsics(camera=optical_camera, settings=settings)
    return SensorCameraRuntime(settings=settings, camera=camera_sensor)


def initialize_sensor_camera_runtimes(
    cameras: tuple[SensorCameraRuntime, ...],
) -> None:
    """Finalize camera resources and subscribe to their native frame completions."""

    newton_cameras = tuple(
        camera.camera
        for camera in cameras
        if isinstance(camera.camera, _NewtonSyntheticDataCamera)
    )
    for camera in newton_cameras:
        camera.initialize()
    for runtime in cameras:
        if isinstance(runtime.camera, _NewtonSyntheticDataCamera):
            runtime.settings = runtime.camera._settings
    for runtime in cameras:
        if runtime._frame_tracker is not None:
            continue
        camera = runtime.camera
        if isinstance(camera, _NewtonSyntheticDataCamera):
            path = camera.render_product_path
        else:
            product = getattr(camera, "render_product", None)
            if product is None:  # injected non-native test/consumer adapter
                continue
            path = str(product.GetPath())
        runtime._frame_tracker = _NativeFrameTracker(
            path,
            frame_source="kit_render_product_frame"
            if isinstance(camera, _NewtonSyntheticDataCamera)
            else "kit_swh",
        )


class _NativeFrameTracker:
    """Match native completion to a product and the first post-publish Kit frame.

    A render call counter or simulation timestamp cannot identify a fresh paused
    image. SWH frame numbers also work when physical time stays frozen. The event
    contract is shared by Isaac's SyntheticData next_sensor_data_async helper.
    """

    def __init__(self, product_path: str, *, frame_source: str = "kit_swh") -> None:
        import carb.eventdispatcher
        import omni.kit.app
        import omni.usd
        from omni.syntheticdata.scripts import helpers

        self.product_path = product_path
        self.frame_source = frame_source
        self.frame_id: int | None = None
        self._minimum_frame: int | None = None
        self._minimum_render_time: float | None = None
        self._render_time: float | None = None
        self._previous_frame: int | None = None
        self._armed = False
        native = helpers._get_syntheticdata_iface()
        self._parse_event = native.parse_rendered_simulation_event
        stage_id = omni.usd.get_context().get_stage_id()
        self._simulation_render_time = lambda: native.get_rational_time_of_simulation(
            stage_id, 0
        )
        dispatcher = carb.eventdispatcher.get_eventdispatcher()
        self._subscriptions = []
        try:
            self._subscriptions.append(
                dispatcher.observe_event(
                    event_name=omni.kit.app.GLOBAL_EVENT_PRE_UPDATE,
                    on_event=self._on_pre_update,
                    observer_name=f"MirrorCamera.pre:{product_path}",
                )
            )
            self._subscriptions.append(
                dispatcher.observe_event(
                    event_name=omni.usd.get_context().stage_rendering_event_name(
                        omni.usd.StageRenderingEventType.NEW_FRAME
                    ),
                    on_event=self._on_render,
                    observer_name=f"MirrorCamera.frame:{product_path}",
                )
            )
        except BaseException:
            self.close()
            raise

    def begin(self) -> None:
        self._minimum_frame = None
        self._minimum_render_time = None
        self._previous_frame = self.frame_id
        self._armed = True

    def _on_pre_update(self, event: Any) -> None:
        if not self._armed:
            return
        if self.frame_source == "kit_swh" and self._minimum_frame is None:
            self._minimum_frame = int(event["SWHFrameNumber"] or event["frameNumber"])
        elif (
            self.frame_source == "kit_render_product_frame"
            and self._minimum_render_time is None
        ):
            numerator, denominator = self._simulation_render_time()
            if denominator > 0:
                self._minimum_render_time = numerator / denominator

    def _on_render(self, event: Any) -> None:
        parsed = self._parse_event(event["product_path_handle"], event["results"])
        if str(parsed[0]) != self.product_path:
            return
        # Newton's exclusive closure has no SWH stage-update owner (SWH == -1).
        # Use the native product frame number and SyntheticData's rational render
        # clock barrier instead, as next_render_simulation_async does. This clock
        # is a renderer clock, never a replacement for Newton simulation_time.
        field = "swh_frame_number" if self.frame_source == "kit_swh" else "frame_number"
        self.frame_id = int(event[field])
        self._render_time = parsed[1] / parsed[2] if parsed[2] > 0 else None

    @property
    def ready(self) -> bool:
        if not self._armed or self.frame_id is None or self.frame_id < 0:
            return False
        if self._previous_frame is not None and self.frame_id <= self._previous_frame:
            return False
        if self.frame_source == "kit_swh":
            return (
                self._minimum_frame is not None and self.frame_id >= self._minimum_frame
            )
        return (
            self._minimum_render_time is not None
            and self._render_time is not None
            and self._render_time >= self._minimum_render_time
        )

    def close(self) -> None:
        for subscription in self._subscriptions:
            subscription.reset()
        self._subscriptions.clear()
        self._armed = False


class _NewtonSyntheticDataCamera:
    """由独占 SyntheticData viewport 支撑的 Newton RGB/depth camera。

    Isaac Sim 6.1 下，本项目经 provenance 审计的 exclusive render closure 不使用
    Replicator/Isaac CameraSensor 的默认依赖闭包，因为其中包含 Newton 模式禁止的
    physics-owner/stage-update 依赖。每台相机拥有独立 viewport 与 render product；Newton
    manager 只负责发布同一份 body transform 快照和轮转 product，相机本身不调用 solver，
    也不拥有 simulation time。
    """

    # render Kit 保留三帧 history。一次新 Newton snapshot 发布后，前三个 app.update 仍可能
    # 只推进 RTX/SyntheticData 内部流水线，第四次才包含完整输出帧；单相机也不能缩成一次。
    # 多相机由 Mirror RenderCoordinator 按 product 逐个执行这四次 update，期间 physics state
    # 与 clock 均冻结；物理 manager 不感知 camera 或 viewport。
    render_update_count = 4

    @property
    def render_product_path(self) -> str:
        return str(self._viewport.render_product_path)

    def __init__(self, *, stage: object, settings: SensorCameraSettings) -> None:
        unsupported = sorted(set(settings.modalities) - {"rgb", "depth"})
        if unsupported:
            raise RuntimeError(
                "Newton camera currently supports RGB/depth modalities only: "
                f"camera={settings.name!r}, unsupported={unsupported}"
            )
        self._stage = stage
        self._settings = settings
        self._closed = False
        self._window = None
        self._viewport = None
        self._sensor_types: dict[str, object] = {}
        self._initialized = False
        try:
            # Robot-mounted camera 的 parent 此时可能尚未随资产导入。这里只预留
            # native viewport；Camera prim 与 sensor graph 都在 initialize() 创建。
            self._create_viewport()
        except BaseException as creation_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                creation_error.add_note(
                    "Newton camera cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

    def initialize(self, *, force: bool = False) -> None:
        if self._closed:
            raise RuntimeError("cannot initialize a closed Newton camera")
        if self._initialized and not force:
            return
        # 资产、Newton model 和 reset 已全部完成，此时 robot link 等挂载父 prim 才稳定。
        # 提前 Define 会制造不完整机器人 namespace，并阻塞 importer 的 canonical target。
        self._settings = _resolve_parent_link(self._stage, self._settings)
        _validate_parent_prim(stage=self._stage, settings=self._settings)
        self._define_camera_prim()
        import omni.syntheticdata as syn
        from pxr import Sdf

        viewport = self._window.viewport_api
        if viewport is None:
            raise RuntimeError(
                f"Newton camera viewport is unavailable: {self._settings.name}"
            )
        width, height = self._settings.resolution
        viewport.camera_path = Sdf.Path(self._settings.prim_path)
        viewport.updates_enabled = True
        try:
            viewport.resolution = (int(width), int(height))
        except (AttributeError, TypeError):
            pass
        available = {
            "rgb": syn._syntheticdata.SensorType.Rgb,
            "depth": syn._syntheticdata.SensorType.DistanceToImagePlane,
        }
        self._sensor_types = {
            name: available[name] for name in self._settings.modalities
        }
        for sensor_type in self._sensor_types.values():
            syn.sensors.create_or_retrieve_sensor(viewport, sensor_type)
        self._viewport = viewport
        self._initialized = True

    def get_data(self, annotator: str) -> tuple[object, dict[str, object]]:
        if self._closed:
            raise RuntimeError("Newton camera is closed")
        import omni.syntheticdata as syn

        getters = {
            "rgb": syn.sensors.get_rgb,
            "distance_to_image_plane": syn.sensors.get_distance_to_image_plane,
        }
        getter = getters.get(str(annotator))
        if getter is None:
            raise ValueError(f"Newton camera annotator is unsupported: {annotator}")
        data = getter(self._viewport)
        if str(annotator) == "rgb" and getattr(data, "ndim", 0) == 3:
            data = data[..., :3]
        return data, {}

    def set_render_active(self, active: bool) -> None:
        """选择本相机 viewport；只控制渲染更新，不推进物理。"""

        if self._closed or self._viewport is None:
            raise RuntimeError("cannot select a closed Newton camera")
        self._viewport.updates_enabled = bool(active)

    def get_world_pose(self) -> tuple[object, object]:
        from linkerbot_sim.isaac.scene.pose import read_prim_world_pose

        pose = read_prim_world_pose(self._stage, self._settings.prim_path)
        if pose is None:
            raise RuntimeError(
                f"Newton camera prim disappeared: {self._settings.prim_path}"
            )
        position, orientation_usd = pose
        return tuple(position), _quat_multiply(
            orientation_usd,
            _USD_TO_WORLD_CAMERA_QUAT_WXYZ,
        )

    def get_intrinsics_matrix(self, *, device: str = "cpu") -> object:
        del device
        intrinsics = self._settings.intrinsics
        if intrinsics is not None:
            return intrinsics.matrix()
        width, height = self._settings.resolution
        return (
            (float(width) * 0.5, 0.0, float(width) * 0.5),
            (0.0, float(height) * 0.5, float(height) * 0.5),
            (0.0, 0.0, 1.0),
        )

    def close(self) -> None:
        if self._closed:
            return
        first_error: BaseException | None = None
        if self._initialized and self._viewport is not None:
            try:
                import omni.syntheticdata as syn

                syn.sensors.disable_sensors(
                    self._viewport,
                    list(self._sensor_types.values()),
                )
            except (ImportError, ModuleNotFoundError):
                pass
            except BaseException as exc:
                first_error = exc
            else:
                self._initialized = False
                self._sensor_types.clear()
        destroy = getattr(self._window, "destroy", None)
        if callable(destroy):
            try:
                destroy()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                self._viewport = None
                self._window = None
                # viewport 销毁后，其拥有的 SyntheticData graph 也随之失效；即使显式 disable
                # 失败，后续 retry 也只需收口本地 Python 状态。
                self._sensor_types.clear()
                self._initialized = False
        elif self._window is not None:
            self._window = None
            self._viewport = None
        if first_error is not None:
            raise first_error
        self._sensor_types.clear()
        self._initialized = False
        self._stage = None
        self._closed = True

    def _define_camera_prim(self) -> None:
        from pxr import Gf, Sdf, UsdGeom

        camera = UsdGeom.Camera.Define(self._stage, self._settings.prim_path)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(*self._settings.clipping_range))
        intrinsics = self._settings.intrinsics
        if intrinsics is not None:
            width, height = self._settings.resolution
            focal_length = 1.0
            camera.GetFocalLengthAttr().Set(focal_length)
            camera.GetHorizontalApertureAttr().Set(
                focal_length * float(width) / intrinsics.fx
            )
            camera.GetVerticalApertureAttr().Set(
                focal_length * float(height) / intrinsics.fy
            )
            schemas, attributes = _opencv_pinhole_schema_settings(self._settings)
            for schema in schemas or ():
                camera.GetPrim().ApplyAPI(schema)
            for name, value in (attributes or {}).items():
                attr = camera.GetPrim().GetAttribute(name)
                if not attr.IsValid():
                    attr = camera.GetPrim().CreateAttribute(
                        name,
                        _sdf_type_for_camera_attribute(name, Sdf),
                        custom=False,
                    )
                attr.Set(value)
        orientation = _quat_multiply(
            _quat_from_extrinsic_xyz_rpy(self._settings.pose_rpy),
            _camera_to_usd_quaternion(self._settings),
        )
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*self._settings.pose_xyz)
        )
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(float(orientation[0]), Gf.Vec3d(*orientation[1:]))
        )

    def _create_viewport(self) -> None:
        from omni.kit.viewport.utility import create_viewport_window

        width, height = self._settings.resolution
        window = create_viewport_window(
            name=f"NewtonCamera:{self._settings.name}",
            width=int(width),
            height=int(height),
            usd_drop_support=False,
        )
        # 在检查 viewport_api 前先接管 window 所有权。这样下面任意一步抛错时，close() 都能
        # 销毁已经创建的 native window，避免失败初始化把 render product 留到 app teardown。
        self._window = window
        viewport = None if window is None else window.viewport_api
        if viewport is None:
            raise RuntimeError(
                f"failed to create Newton camera viewport: {self._settings.name}"
            )
        # Kit 的 camera manipulator extension 会给所有 ViewportWindow 自动挂载导航 layer。
        # 该窗口是 SyntheticData render product，不是人工观察视角；必须在返回事件循环前
        # 按窗口关闭导航，避免鼠标拖动直接改写 /World 下的传感器 Camera prim 外参。
        set_viewport_camera_navigation_enabled(window, enabled=False)
        # 预留阶段只激活 Hydra viewport，不绑定尚未 author 的 Camera prim，也不创建
        # SyntheticData graph；后两步严格留给完整 Newton scene 之后的 initialize()。
        viewport.updates_enabled = True
        try:
            viewport.resolution = (int(width), int(height))
        except (AttributeError, TypeError):
            pass
        self._viewport = viewport


def _create_newton_camera_runtime(
    *,
    stage: object,
    settings: SensorCameraSettings,
) -> SensorCameraRuntime:
    """创建 Newton render camera，并把所有权直接交给 Mirror CameraBundle。

    物理 runtime 只发布 physics-to-USD transform，不再登记或代关 camera。这样 camera 的
    生命周期与输出 sink 一起先于 ``IsaacSession`` 结束，也避免 manager close 失败时丢失渲染资源。
    """

    camera = _NewtonSyntheticDataCamera(stage=stage, settings=settings)
    return SensorCameraRuntime(
        settings=settings,
        camera=camera,
    )


def _sdf_type_for_camera_attribute(name: str, sdf: object) -> object:
    if name.endswith("imageSize"):
        return sdf.ValueTypeNames.Int2
    return sdf.ValueTypeNames.Float


def _opencv_pinhole_schema_settings(
    settings: SensorCameraSettings,
) -> tuple[list[str] | None, dict[str, object] | None]:
    """为显式 pixel intrinsics 构造 OpenCV pinhole schema 参数。"""

    intrinsics = settings.intrinsics
    if intrinsics is None:
        return None, None
    from pxr import Gf

    width, height = settings.resolution
    return ["OmniLensDistortionOpenCvPinholeAPI"], {
        "omni:lensdistortion:opencvPinhole:cx": intrinsics.cx,
        "omni:lensdistortion:opencvPinhole:cy": intrinsics.cy,
        "omni:lensdistortion:opencvPinhole:fx": intrinsics.fx,
        "omni:lensdistortion:opencvPinhole:fy": intrinsics.fy,
        "omni:lensdistortion:opencvPinhole:imageSize": Gf.Vec2i(width, height),
    }


def _apply_camera_intrinsics(*, camera: object, settings: SensorCameraSettings) -> None:
    """把显式 pinhole intrinsics 写入 experimental Camera optical wrapper。"""

    intrinsics = settings.intrinsics
    if intrinsics is None:
        return
    _apply_aperture_intrinsics(
        camera=camera,
        intrinsics=intrinsics,
        resolution=settings.resolution,
        camera_name=settings.name,
    )


def _apply_aperture_intrinsics(
    *,
    camera: object,
    intrinsics: SensorCameraIntrinsicsSettings,
    resolution: tuple[int, int],
    camera_name: str,
) -> None:
    """同步 native Camera 的 focal/aperture 与 pixel intrinsics。"""

    width, height = resolution
    focal_length = 1.0
    horizontal_aperture = focal_length * float(width) / intrinsics.fx
    vertical_aperture = focal_length * float(height) / intrinsics.fy

    _required_camera_method(camera, "set_focal_lengths", camera_name=camera_name)(
        focal_length
    )
    _required_camera_method(camera, "set_apertures", camera_name=camera_name)(
        horizontal_aperture,
        vertical_aperture,
    )


def _required_camera_method(
    camera: object, name: str, *, camera_name: str
) -> Callable[..., Any]:
    """读取显式 intrinsics 所需的 optical Camera method。"""

    method = getattr(camera, name, None)
    if not callable(method):
        raise RuntimeError(
            f"sensors.cameras.{camera_name}.intrinsics requires Camera.{name}"
        )
    return method


def _load_camera_dependencies() -> tuple[type, type, ArrayFactory, QuatFactory]:
    """延迟加载 Experimental RTX camera 依赖。"""

    import numpy as np
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

    def array_factory(values: object) -> object:
        return np.asarray(values, dtype=float)

    return RtxCamera, CameraSensor, array_factory, _quat_from_extrinsic_xyz_rpy


def _quat_from_extrinsic_xyz_rpy(
    rpy: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """把 extrinsic XYZ RPY 转成 scalar-first quaternion。"""

    return _float_sequence(rpy_xyz_to_quat_wxyz(rpy), size=4)


def _quat_multiply(left: object, right: object) -> tuple[float, float, float, float]:
    """计算两个 wxyz quaternion 的 Hamilton product。"""

    lw, lx, ly, lz = _float_sequence(left, size=4)
    rw, rx, ry, rz = _float_sequence(right, size=4)
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _array_for_consumer(
    value: object,
    *,
    device: str | None,
    clone: bool = False,
) -> object:
    """按 facade 请求把 Warp/array-like payload 转为消费者可用数据。"""

    if value is None:
        return None
    if device == "cpu":
        numpy_method = getattr(value, "numpy", None)
        if callable(numpy_method):
            value = numpy_method()
    if clone:
        copy_method = getattr(value, "copy", None)
        if callable(copy_method):
            value = copy_method()
    return value


def _first_scalar(value: object) -> float:
    """从 experimental API 的 `(N, 1)` 输出读取首个标量。"""

    current = _array_for_consumer(value, device="cpu")
    while isinstance(current, Sequence) or hasattr(current, "shape"):
        try:
            current = current[0]  # type: ignore[index]
        except (IndexError, TypeError):
            break
    return float(current)


def _first_row(value: object) -> tuple[float, ...]:
    """从 experimental API 的 `(1, D)` 输出去掉 batch 维。"""

    try:
        first = value[0]  # type: ignore[index]
    except (IndexError, TypeError):
        first = value
    if hasattr(first, "tolist"):
        first = first.tolist()
    return tuple(float(item) for item in first)


def _float_sequence(value: object, *, size: int) -> tuple[float, ...]:
    """把 array-like 转成固定长度 float tuple。"""

    if hasattr(value, "tolist"):
        value = value.tolist()
    result = tuple(float(item) for item in value)  # type: ignore[arg-type]
    if len(result) != size:
        raise ValueError(f"expected {size} values, got {len(result)}")
    return result


def _camera_to_usd_quaternion(settings: SensorCameraSettings) -> tuple[float, ...]:
    # USD looks along -Z with +Y up. OpenCV uses +Z forward and +Y down.
    return (
        (0.0, 1.0, 0.0, 0.0)
        if settings.pose_axes == "opencv"
        else _WORLD_CAMERA_TO_USD_QUAT_WXYZ
    )


def _resolve_parent_link(
    stage: object, settings: SensorCameraSettings
) -> SensorCameraSettings:
    if settings.parent_link is None:
        return settings
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(settings.parent_prim_path)
    matches = (
        [
            prim
            for prim in Usd.PrimRange(root)
            if prim.GetName() == settings.parent_link and prim.IsA(UsdGeom.Xform)
        ]
        if root.IsValid()
        else []
    )
    if len(matches) != 1:
        raise ValueError(
            f"camera {settings.name}: expected one parent_link {settings.parent_link!r} under {settings.parent_prim_path}; found {len(matches)}"
        )
    parent = str(matches[0].GetPath())
    return replace(
        settings,
        parent_prim_path=parent,
        prim_path=parent + "/" + settings.prim_path.rsplit("/", 1)[-1],
        parent_link=None,
    )


def _validate_parent_prim(*, stage: object, settings: SensorCameraSettings) -> None:
    """校验 parent_prim_path 指向当前 stage 中已有 prim。"""

    if settings.parent_prim_path is None:
        return
    prim = stage.GetPrimAtPath(settings.parent_prim_path)
    if prim is None or not _prim_is_valid(prim):
        raise ValueError(
            f"sensors.cameras.{settings.name}.parent_prim_path does not exist: "
            f"{settings.parent_prim_path}"
        )


def _prim_is_valid(prim: Any) -> bool:
    """兼容 USD Prim 和测试 fake prim 的有效性检查。"""

    is_valid = getattr(prim, "IsValid", None)
    if is_valid is None:
        return False
    return bool(is_valid())

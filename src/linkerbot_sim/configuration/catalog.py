"""新配置图唯一的 YAML/profile/path I/O 入口。

本模块不做 last-writer-wins merge。mode 文件只列出 profile 引用，每个 leaf profile 只有
一个 canonical writer；catalog reader 记录所有来源路径并在返回根配置前完成 strict validation。
robot/object/controller 仍由各领域 parser 校验，但文件路径与读取时机统一属于当前 mode
graph；运行时只能消费这里绑定的结果，不能按名称回到仓库全局配置根重新解析。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from .common import (
    ConfigurationError,
    require_keys,
    strict_mapping,
)
from .curobo import CuroboProfileSettings
from .control import HybridForcePositionSettings, MirrorControlSettings
from .controllers import (
    ControllerProfiles,
    controller_profiles_from_mappings,
    normalize_controller_bundle_name,
)
from .modes.kaleidoscope import (
    KaleidoscopeConfig,
    kaleidoscope_mode_from_mapping,
    validate_kaleidoscope_closure,
)
from .modes.mirror import (
    MirrorConfig,
    MirrorPhysicsSettings,
    mirror_mode_from_mapping,
)
from .objects import ObjectProfileConfig, object_profile_from_mapping
from .outputs import MirrorOutputsSettings
from .physics import (
    NewtonCudaSettings,
    PhysxCudaSettings,
    physics_settings_from_mapping,
)
from .planning import MirrorPlanningSettings
from .robots import RobotProfileSettings
from .scenes import KaleidoscopeSceneSettings, MirrorSceneSettings
from .tasks.kaleidoscope import (
    JointControlActionSettings,
    JointDeltaActionSettings,
    KaleidoscopeTaskSettings,
)
from .training.skrl import SkrlTrainingSettings
from .visualization.kaleidoscope import KaleidoscopeViewportSettings
from linkerbot_sim.utils.config import load_yaml


_DEFAULT_CONFIGS_ROOT = Path(__file__).resolve().parents[3] / "configs"


def load_yaml_mapping(path: str | Path) -> dict[str, object]:
    """安全读取一个非空、字符串键的 YAML mapping。"""

    candidate = Path(path).expanduser()
    resolved = candidate
    try:
        resolved = candidate.resolve()
        data = load_yaml(resolved)
    except OSError as exc:
        raise ConfigurationError(
            f"failed to read YAML configuration {candidate}: {exc}"
        ) from exc
    except ValueError as exc:
        raise ConfigurationError(
            f"invalid YAML configuration {resolved}: {exc}"
        ) from exc
    try:
        document = strict_mapping(data, label=str(resolved))
        _require_string_mapping_keys(document, label=str(resolved))
        return document
    except ConfigurationError as exc:
        raise ConfigurationError(
            f"invalid YAML configuration {resolved}: {exc}"
        ) from exc


def _require_string_mapping_keys(
    value: object,
    *,
    label: str,
    path: str = "<root>",
    _ancestors: set[int] | None = None,
) -> None:
    """递归拒绝非法 key/循环 alias，保证 leaf parser 可安全遍历字段。"""

    is_mapping = isinstance(value, Mapping)
    is_sequence = isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )
    if not is_mapping and not is_sequence:
        return

    ancestors = set() if _ancestors is None else _ancestors
    identity = id(value)
    if identity in ancestors:
        raise ConfigurationError(
            f"{label}:{path} must not contain recursive YAML alias"
        )
    ancestors.add(identity)
    try:
        if is_mapping:
            assert isinstance(value, Mapping)
            for key, item in value.items():
                if not isinstance(key, str) or not key:
                    raise ConfigurationError(
                        f"{label}:{path} keys must be non-empty strings, got {key!r}"
                    )
                child_path = key if path == "<root>" else f"{path}.{key}"
                _require_string_mapping_keys(
                    item,
                    label=label,
                    path=child_path,
                    _ancestors=ancestors,
                )
        else:
            assert isinstance(value, Sequence)
            for index, item in enumerate(value):
                _require_string_mapping_keys(
                    item,
                    label=label,
                    path=f"{path}[{index}]",
                    _ancestors=ancestors,
                )
    finally:
        ancestors.remove(identity)


def _profile_reference(reference: str, *, label: str) -> tuple[str, ...]:
    """校验可含子目录的 profile stem，禁止绝对路径和目录逃逸。"""

    if not isinstance(reference, str) or not reference:
        raise ConfigurationError(f"{label} must be a non-empty profile reference")
    if "\\" in reference:
        raise ConfigurationError(
            f"{label} must use '/' as the profile namespace separator"
        )
    if reference.startswith("/") or reference.endswith(".yaml"):
        raise ConfigurationError(
            f"{label} must be a relative profile stem without an extension"
        )
    parts = tuple(reference.split("/"))
    if any(not part or part in {".", ".."} for part in parts):
        raise ConfigurationError(
            f"{label} contains an invalid path component: {reference!r}"
        )
    if any("." in part for part in parts):
        raise ConfigurationError(f"{label} profile stem must not contain '.'")
    return parts


def _within(path: Path, root: Path, *, label: str) -> Path:
    """解析 symlink 后仍要求路径位于配置根内。"""

    resolved = path.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ConfigurationError(
            f"{label} escapes the configuration root directory: {resolved}"
        ) from exc
    return resolved


class _ConfigurationGraphReader:
    """一次 root load 的有状态 reader，只负责路径解析和 provenance。"""

    def __init__(self, configs_root: Path) -> None:
        self.configs_root = configs_root.expanduser().resolve()
        self.sources: dict[str, Path] = {}
        self._controller_bundles: dict[str, ControllerProfiles] = {}

    def mode_path(
        self,
        source: str | Path,
        *,
        mode: Literal["mirror", "kaleidoscope"],
    ) -> Path:
        if isinstance(source, str) and "/" not in source and "\\" not in source:
            if not source or source in {".", ".."} or source.endswith(".yaml"):
                raise ConfigurationError(
                    "mode selector must be a simple stem; path callers should pass an explicit Path"
                )
            candidate = self.configs_root / "modes" / mode / f"{source}.yaml"
        else:
            candidate = Path(source)
            if not candidate.is_absolute():
                # 显式 Path 可以相对 configs root，也可以是当前 cwd 下已存在的路径。
                cwd_candidate = candidate.resolve()
                candidate = (
                    cwd_candidate
                    if cwd_candidate.exists()
                    else self.configs_root / candidate
                )
        candidate = _within(candidate, self.configs_root, label="mode config")
        if candidate.suffix != ".yaml":
            raise ConfigurationError(
                f"mode config must use the .yaml extension: {candidate}"
            )
        expected_parent = self.configs_root / "modes" / mode
        _within(candidate, expected_parent, label=f"{mode} mode config")
        self.sources["mode"] = candidate
        return candidate

    def profile(
        self,
        *,
        group: str,
        reference: str,
        root_key: str,
        namespace: str | None = None,
        provenance_key: str | None = None,
    ) -> dict[str, object]:
        parts = _profile_reference(reference, label=f"profiles.{group}")
        if namespace is not None and (len(parts) < 2 or parts[0] != namespace):
            raise ConfigurationError(
                f"profiles.{group} must reside within the {namespace!r} namespace, got {reference!r}"
            )
        group_root = _within(
            self.configs_root / group,
            self.configs_root,
            label=f"profiles.{group} root",
        )
        profile_root = group_root
        relative_parts = parts
        if namespace is not None:
            # namespace 不只是 selector 的字符串前缀，也是解析后路径的安全边界。
            # scene 文件即使通过 symlink 引用，也不得跨入另一个产品目录。
            namespace_path = group_root / namespace
            profile_root = _within(
                namespace_path,
                group_root,
                label=f"profiles.{group} {namespace!r} namespace root",
            )
            if profile_root != namespace_path:
                raise ConfigurationError(
                    f"profiles.{group} {namespace!r} namespace root directory must not be a symbolic link"
                )
            relative_parts = parts[1:]
        path = _within(
            profile_root.joinpath(*relative_parts).with_suffix(".yaml"),
            profile_root,
            label=(
                f"profiles.{group} {namespace!r} namespace"
                if namespace is not None
                else f"profiles.{group}"
            ),
        )
        document = load_yaml_mapping(path)
        require_keys(document, required={root_key}, label=str(path))
        payload = strict_mapping(document[root_key], label=f"{path}:{root_key}")
        # provenance 使用配置事实的单数名称；磁盘目录仍保留人类易读的复数形式。
        source_key = provenance_key or {"scenes": "scene", "tasks": "task"}.get(
            group, group
        )
        if source_key in self.sources:
            raise ConfigurationError(
                f"configuration provenance key {source_key!r} is already in use"
            )
        self.sources[source_key] = path
        return payload

    def immutable_sources(self) -> Mapping[str, Path]:
        # MappingProxyType 防止调用方修改 provenance；路径在 load 时均已 canonicalize。
        return MappingProxyType(dict(sorted(self.sources.items())))

    def object_profile(
        self,
        *,
        instance_name: str,
        reference: str,
    ) -> ObjectProfileConfig:
        """从同一 configs root 解析对象 profile，并登记实例级 provenance。"""

        parts = _profile_reference(reference, label="scene.objects[].object_profile")
        group_root = _within(
            self.configs_root / "objects",
            self.configs_root,
            label="object profiles root",
        )
        path = _within(
            group_root.joinpath(*parts).with_suffix(".yaml"),
            group_root,
            label="scene.objects[].object_profile",
        )
        document = load_yaml_mapping(path)
        try:
            profile = object_profile_from_mapping(
                document,
                profile_name=parts[-1],
                source=str(path),
            )
        except ValueError as exc:
            raise ConfigurationError(f"invalid object profile {path}: {exc}") from exc
        self.sources[f"object.{instance_name}"] = path
        return profile

    def robot_profile(
        self,
        *,
        instance_label: str,
        reference: str,
    ) -> RobotProfileSettings:
        """从当前配置根解析并严格校验一个 robot instance 的资产 profile。"""

        parts = _profile_reference(reference, label="scene.robots[].robot_profile")
        group_root = _within(
            self.configs_root / "robots",
            self.configs_root,
            label="robot profiles root",
        )
        path = _within(
            group_root.joinpath(*parts).with_suffix(".yaml"),
            group_root,
            label="scene.robots[].robot_profile",
        )
        try:
            profile = RobotProfileSettings.from_mapping(
                load_yaml_mapping(path),
                source=str(path),
            )
        except ValueError as exc:
            raise ConfigurationError(f"invalid robot profile {path}: {exc}") from exc
        self.sources[f"robot.{instance_label}"] = path
        return profile

    def controller_bundle(self, name: str) -> ControllerProfiles:
        """从当前配置根加载一个完整 controller bundle，并登记实际读取的文件。"""

        try:
            bundle_name = normalize_controller_bundle_name(
                name,
                label="controller bundle",
            )
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        cached = self._controller_bundles.get(bundle_name)
        if cached is not None:
            return cached

        controllers_root = _within(
            self.configs_root / "controllers",
            self.configs_root,
            label="controller profiles root",
        )
        bundle_root = _within(
            controllers_root / bundle_name,
            controllers_root,
            label="controller bundle",
        )
        component_files = {
            "arm": bundle_root / "arm_controller.yaml",
            "hand": bundle_root / "hand_controller.yaml",
        }
        default_path = bundle_root / "default_controller.yaml"
        if default_path.is_file():
            component_files["default"] = default_path
        documents: dict[str, Mapping[str, object]] = {}
        try:
            for component, candidate in component_files.items():
                path = _within(
                    candidate,
                    bundle_root,
                    label=f"controller bundle {bundle_name!r}",
                )
                documents[component] = load_yaml_mapping(path)
                self.sources[f"controller.{bundle_name}.{component}"] = path
            profiles = controller_profiles_from_mappings(
                documents,
                source=str(bundle_root),
            )
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                f"invalid controller bundle {bundle_name!r} ({bundle_root}): {exc}"
            ) from exc
        self._controller_bundles[bundle_name] = profiles
        return profiles

    def controller_bundles(
        self,
        names: tuple[str, ...],
    ) -> Mapping[str, ControllerProfiles]:
        """加载去重后的 bundle 集合，并冻结名称到配置对象的绑定。"""

        resolved: dict[str, ControllerProfiles] = {}
        for name in names:
            try:
                canonical = normalize_controller_bundle_name(
                    name,
                    label="resolved controller bundle",
                )
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc
            resolved.setdefault(canonical, self.controller_bundle(canonical))
        return MappingProxyType(dict(sorted(resolved.items())))


def _resolve_scene_asset_profiles(
    scene: MirrorSceneSettings | KaleidoscopeSceneSettings,
    *,
    reader: _ConfigurationGraphReader,
) -> MirrorSceneSettings | KaleidoscopeSceneSettings:
    """把 robot/object profile 绑定到 scene instance，保持配置根闭包。"""

    return replace(
        scene,
        robots=tuple(
            replace(
                instance,
                resolved_profile=reader.robot_profile(
                    instance_label=instance.label,
                    reference=instance.robot_profile,
                ),
            )
            for instance in scene.robots
        ),
        objects=tuple(
            replace(
                instance,
                resolved_profile=reader.object_profile(
                    instance_name=instance.name,
                    reference=instance.object_profile,
                ),
            )
            for instance in scene.objects
        ),
    )


def _effective_controller_bundle_names(
    scene: MirrorSceneSettings | KaleidoscopeSceneSettings,
    *,
    runtime_default: str,
) -> tuple[str, ...]:
    """解析 robot profile 覆盖与 physics 派生默认值形成的完整 bundle 闭包。"""

    # physics 派生的默认 bundle 也是配置事实；即使每个实例都覆盖它，仍需验证并登记
    # provenance，避免 controller 资源损坏潜伏到后续 scene 变更才暴露。
    result: list[str] = [runtime_default]
    for instance in scene.robots:
        profile = instance.resolved_profile
        if profile is None:
            raise ConfigurationError(
                f"robot instance {instance.label!r} has not yet been bound to a resolved profile"
            )
        selected = (
            instance.controller_profile or profile.controller_profile or runtime_default
        )
        if not isinstance(selected, str):
            raise ConfigurationError(
                f"robot instance {instance.label!r} controller profile must be a string"
            )
        result.append(selected)
    return tuple(result)


def _root_and_source(
    source: str | Path,
    *,
    mode: Literal["mirror", "kaleidoscope"],
    configs_root: str | Path | None,
) -> tuple[Path, str | Path]:
    """支持 selector、显式根以及位于 ``modes/<mode>`` 的直接 Path。"""

    if configs_root is not None:
        return Path(configs_root).expanduser().resolve(), source
    if isinstance(source, Path) or (
        isinstance(source, str) and ("/" in source or "\\" in source)
    ):
        path = Path(source).expanduser().resolve()
        if path.parent.name == mode and path.parent.parent.name == "modes":
            return path.parents[2], path
    return _DEFAULT_CONFIGS_ROOT, source


def load_mirror_config(
    source: str | Path = "physx_cpu",
    *,
    configs_root: str | Path | None = None,
) -> MirrorConfig:
    """加载 Mirror mode 及其六个 canonical leaf profiles。"""

    root, normalized_source = _root_and_source(
        source, mode="mirror", configs_root=configs_root
    )
    reader = _ConfigurationGraphReader(root)
    mode_path = reader.mode_path(normalized_source, mode="mirror")
    mode, references, compute = mirror_mode_from_mapping(
        load_yaml_mapping(mode_path), label=str(mode_path)
    )

    scene_raw = reader.profile(
        group="scenes",
        reference=references.scene,
        root_key="scene",
        namespace="mirror",
    )
    physics_raw = reader.profile(
        group="physics", reference=references.physics, root_key="physics"
    )
    control_raw = reader.profile(
        group="control", reference=references.control, root_key="control"
    )
    hybrid_control_raw = (
        reader.profile(
            group="control",
            reference=references.hybrid_control,
            root_key="hybrid_force_position",
            provenance_key="hybrid_control",
        )
        if references.hybrid_control is not None
        else None
    )
    curobo_raw = reader.profile(
        group="curobo", reference=references.curobo, root_key="curobo"
    )
    planning_raw = reader.profile(
        group="planning", reference=references.planning, root_key="planning"
    )
    outputs_raw = reader.profile(
        group="outputs", reference=references.outputs, root_key="outputs"
    )

    scene = _resolve_scene_asset_profiles(
        MirrorSceneSettings.from_mapping(scene_raw),
        reader=reader,
    )
    assert isinstance(scene, MirrorSceneSettings)
    if scene.scene_id != Path(references.scene).name:
        raise ConfigurationError(
            "scene.id must match the scene profile stem: "
            f"{scene.scene_id!r} != {Path(references.scene).name!r}"
        )
    # The shared parser returns all four supported physics variants. MirrorConfig
    # rejects PhysX CUDA at runtime; narrow the already-validated support matrix at
    # this constructor boundary for the static checker.
    physics = cast(
        MirrorPhysicsSettings,
        physics_settings_from_mapping(physics_raw),
    )
    control = MirrorControlSettings.from_mapping(control_raw)
    controller_bundles = reader.controller_bundles(
        _effective_controller_bundle_names(
            scene,
            runtime_default=physics.engine,
        )
    )
    return MirrorConfig(
        mode=mode,
        profiles=references,
        compute=compute,
        scene=scene,
        physics=physics,
        control=control,
        curobo=CuroboProfileSettings.from_mapping(curobo_raw),
        planning=MirrorPlanningSettings.from_mapping(planning_raw),
        outputs=MirrorOutputsSettings.from_mapping(outputs_raw),
        hybrid_control=(
            None
            if hybrid_control_raw is None
            else HybridForcePositionSettings.from_mapping(hybrid_control_raw)
        ),
        controller_bundles=controller_bundles,
        sources=reader.immutable_sources(),
    )


def load_kaleidoscope_config(
    source: str | Path = "physx_cuda",
    *,
    configs_root: str | Path | None = None,
) -> KaleidoscopeConfig:
    """加载 PhysX CUDA 或 Newton CUDA Kaleidoscope 配置图。"""

    root, normalized_source = _root_and_source(
        source, mode="kaleidoscope", configs_root=configs_root
    )
    reader = _ConfigurationGraphReader(root)
    mode_path = reader.mode_path(normalized_source, mode="kaleidoscope")
    mode_document = load_yaml_mapping(mode_path)
    mode, references, compute, environments = kaleidoscope_mode_from_mapping(
        mode_document, label=str(mode_path)
    )

    raw_profiles = {
        "scene": reader.profile(
            group="scenes",
            reference=references.scene,
            root_key="scene",
            namespace="kaleidoscope",
        ),
        "physics": reader.profile(
            group="physics", reference=references.physics, root_key="physics"
        ),
        "task": reader.profile(
            group="tasks", reference=references.task, root_key="task"
        ),
    }
    for name, payload in raw_profiles.items():
        validate_kaleidoscope_closure(payload, label=f"{reader.sources[name]}")

    physics = physics_settings_from_mapping(raw_profiles["physics"])
    if not isinstance(physics, (PhysxCudaSettings, NewtonCudaSettings)):
        raise ConfigurationError(
            "Kaleidoscope physics profile must use PhysX CUDA or Newton CUDA; "
            f"got engine={physics.engine!r}, execution={physics.execution!r}"
        )
    scene = _resolve_scene_asset_profiles(
        KaleidoscopeSceneSettings.from_mapping(raw_profiles["scene"]),
        reader=reader,
    )
    assert isinstance(scene, KaleidoscopeSceneSettings)
    task = KaleidoscopeTaskSettings.from_mapping(raw_profiles["task"])
    if scene.scene_id != Path(references.scene).name:
        raise ConfigurationError(
            "scene.id must match the scene profile stem: "
            f"{scene.scene_id!r} != {Path(references.scene).name!r}"
        )
    if task.task_id != Path(references.task).name:
        raise ConfigurationError(
            "task.id must match the task profile stem: "
            f"{task.task_id!r} != {Path(references.task).name!r}"
        )
    curobo: CuroboProfileSettings | None = None
    needs_curobo = not isinstance(
        task.action,
        (JointControlActionSettings, JointDeltaActionSettings),
    )
    if needs_curobo != (references.curobo is not None):
        raise ConfigurationError(
            "Kaleidoscope profiles.curobo must be used only for EE/linear action"
        )
    if references.curobo is not None:
        # task 只表达动作语义；数值实现由 mode composition 显式选择。joint_delta 不读取
        # profile，从而不会在环境构造时创建无用 cuRobo context。
        curobo_raw = reader.profile(
            group="curobo",
            reference=references.curobo,
            root_key="curobo",
        )
        validate_kaleidoscope_closure(
            curobo_raw,
            label=f"{reader.sources['curobo']}",
        )
        curobo = CuroboProfileSettings.from_mapping(curobo_raw)
    controller_bundles = reader.controller_bundles(
        _effective_controller_bundle_names(
            scene,
            runtime_default=physics.engine,
        )
    )
    return KaleidoscopeConfig(
        mode=mode,
        profiles=references,
        compute=compute,
        environments=environments,
        scene=scene,
        physics=physics,
        task=task,
        curobo=curobo,
        controller_bundles=controller_bundles,
        sources=reader.immutable_sources(),
    )


def load_kaleidoscope_viewport_config(
    source: str | Path = "kaleidoscope",
    *,
    configs_root: str | Path | None = None,
) -> KaleidoscopeViewportSettings:
    """加载独立的 Kaleidoscope human viewport 启动配置。

    字符串 ``source`` 是 ``configs/visualization`` 下不带扩展名的 profile 引用；
    调用方如需传入文件路径必须显式使用 ``Path``。该对象不会附加到训练配置，因此
    不参与 episode snapshot/clone 的兼容性 fingerprint。
    """

    root = (
        _DEFAULT_CONFIGS_ROOT
        if configs_root is None
        else Path(configs_root).expanduser().resolve()
    )
    group_root = _within(
        root / "visualization",
        root,
        label="visualization profiles root",
    )
    if isinstance(source, Path):
        candidate = source.expanduser()
        if not candidate.is_absolute():
            cwd_candidate = candidate.resolve()
            candidate = cwd_candidate if cwd_candidate.exists() else root / candidate
        path = _within(candidate, group_root, label="viewport config")
        if path.suffix != ".yaml":
            raise ConfigurationError(
                f"viewport config must use the .yaml extension: {path}"
            )
    else:
        parts = _profile_reference(source, label="viewport profile")
        path = _within(
            group_root.joinpath(*parts).with_suffix(".yaml"),
            group_root,
            label="viewport profile",
        )
    document = load_yaml_mapping(path)
    require_keys(document, required={"viewport"}, label=str(path))
    payload = strict_mapping(document["viewport"], label=f"{path}:viewport")
    return KaleidoscopeViewportSettings.from_mapping(
        payload,
        label=f"{path}:viewport",
    )


def load_skrl_training_settings(
    source: str | Path = "tblock_push_v1_ppo",
    *,
    configs_root: str | Path | None = None,
) -> SkrlTrainingSettings:
    """从 catalog 唯一文件边界加载一个严格的 skrl 训练 profile。"""

    root = (
        _DEFAULT_CONFIGS_ROOT
        if configs_root is None
        else Path(configs_root).expanduser().resolve()
    )
    group_root = _within(
        root / "training" / "skrl",
        root,
        label="training profiles root",
    )
    if isinstance(source, Path):
        candidate = source.expanduser()
        if not candidate.is_absolute():
            cwd_candidate = candidate.resolve()
            candidate = cwd_candidate if cwd_candidate.exists() else root / candidate
        path = _within(candidate, group_root, label="training config")
        if path.suffix != ".yaml":
            raise ConfigurationError(
                f"training config must use the .yaml extension: {path}"
            )
    else:
        parts = _profile_reference(source, label="training profile")
        path = _within(
            group_root.joinpath(*parts).with_suffix(".yaml"),
            group_root,
            label="training profile",
        )
    document = load_yaml_mapping(path)
    require_keys(document, required={"training"}, label=str(path))
    return SkrlTrainingSettings.from_mapping(
        document["training"],
        label=f"{path}:training",
    )


__all__ = [
    "load_kaleidoscope_config",
    "load_kaleidoscope_viewport_config",
    "load_mirror_config",
    "load_skrl_training_settings",
    "load_yaml_mapping",
]

from __future__ import annotations

from types import SimpleNamespace

import pytest

from linkerbot_sim.isaac import provenance as provenance_module
from linkerbot_sim.isaac.provenance import (
    KitExtensionProvenance,
    ModuleProvenance,
    PhysicsEngineProvenance,
    RuntimeProvenance,
    _distribution_vcs_commit,
    _module_provenance,
    _nvidia_driver_version,
    collect_runtime_provenance,
    format_runtime_provenance,
    validate_target_runtime,
)


def _target_provenance(**overrides: object) -> RuntimeProvenance:
    values = {
        "python": "3.12.3",
        "executable": "/tmp/python",
        "platform": "linux",
        "physics_backend": "physx",
        "physics_execution": "cpu",
        "physics_engines": (PhysicsEngineProvenance("physx", True),),
        "isaacsim": ModuleProvenance(
            "isaacsim", "6.0.1.0", "isaacsim", "/isaac/__init__.py"
        ),
        "torch": ModuleProvenance(
            "torch", "2.11.0+cu128", "torch", "/torch", "2.11.0+cu128"
        ),
        "warp": ModuleProvenance(
            "warp-lang",
            "1.13.0",
            "warp",
            "/isaac/extscache/omni.warp.core-1.13.0/warp/__init__.py",
            "1.13.0",
        ),
        "newton": ModuleProvenance(
            "newton", "1.2.1", "newton", "/newton/__init__.py", "1.2.1"
        ),
        "mujoco_warp": ModuleProvenance(
            "mujoco-warp",
            "3.8.0.3",
            "mujoco_warp",
            "/mujoco_warp/__init__.py",
            "3.8.0.3",
        ),
        "pxr": ModuleProvenance("isaacsim-kernel", "6.0.1.0", "pxr.Usd", "/pxr/Usd.so"),
        "torch_cuda": "12.8",
        "cuda_available": True,
        "cuda_device": 0,
        "cuda_device_name": "GPU",
        "cuda_device_capability": (12, 0),
        "nvidia_driver": "580.159.03",
        "usd_core_installed": False,
        "kit_extensions": (
            KitExtensionProvenance(
                "isaacsim.simulation_app", "2.18.4", "/simulation-app"
            ),
            KitExtensionProvenance("isaacsim.core.api", "6.0.1", "/core"),
            KitExtensionProvenance("isaacsim.asset.importer.urdf", "3.0.0", "/urdf"),
            KitExtensionProvenance("isaacsim.asset.importer.mjcf", "3.0.0", "/mjcf"),
            KitExtensionProvenance(
                "isaacsim.sensors.experimental.rtx", "0.1.0", "/rtx"
            ),
            KitExtensionProvenance(
                "omni.warp.core",
                "1.13.0",
                "/isaac/extscache/omni.warp.core-1.13.0",
            ),
            KitExtensionProvenance("omni.kit.loop-isaac", "1.6.0", "/loop"),
            KitExtensionProvenance("omni.kit.usd.layers", "2.6.1", "/usd-layers"),
            KitExtensionProvenance("omni.physics.physx", "110.1.0", "/physx"),
        ),
        "curobo": ModuleProvenance("nvidia-curobo", "0.8.0", "curobo", "/curobo"),
        "curobo_backend": "cuda_core",
        "curobo_commit": "4ea77366ca48ee453e7df139e39fa6532af49f3b",
    }
    values.update(overrides)
    if "physics_execution" not in overrides and values["physics_backend"] == "newton":
        values["physics_execution"] = "cuda"
    return RuntimeProvenance(**values)  # type: ignore[arg-type]


def test_validate_target_runtime_accepts_release_stack() -> None:
    validate_target_runtime(_target_provenance(), require_curobo=True)


def test_validate_target_runtime_accepts_exclusive_newton_runtime_stack() -> None:
    base = _target_provenance()
    provenance = _target_provenance(
        physics_backend="newton",
        physics_engines=(),
        kit_extensions=tuple(
            extension
            for extension in base.kit_extensions
            if extension.name
            not in {
                "isaacsim.core.api",
                "isaacsim.core.cloner",
                "isaacsim.core.experimental.prims",
                "isaacsim.core.simulation_manager",
                "isaacsim.core.utils",
                "omni.physics.physx",
                "omni.physics.stageupdate",
                "isaacsim.sensors.physx",
            }
        ),
    )

    validate_target_runtime(
        provenance,
        expected_physics_backend="newton",
        physics_execution="cuda",
    )


def test_validate_kaleidoscope_accepts_renderer_free_newton_runtime_stack() -> None:
    base = _target_provenance()
    forbidden = {
        "isaacsim.core.api",
        "isaacsim.core.cloner",
        "isaacsim.core.experimental.prims",
        "isaacsim.core.simulation_manager",
        "isaacsim.core.utils",
        "isaacsim.sensors.experimental.rtx",
        "isaacsim.sensors.physx",
        "omni.physics.physx",
        "omni.physics.stageupdate",
    }
    provenance = _target_provenance(
        physics_backend="newton",
        physics_engines=(),
        kit_extensions=tuple(
            extension
            for extension in base.kit_extensions
            if extension.name not in forbidden
        ),
    )

    validate_target_runtime(
        provenance,
        expected_physics_backend="newton",
        physics_execution="cuda",
        experience_family="kaleidoscope",
    )


def test_validate_kaleidoscope_accepts_camera_free_newton_runtime_viewport() -> None:
    base = _target_provenance()
    forbidden = {
        "isaacsim.core.api",
        "isaacsim.core.cloner",
        "isaacsim.core.experimental.prims",
        "isaacsim.core.simulation_manager",
        "isaacsim.core.utils",
        "isaacsim.sensors.experimental.rtx",
        "isaacsim.sensors.physx",
        "omni.physics.physx",
        "omni.physics.stageupdate",
    }
    provenance = _target_provenance(
        physics_backend="newton",
        physics_engines=(),
        kit_extensions=tuple(
            extension
            for extension in base.kit_extensions
            if extension.name not in forbidden
        )
        + (
            KitExtensionProvenance("omni.hydra.rtx", "1.0.0", "/rtx"),
            KitExtensionProvenance(
                "omni.kit.viewport.utility", "1.0.0", "/viewport-utility"
            ),
            KitExtensionProvenance("omni.kit.viewport.window", "1.0.0", "/viewport"),
        ),
    )

    validate_target_runtime(
        provenance,
        expected_physics_backend="newton",
        physics_execution="cuda",
        experience_family="kaleidoscope",
        rendering_required=True,
    )


def test_validate_mirror_requires_camera_navigation_in_newton_render_closure() -> None:
    base = _target_provenance()
    excluded = {
        "isaacsim.core.api",
        "isaacsim.core.cloner",
        "isaacsim.core.experimental.prims",
        "isaacsim.core.simulation_manager",
        "isaacsim.core.utils",
        "omni.physics.physx",
        "omni.physics.stageupdate",
        "isaacsim.sensors.physx",
    }
    base_extensions = tuple(
        extension for extension in base.kit_extensions if extension.name not in excluded
    )
    render_extensions = (
        KitExtensionProvenance("omni.hydra.rtx", "1.0.0", "/rtx"),
        KitExtensionProvenance(
            "omni.kit.viewport.utility", "1.0.0", "/viewport-utility"
        ),
        KitExtensionProvenance("omni.kit.viewport.window", "1.0.0", "/viewport"),
        KitExtensionProvenance("omni.syntheticdata", "1.0.0", "/syntheticdata"),
        KitExtensionProvenance(
            "omni.usd.schema.omni_lens_distortion", "1.0.0", "/lens"
        ),
    )
    without_navigation = _target_provenance(
        physics_backend="newton",
        physics_engines=(),
        kit_extensions=base_extensions + render_extensions,
    )

    with pytest.raises(RuntimeError, match="omni.kit.manipulator.camera"):
        validate_target_runtime(
            without_navigation,
            expected_physics_backend="newton",
            physics_execution="cuda",
            experience_family="mirror",
            rendering_required=True,
        )

    with_navigation = _target_provenance(
        physics_backend="newton",
        physics_engines=(),
        kit_extensions=base_extensions
        + render_extensions
        + (
            KitExtensionProvenance(
                "omni.kit.manipulator.camera", "110.0.0", "/camera-manipulator"
            ),
        ),
    )
    validate_target_runtime(
        with_navigation,
        expected_physics_backend="newton",
        physics_execution="cuda",
        experience_family="mirror",
        rendering_required=True,
    )


def test_validate_kaleidoscope_viewport_still_rejects_syntheticdata() -> None:
    base = _target_provenance(newton=None, mujoco_warp=None)
    extensions = tuple(
        extension
        for extension in base.kit_extensions
        if extension.name != "isaacsim.sensors.experimental.rtx"
    ) + (
        KitExtensionProvenance("omni.hydra.rtx", "1.0.0", "/rtx"),
        KitExtensionProvenance(
            "omni.kit.viewport.utility", "1.0.0", "/viewport-utility"
        ),
        KitExtensionProvenance("omni.kit.viewport.window", "1.0.0", "/viewport"),
        KitExtensionProvenance("omni.syntheticdata", "1.0.0", "/synthetic"),
    )

    with pytest.raises(RuntimeError, match="forbidden.*omni.syntheticdata"):
        validate_target_runtime(
            _target_provenance(
                newton=None,
                mujoco_warp=None,
                kit_extensions=extensions,
            ),
            expected_physics_backend="physx",
            experience_family="kaleidoscope",
            rendering_required=True,
        )


def test_validate_target_runtime_rejects_registry_engine_in_newton_runtime() -> None:
    provenance = _target_provenance(
        physics_backend="newton",
        physics_engines=(PhysicsEngineProvenance("physx", True),),
        kit_extensions=tuple(
            extension
            for extension in _target_provenance().kit_extensions
            if extension.name != "omni.physics.physx"
        ),
    )

    with pytest.raises(RuntimeError, match=r"physics registry engines=\['physx'\]"):
        validate_target_runtime(
            provenance,
            expected_physics_backend="newton",
            physics_execution="cuda",
        )


def test_validate_target_runtime_rejects_inactive_registry_engine_in_newton_runtime() -> (
    None
):
    base = _target_provenance()
    provenance = _target_provenance(
        physics_backend="newton",
        physics_engines=(PhysicsEngineProvenance("physx", False),),
        kit_extensions=tuple(
            extension
            for extension in base.kit_extensions
            if extension.name
            not in {
                "isaacsim.core.api",
                "omni.physics.physx",
            }
        ),
    )

    with pytest.raises(
        RuntimeError, match=r"physics registry engines=\['physx'\].*expected \[\]"
    ):
        validate_target_runtime(
            provenance,
            expected_physics_backend="newton",
            physics_execution="cuda",
        )


def test_validate_target_runtime_rejects_prefixed_physics_extension_in_newton_runtime() -> (
    None
):
    base = _target_provenance()
    provenance = _target_provenance(
        physics_backend="newton",
        physics_engines=(),
        kit_extensions=tuple(
            extension
            for extension in base.kit_extensions
            if extension.name
            not in {
                "isaacsim.core.api",
                "omni.physics.physx",
            }
        )
        + (
            KitExtensionProvenance(
                "omni.physx.fabric",
                "110.1.0",
                "/physx-fabric",
            ),
        ),
    )

    with pytest.raises(
        RuntimeError, match="forbidden enabled extensions: omni.physx.fabric"
    ):
        validate_target_runtime(
            provenance,
            expected_physics_backend="newton",
            physics_execution="cuda",
        )


def test_validate_target_runtime_rejects_physx_owner_extension_in_newton_runtime() -> (
    None
):
    base = _target_provenance()
    direct_extensions = tuple(
        extension
        for extension in base.kit_extensions
        if extension.name
        not in {
            "isaacsim.core.api",
            "isaacsim.core.cloner",
            "isaacsim.core.experimental.prims",
            "isaacsim.core.simulation_manager",
            "isaacsim.core.utils",
            "omni.physics.physx",
            "omni.physics.stageupdate",
            "isaacsim.sensors.physx",
        }
    )
    provenance = _target_provenance(
        physics_backend="newton",
        physics_engines=(),
        kit_extensions=direct_extensions
        + (KitExtensionProvenance("omni.physics.physx", "110.1.0", "/physx"),),
    )

    with pytest.raises(
        RuntimeError, match="forbidden enabled extensions: omni.physics.physx"
    ):
        validate_target_runtime(
            provenance,
            expected_physics_backend="newton",
            physics_execution="cuda",
        )


def test_validate_target_runtime_rejects_physx_dependency_carrier_in_newton_runtime() -> (
    None
):
    base = _target_provenance()
    provenance = _target_provenance(
        physics_backend="newton",
        physics_engines=(),
        kit_extensions=tuple(
            extension
            for extension in base.kit_extensions
            if extension.name not in {"isaacsim.core.api", "omni.physics.physx"}
        )
        + (KitExtensionProvenance("isaacsim.core.cloner", "1.7.3", "/cloner"),),
    )

    with pytest.raises(
        RuntimeError,
        match="forbidden enabled extensions: isaacsim.core.cloner",
    ):
        validate_target_runtime(
            provenance,
            expected_physics_backend="newton",
            physics_execution="cuda",
        )


def test_validate_target_runtime_rejects_backend_mismatch() -> None:
    try:
        validate_target_runtime(
            _target_provenance(),
            expected_physics_backend="newton",
            physics_execution="cuda",
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "physics_backend='physx' expected 'newton'" in message
        assert "Newton physics registry engines=['physx'] expected []" in message
        assert "Newton has forbidden enabled extensions" in message
    else:
        raise AssertionError("backend mismatch was accepted")


def test_validate_target_runtime_rejects_multiple_active_registry_engines() -> None:
    provenance = _target_provenance(
        physics_engines=(
            PhysicsEngineProvenance("newton", True),
            PhysicsEngineProvenance("physx", True),
        )
    )

    try:
        validate_target_runtime(provenance)
    except RuntimeError as exc:
        assert "active physics registry engines=['newton', 'physx']" in str(exc)
    else:
        raise AssertionError("multiple active physics engines were accepted")


def test_validate_target_runtime_rejects_missing_active_registry_engine() -> None:
    provenance = _target_provenance(
        physics_engines=(PhysicsEngineProvenance("physx", False),)
    )

    try:
        validate_target_runtime(provenance)
    except RuntimeError as exc:
        assert "active physics registry engines=[] expected ['physx']" in str(exc)
    else:
        raise AssertionError("missing active physics engine was accepted")


def test_validate_target_runtime_rejects_shadow_warp_module() -> None:
    provenance = _target_provenance(
        warp=ModuleProvenance(
            "warp-lang",
            "1.13.0",
            "warp",
            "/shadow/warp/__init__.py",
            "1.13.0",
        )
    )

    try:
        validate_target_runtime(provenance)
    except RuntimeError as exc:
        assert "warp module path is outside enabled omni.warp.core extension" in str(
            exc
        )
    else:
        raise AssertionError("shadow Warp module was accepted")


def test_validate_target_runtime_rejects_newton_module_version_mismatch() -> None:
    base = _target_provenance()
    provenance = _target_provenance(
        physics_backend="newton",
        physics_engines=(),
        kit_extensions=tuple(
            extension
            for extension in base.kit_extensions
            if extension.name
            not in {
                "isaacsim.core.api",
                "omni.physics.physx",
                "isaacsim.sensors.physx",
            }
        ),
        newton=ModuleProvenance(
            "newton", "1.2.1", "newton", "/newton/__init__.py", "1.3.0"
        ),
    )

    try:
        validate_target_runtime(
            provenance,
            expected_physics_backend="newton",
            physics_execution="cuda",
        )
    except RuntimeError as exc:
        assert "newton.__version__='1.3.0' expected '1.2.1'" in str(exc)
    else:
        raise AssertionError("Newton module/distribution mismatch was accepted")


def test_validate_physx_runtime_does_not_require_newton_modules() -> None:
    validate_target_runtime(
        _target_provenance(newton=None, mujoco_warp=None),
        expected_physics_backend="physx",
    )


def test_validate_kaleidoscope_accepts_renderer_free_physx_closure() -> None:
    base = _target_provenance()
    provenance = _target_provenance(
        newton=None,
        mujoco_warp=None,
        kit_extensions=tuple(
            extension
            for extension in base.kit_extensions
            if extension.name != "isaacsim.sensors.experimental.rtx"
        ),
    )

    validate_target_runtime(
        provenance,
        expected_physics_backend="physx",
        experience_family="kaleidoscope",
    )


def test_validate_kaleidoscope_accepts_core_usdrt_delegate_dependency() -> None:
    """Core API 的 USD Runtime delegate 不等同于启用 RTX renderer。"""

    base = _target_provenance()
    provenance = _target_provenance(
        newton=None,
        mujoco_warp=None,
        kit_extensions=tuple(
            extension
            for extension in base.kit_extensions
            if extension.name != "isaacsim.sensors.experimental.rtx"
        )
        + (
            KitExtensionProvenance(
                "omni.hydra.usdrt_delegate",
                "8.0.0",
                "/isaac/extscache/omni.hydra.usdrt_delegate",
            ),
        ),
    )

    validate_target_runtime(
        provenance,
        expected_physics_backend="physx",
        experience_family="kaleidoscope",
    )


def test_validate_kaleidoscope_rejects_hydra_rtx_renderer() -> None:
    base = _target_provenance()
    provenance = _target_provenance(
        newton=None,
        mujoco_warp=None,
        kit_extensions=tuple(
            extension
            for extension in base.kit_extensions
            if extension.name != "isaacsim.sensors.experimental.rtx"
        )
        + (KitExtensionProvenance("omni.hydra.rtx", "1.0.0", "/rtx"),),
    )

    with pytest.raises(RuntimeError, match="forbidden.*omni.hydra.rtx"):
        validate_target_runtime(
            provenance,
            expected_physics_backend="physx",
            experience_family="kaleidoscope",
        )


def test_validate_kaleidoscope_rejects_renderer_extension() -> None:
    with pytest.raises(RuntimeError, match="forbidden.*experimental.rtx"):
        validate_target_runtime(
            _target_provenance(newton=None, mujoco_warp=None),
            expected_physics_backend="physx",
            experience_family="kaleidoscope",
        )


def test_collect_physx_provenance_never_imports_newton_modules(monkeypatch) -> None:
    imported: list[str] = []
    selected_devices: list[tuple[str, int]] = []
    modules = {
        "torch": SimpleNamespace(
            __file__="/torch/__init__.py",
            __version__="2.11.0",
            version=SimpleNamespace(cuda="12.8"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda index: (
                    selected_devices.append(("name", index)) or "GPU 3"
                ),
                get_device_capability=lambda index: (
                    selected_devices.append(("capability", index)) or (12, 0)
                ),
            ),
        ),
        "warp": SimpleNamespace(
            __file__="/warp/__init__.py",
            __version__="1.13.0",
        ),
        "pxr.Usd": SimpleNamespace(__file__="/isaac/pxr/Usd.so"),
        "isaacsim": SimpleNamespace(__file__="/isaac/isaacsim/__init__.py"),
    }

    def import_module(name: str):
        imported.append(name)
        if name in {"newton", "mujoco_warp"}:
            raise AssertionError(f"PhysX imported forbidden module {name}")
        return modules[name]

    monkeypatch.setattr(provenance_module.importlib, "import_module", import_module)
    monkeypatch.setattr(provenance_module, "active_physics_backend", lambda: "physx")
    monkeypatch.setattr(provenance_module, "_physics_engine_provenance", lambda: ())
    monkeypatch.setattr(provenance_module, "_kit_extension_provenance", lambda: ())
    monkeypatch.setattr(provenance_module, "_nvidia_driver_version", lambda: None)
    monkeypatch.setattr(
        provenance_module, "_distribution_installed", lambda _name: False
    )
    monkeypatch.setattr(
        provenance_module.importlib.metadata,
        "version",
        lambda distribution: {
            "torch": "2.11.0",
            "warp-lang": "1.13.0",
            "isaacsim": "6.0.1.0",
            "isaacsim-kernel": "6.0.1.0",
        }[distribution],
    )

    result = collect_runtime_provenance(
        cuda_device=3,
        include_curobo=False,
        physics_execution="cpu",
    )

    assert result.physics_backend == "physx"
    assert result.cuda_device == 3
    assert result.cuda_device_name == "GPU 3"
    assert result.cuda_device_capability == (12, 0)
    assert selected_devices == [("name", 3), ("capability", 3)]
    assert result.newton is None
    assert result.mujoco_warp is None
    assert "newton" not in imported
    assert "mujoco_warp" not in imported


def test_collect_provenance_requires_explicit_nonnegative_cuda_device() -> None:
    with pytest.raises(TypeError, match="cuda_device"):
        collect_runtime_provenance()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="non-negative"):
        collect_runtime_provenance(cuda_device=-1)


def test_validate_target_runtime_reports_all_mismatches() -> None:
    provenance = _target_provenance(
        python="3.11.9",
        warp=ModuleProvenance("warp-lang", "1.15.0", "warp", "/warp", "1.15.0"),
        cuda_available=False,
        curobo_backend="pybind",
        curobo_commit="wrong",
        usd_core_installed=True,
        kit_extensions=(KitExtensionProvenance("omni.warp.core", "1.8.2", "/isaac51"),),
    )
    try:
        validate_target_runtime(provenance, require_curobo=True)
    except RuntimeError as exc:
        message = str(exc)
        assert "python='3.11.9'" in message
        assert "warp='1.15.0'" in message
        assert "curobo_backend='pybind'" in message
        assert "curobo_commit='wrong'" in message
        assert "usd-core must not be installed" in message
        assert "omni.warp.core='1.8.2'" in message
        assert "missing Kit extensions" in message
        assert "torch.cuda.is_available() is false" in message
    else:
        raise AssertionError("invalid runtime provenance was accepted")


def test_format_runtime_provenance_is_stable_json() -> None:
    text = format_runtime_provenance(_target_provenance())
    assert '"physics_backend": "physx"' in text
    assert '"warp"' in text
    assert '"nvidia_driver": "580.159.03"' in text


def test_reads_nvidia_driver_version_from_proc_format(tmp_path) -> None:
    version = tmp_path / "version"
    version.write_text(
        "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  "
        "580.159.03  Release Build\n",
        encoding="utf-8",
    )

    assert _nvidia_driver_version(version) == "580.159.03"


def test_reads_curobo_commit_from_direct_url(monkeypatch) -> None:
    class Distribution:
        @staticmethod
        def read_text(name: str) -> str:
            assert name == "direct_url.json"
            return (
                '{"vcs_info":{"vcs":"git","commit_id":'
                '"4ea77366ca48ee453e7df139e39fa6532af49f3b"}}'
            )

    monkeypatch.setattr(
        "linkerbot_sim.isaac.provenance.importlib.metadata.distribution",
        lambda _name: Distribution(),
    )

    assert _distribution_vcs_commit("nvidia-curobo") == (
        "4ea77366ca48ee453e7df139e39fa6532af49f3b"
    )


def test_module_provenance_records_distribution_and_module_versions(
    monkeypatch,
) -> None:
    module = type(
        "FakeModule",
        (),
        {"__version__": "1.2.1", "__file__": "/runtime/newton/__init__.py"},
    )()
    monkeypatch.setattr(
        "linkerbot_sim.isaac.provenance.importlib.metadata.version",
        lambda distribution: "1.2.1" if distribution == "newton" else "unexpected",
    )

    result = _module_provenance("newton", "newton", module)

    assert result.version == "1.2.1"
    assert result.module_version == "1.2.1"
    assert result.path == "/runtime/newton/__init__.py"

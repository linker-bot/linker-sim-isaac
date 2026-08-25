# Python API Reference

Language: [English](python-api.md) | [中文](../../zh-CN/reference/python-api.md)

The stable Python surface is organized by product. This page lists only supported
facade exports; implementation modules in the source map are not compatibility
surfaces. Importing a facade does not start Kit, initialize CUDA, or read YAML.

## `linkerbot_sim`

| Symbol | Contract |
| --- | --- |
| `REPO_ROOT` | Checkout root. |

## `linkerbot_sim.configuration`

Import from `linkerbot_sim.configuration`:

| Symbol | Contract |
| --- | --- |
| `ComputeSettings` | Mode-root owner of the single CUDA device index. |
| `MirrorConfig` | Frozen resolved Mirror graph. |
| `KaleidoscopeConfig` | Frozen resolved Kaleidoscope graph. |
| `KaleidoscopeEnvironmentSettings` | Mode-root owner of Kaleidoscope environment count, path naming, and base origin. |
| `KaleidoscopeViewportSettings` | Separate launch-only human viewport configuration. |
| `NewtonCpuSettings` | Mirror Newton/CPU leaf settings; physics stays on CPU while root compute still selects cuRobo/RTX GPU. |
| `NewtonCudaSettings` | Public Newton/CUDA physics leaf settings; device and world count are derived later. |
| `PhysicsEngine` | Public physics engine selector: `physx` or `newton`. |
| `PhysicsExecution` | Public execution selector: `cpu` or `cuda`. |
| `PhysicsSettings` | Strict union of schema-valid physics leaves; product roots enforce the narrower runtime capability matrix. |
| `PhysxCpuSettings` | Mirror PhysX/CPU leaf settings. |
| `PhysxCudaSettings` | Kaleidoscope PhysX/CUDA leaf settings. |
| `SkrlTrainingSettings` | Strict downstream skrl training leaf; its device is inherited from the environment. |
| `load_mirror_config(source="physx_cpu", *, configs_root=None)` | Load and validate a Mirror graph. |
| `load_kaleidoscope_config(source="physx_cuda", *, configs_root=None)` | Load and validate a Kaleidoscope graph. |
| `load_kaleidoscope_viewport_config(source="kaleidoscope", *, configs_root=None)` | Strictly load the launch-only viewport profile without changing episode fingerprints. |
| `load_skrl_training_settings(source="tblock_push_v1_ppo", *, configs_root=None)` | Strictly load a downstream skrl training profile through the catalog I/O boundary. |
| `semantic_config_payload(config)` | Return the canonical JSON-compatible graph with provenance excluded. |
| `semantic_config_fingerprint(config)` | Return the canonical semantic SHA-256 used by validation and snapshot compatibility. |

## `linkerbot_sim.mirror`

Import from `linkerbot_sim.mirror`:

| Symbol | Purpose |
| --- | --- |
| `MirrorConfig` | Re-export of the strict product configuration. |
| `MirrorRuntime` | Owner-thread runtime and lifecycle root. |
| `MirrorController` | Typed request dispatch and admission coordination. |
| `create_mirror_runtime(config, *, assembly_factory=None)` | Construct the product resource graph; queue capacities come only from the strict control profile. |
| `run_mirror(runtime, *, endpoints=(), poll_timeout_s=None, should_stop=None, on_ready=None, before_session_close=None, max_iterations=None, close_on_exit=True)` | Run the owner-thread event loop; an omitted poll timeout comes from the strict control profile, and the optional close hook uses the same drained pre-session boundary as `MirrorRuntime.close`. |

### `MirrorRuntime`

Key methods and properties:

| Member | Contract |
| --- | --- |
| `physics_runtime` | Borrowed concrete runtime from the unique session. |
| `step(render=False)` | Advance physics exactly once; optional render occurs afterward. |
| `render()` | Run one explicit render transaction; fails when rendering is disabled. |
| `get_state()` | Return an owned state mapping. |
| `set_state(state, strict=True)` | Transactional state mutation and collision-dirty mark. |
| `capture_snapshot()` | Return an owned versioned snapshot mapping. |
| `restore_snapshot(snapshot, label_map=None, strict=True)` | Restore and mark collision data dirty. |
| `reset(hold_after_reset=True)` | Restore configured initial state, rebase the timeline to step zero, and by default compile one synchronized arm/hand hold of `control.idle_step_duration_s` through the normal executor/render path. |
| `get_control_mode()` | Read immutable initial/active mode, generation, supported modes, and all-robot scope. |
| `set_control_mode(mode, expected_generation=None)` | Transactionally switch all robots between complete motions without rebuilding the runtime. |
| `status()` | Product, physics, collision, scene, and shutdown status. |
| `close(*, before_session_close=None)` | Idempotent dependency-ordered close returning `MirrorCloseReport`; the optional supervisor hook runs only after ingress, outputs/cameras, planners, controllers, and views have stopped, while the native session is still alive. |

All members that touch runtime state require the thread that created the runtime.
There is deliberately no generic `world` property because Newton is not an
Isaac World.

### Embedded Protocol Dispatch

`MirrorController.dispatch(request)` is for owner-thread embedding and tests.
Ingress workers use `submit_and_wait`, while the owner loop calls `process_next`.
Construct requests with the exact v1 or v2 envelope; do not instantiate
internal timeline DTOs as a wire API.

## `linkerbot_sim.kaleidoscope`

Import from `linkerbot_sim.kaleidoscope`:

| Symbol | Purpose |
| --- | --- |
| `KaleidoscopeConfig` | Strict resolved PhysX CUDA or project-owned Newton graph. |
| `TorchKaleidoscopeEnv` | Native CUDA tensor vector environment. |
| `KaleidoscopeTrainingPort` | Minimal runtime-checkable protocol consumed by trainers. |
| `KaleidoscopeEpisodeSnapshot` | Owned GPU episode snapshot. |
| `ControlModeState` | Immutable initial/active mode, generation, supported modes, and scope. |
| `ControlModeChange` | Result of an idempotent or real mode change. |
| `ControlModeGenerationConflict` | Optimistic generation precondition failed. |
| `ControlModeIncompatibleError` | Requested mode or trajectory is incompatible with the fixed action. |
| `ControlModeLockedError` | Runtime phase or SAME_STEP transaction forbids switching. |
| `ControlModeSwitchError` | Forward mode transaction failed and rolled back. |
| `ControlModeRollbackError` | Rollback failed and the runtime entered permanent fail-stop. |
| `GymnasiumKaleidoscopeAdapter` | Explicit NumPy `VectorEnv` boundary. |
| `make_torch_env(...)` | Production native environment composition. |
| `make_viewport_env(..., viewport=None, viewport_profile="kaleidoscope")` | Explicit PhysX/Newton human viewport for one `selected_env`. |
| `make_gymnasium_env(..., viewport_profile="kaleidoscope")` | Native environment plus NumPy adapter; the viewport profile is used only for human rendering. |
| `register_gymnasium_envs()` | Idempotently register the maintained environment ID. |

`TorchKaleidoscopeEnv` supports `reset`, `reset_idx`, `step`, `get_control_mode`,
`set_control_mode`, `get_state`, `set_state`,
`snapshot`, `restore_snapshot`, `clone_state`, `render`, `is_running`, and `close`.
`render` is available only on an environment constructed by `make_viewport_env` and
never runs implicitly inside a training step. It also implements the
same-decision token methods consumed by the training port; application code should
prefer the higher-level skrl adapter rather than managing tokens directly.
Mode mutation is native-only and is deliberately absent from
`KaleidoscopeTrainingPort`, Gymnasium, and skrl.

The native bootstrap is independent of Gymnasium: importing it or calling
`make_torch_env()` does not import Gymnasium. `GymnasiumKaleidoscopeAdapter` is imported
inside `make_gymnasium_env()` only, so the optional training dependency is required at
the explicit NumPy boundary rather than on the native Torch path.

See [Kaleidoscope API](kaleidoscope-api.md) for tensor shapes, selector rules, action
variants, and failure semantics.

The `physx_cuda` and `newton_cuda` profiles expose the same facade. Backend
selection is a cold composition choice; Newton uses the project's multi-world
owner, not the Isaac Newton extension.

## `linkerbot_sim.training.skrl`

Import from `linkerbot_sim.training.skrl`:

| Symbol | Contract |
| --- | --- |
| `SkrlTorchAdapter` | CUDA same-decision environment wrapper with terminal observation preservation. |
| `CudaRolloutMemory` | CUDA selector and mini-batch implementation. |
| `FinalObservationPPO` | skrl 2.1 PPO replacement with final-observation bootstrap and source guards. |
| `make_skrl_trainer` | Construct a trainer from the maintained training profile. |

This layer consumes `KaleidoscopeTrainingPort`; it does not own Isaac, a physics
world, or scene handles.

## `linkerbot_sim.snapshots`

| Symbol | Contract |
| --- | --- |
| `SceneSnapshot` | Versioned CPU/NumPy scene snapshot. |
| `load_scene_snapshot` | Read and validate a persisted snapshot. |
| `save_scene_snapshot` | Atomically save a scene snapshot. |
| `validate_scene_snapshot` | Validate without mutating a runtime. |

Mirror adapters and Kaleidoscope episode snapshots remain product-owned and are not
re-exported from this facade.

## `linkerbot_sim.backends.curobo`

This is the capability-oriented numerical backend facade. Callers own and close the
contexts they construct. Kaleidoscope composes only device batch kinematics; Mirror
may additionally compose planning and collision-world capabilities. The frozen
export inventory below is the compatibility surface.

| Symbol | Contract |
| --- | --- |
| `CuroboConfig` | Strict shared backend configuration root. |
| `CuroboContext` | Mirror planning and collision-capable resource owner. |
| `CuroboDeviceBatchIKSolver` | CUDA-only batched IK adapter used by Kaleidoscope. |
| `CuroboKinematicsContext` | Planner-free and collision-world-free kinematics owner. |
| `create_kinematics_context` | Construct the narrow Kaleidoscope kinematics capability. |
| `curobo_config_from_profiles` | Project typed robot and cuRobo profiles plus the canonical CUDA device into one numerical backend configuration. |

YAML profile loading is intentionally absent from this backend facade. The
`linkerbot_sim.configuration` catalog resolves `configs/curobo/`, injects the mode
root CUDA device, and passes a typed projection to the numerical backend.

## Isaac Infrastructure Facade

Import infrastructure types from `linkerbot_sim.isaac` only when building a custom
composition root:

- `IsaacAppSpec`, `IsaacRenderSpec`, and `IsaacSessionSpec`;
- `IsaacPhysxCpuSpec` and `IsaacPhysxCudaSpec`;
- `IsaacNewtonCpuSpec` and `IsaacNewtonCudaSpec`;
- `IsaacSession` and `create_isaac_session_from_spec`.

Product consumers normally use the Mirror or Kaleidoscope factories. An
`IsaacSession` owns its app, stage, and physics runtime and is the only closer for
those resources.

## Import And Threading Rules

- Import facades before Kit only for pure types and factories; call heavy factories
  after the deployment has accepted the EULA.
- Do not access Omni/Isaac objects from Mirror ingress workers.
- Do not move Kaleidoscope tensors to CPU inside `step`, reset, reward, termination,
  state, or skrl hot paths.
- Do not import internal product builders as stable application APIs.

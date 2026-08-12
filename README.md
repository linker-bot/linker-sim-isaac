# LinkerHand Simulation

Choose a language:

- [English README](README.en.md)
- [中文 README](README.zh-CN.md)

The workspace has two deliberately separate products:

- **Mirror** maps one real workcell into one interactive simulation world.
- **Kaleidoscope** replicates one task scene into CUDA-native reinforcement-learning
  environments. PhysX CUDA and the project's multi-world Newton runtime both keep a
  renderer-free training path and provide an explicit single-environment debug viewport.

The product factory selects exactly one of seven formal Kit experiences:

| Product | Engine / execution | Render closure | Formal Kit experience |
| --- | --- | --- | --- |
| Mirror | PhysX / CPU | Controlled by the outputs profile | `apps/linkerbot_sim.mirror.physx.python.kit` |
| Mirror | Newton / CPU or CUDA | Disabled | `apps/linkerbot_sim.mirror.newton.python.kit` |
| Mirror | Newton / CPU or CUDA | Enabled | `apps/linkerbot_sim.mirror.newton_render.python.kit` |
| Kaleidoscope | PhysX / CUDA | Training headless | `apps/linkerbot_sim.kaleidoscope.physx_cuda.python.kit` |
| Kaleidoscope | Newton / CUDA | Training headless | `apps/linkerbot_sim.kaleidoscope.newton.python.kit` |
| Kaleidoscope | PhysX / CUDA | Explicit selected-environment viewport | `apps/linkerbot_sim.kaleidoscope.physx_cuda_viewport.python.kit` |
| Kaleidoscope | Newton / CUDA | Explicit selected-environment viewport | `apps/linkerbot_sim.kaleidoscope.newton_viewport.python.kit` |

Kaleidoscope viewport launch configuration is independent from the task/physics graph and its
snapshot fingerprint. It adds no camera, SyntheticData, Replicator, recording, or telemetry.
Public mode profiles are `mirror/{physx_cpu,newton_cpu,newton_cuda}` and
`kaleidoscope/{physx_cuda,newton_cuda}`.
Their scene selectors are respectively `mirror/scene3` and
`kaleidoscope/tblock_push`; the unqualified `scene.id` remains the stable identity
inside each scene file.

Kaleidoscope's native/debug `step` performs one synchronous done-scalar guard so an
unreset terminal row is rejected before physics advances. The skrl SAME_STEP path does
not execute that guard and keeps the training step on CUDA.

Documentation:

- [English documentation](docs/en/index.md)
- [中文文档](docs/zh-CN/index.md)

Start with the [runtime and API chooser](docs/en/getting-started/choose-runtime-and-api.md)
or its [中文版](docs/zh-CN/getting-started/choose-runtime-and-api.md).

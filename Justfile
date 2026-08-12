set shell := ["bash", "-uc"]

uv_dev := "uv run --frozen --extra dev"
uv_simulation := "uv run --frozen --all-extras"
python := ".venv/bin/python"
python_dev := ".venv-dev/bin/python"

alias check := quality

whitespace:
    git diff --check
    git diff --cached --check

format:
    {{uv_dev}} ruff format .

format-check:
    {{uv_dev}} ruff format --check .

lint:
    {{uv_dev}} ruff check .

check-docs:
    # 本地重构会先产生尚未暂存的新文档；只接受 Git 未忽略的 worktree 文件，避免质量
    # 门禁迫使开发工具改写用户 index。干净 CI checkout 中该集合仍等于 tracked files。
    {{uv_dev}} python scripts/check_markdown_links.py --allow-untracked-existing

test:
    {{uv_simulation}} coverage erase
    {{uv_simulation}} coverage run -m pytest -q
    {{uv_simulation}} coverage report

test-pure:
    {{python_dev}} -m pytest -q --ignore=tests/test_curobo_device_batch_ik.py --ignore=tests/test_curobo_kinematics_context.py --ignore=tests/test_kaleidoscope_composition.py --ignore=tests/test_kaleidoscope_isaac_views.py --ignore=tests/test_kaleidoscope_physics_smoke.py --ignore=tests/test_kaleidoscope_physx_ports.py --ignore=tests/test_kaleidoscope_scene_assembly.py --ignore=tests/test_kaleidoscope_viewer.py --ignore=tests/test_physx_gpu_memory_budget_smoke.py --ignore=tests/test_skrl_cuda_integration.py --ignore=tests/test_target_architecture.py

# 架构清单的手写规则与机械 inventory 分离。开发中移动文件后显式执行 write；
# 发布门禁只执行 check，并要求所有 facade 已完成最终冻结。
update-architecture:
    {{python_dev}} scripts/update_architecture_inventory.py --write

check-architecture-inventory:
    {{python_dev}} scripts/update_architecture_inventory.py --check --require-final

test-architecture: check-architecture-inventory
    {{uv_dev}} pytest -q tests/test_target_architecture.py tests/test_documented_module_map.py tests/test_hardcoded_configuration_allowlist.py tests/test_configuration_modes.py tests/test_kaleidoscope_gpu_residency_source.py

test-kaleidoscope:
    {{uv_simulation}} pytest -q tests/test_kaleidoscope_*.py tests/test_curobo_device_batch_ik.py tests/test_curobo_kinematics_context.py

test-training:
    {{uv_simulation}} pytest -q tests/test_skrl_cuda_integration.py

test-gpu-kaleidoscope:
    {{python}} -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for the Kaleidoscope GPU gate"'
    {{uv_simulation}} pytest -q tests/test_kaleidoscope_gpu_contracts.py tests/test_kaleidoscope_composition.py tests/test_kaleidoscope_isaac_views.py tests/test_kaleidoscope_kit.py tests/test_kaleidoscope_physics_smoke.py tests/test_kaleidoscope_physx_ports.py tests/test_kaleidoscope_newton_ports.py tests/test_kaleidoscope_scene_assembly.py tests/test_isaac_replicated_newton_scene.py tests/test_curobo_device_batch_ik.py tests/test_curobo_kinematics_context.py tests/test_skrl_cuda_integration.py

# 两个真实 physics owner 必须由不同 Kit 进程启动和关闭；这条门禁只属于有 CUDA/Isaac 的
# simulation CI，不能成为普通 CPU quality 的隐式依赖。
smoke-kaleidoscope:
    {{python}} -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for the Kaleidoscope physics smoke"'
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/smoke_kaleidoscope_physics.py --profile physx_cuda --num-envs 2 --steps 2
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/smoke_kaleidoscope_physics.py --profile newton_cuda --num-envs 2 --steps 2 --exercise-training-adapters
    # 这两条仍启动正式 Newton Kit/runtime；临时 action variant 经过同一个 strict loader，
    # 分别证明真实 cuRobo batch IK 与固定 waypoint 同步直线动作，不创建 planner/collision world。
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/smoke_kaleidoscope_physics.py --profile newton_cuda --num-envs 2 --steps 1 --action-mode ee_delta_position
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/smoke_kaleidoscope_physics.py --profile newton_cuda --num-envs 2 --steps 1 --action-mode ee_linear_path_position

# 正式 Newton multi-world composition 的容量正确性门禁。它验证 256 worlds 的构造、推进与
# state/snapshot/clone 事务，不替代下面独立的进程显存采样。
smoke-kaleidoscope-newton-capacity:
    {{python}} -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for the Kaleidoscope Newton capacity smoke"'
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/smoke_kaleidoscope_physics.py --profile newton_cuda --num-envs 256 --steps 2

# 显存预算是独立的冷路径验收：它需要更长 warmup/steady 窗口和 NVML 进程级采样，
# 不得被塞进每拍训练或普通 CPU quality。Newton 的容量由 per-world model 配置约束，
# 当前这条门禁只消费 physx/cuda.yaml 中的 GpuMemoryBudget。
smoke-kaleidoscope-memory:
    {{python}} -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for the Kaleidoscope GPU memory smoke"'
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/smoke_physx_gpu_memory_budget.py --profile physx_cuda --num-envs 2 --warmup-steps 8 --steady-steps 16

test-mirror:
    {{uv_simulation}} pytest -q tests/test_mirror_*.py

test-isaac:
    {{uv_simulation}} pytest -q tests/test_isaac_*.py tests/test_check_isaac_runtime.py tests/test_physics_runtime_ownership.py tests/test_runtime_provenance.py tests/test_simulation_app_lifecycle.py

# 七个正式 Kit 各自在独立进程中启动；Mirror 的两个共享 Newton Kit 还分别验证 CPU/CUDA
# session 变体，因此总计九个受监督进程。两个 Kaleidoscope viewport Kit 仍不创建
# camera/SyntheticData/Replicator；产品级 viewer smoke 负责实际场景与画面。
smoke-runtime-kits:
    {{python}} -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for the formal Kit runtime smoke"'
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/check_isaac_runtime.py --profile mirror-physx-cpu --cuda-device 0
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/check_isaac_runtime.py --profile mirror-newton-cpu --cuda-device 0
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/check_isaac_runtime.py --profile mirror-newton-cpu-render --cuda-device 0
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/check_isaac_runtime.py --profile mirror-newton-cuda --cuda-device 0
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/check_isaac_runtime.py --profile mirror-newton-cuda-render --cuda-device 0
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/check_isaac_runtime.py --profile kaleidoscope-physx-cuda --cuda-device 0
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/check_isaac_runtime.py --profile kaleidoscope-newton-cuda --cuda-device 0
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/check_isaac_runtime.py --profile kaleidoscope-physx-cuda-viewport --cuda-device 0
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/check_isaac_runtime.py --profile kaleidoscope-newton-cuda-viewport --cuda-device 0

# Mirror 的四个正式 mode profile 必须验证完整单场景，而不只验证空 Kit closure。Newton
# profiles 的 outputs 开启相机，因此还会选择 render Kit 并验证 physics-to-USD 同步。
smoke-mirror:
    {{python}} -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for the Mirror physics smoke"'
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/smoke_mirror_physics.py --profile physx_cpu --steps 8
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/smoke_mirror_physics.py --profile physx_cpu_hybrid --steps 8
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/smoke_mirror_physics.py --profile newton_cpu --steps 8
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src {{python}} scripts/smoke_mirror_physics.py --profile newton_cuda --steps 8

test-simulation: test-isaac test-gpu-kaleidoscope smoke-runtime-kits smoke-mirror smoke-kaleidoscope smoke-kaleidoscope-newton-capacity smoke-kaleidoscope-memory

validate-config:
    {{uv_dev}} python scripts/validate_mode_config.py --mode mirror --profile physx_cpu
    {{uv_dev}} python scripts/validate_mode_config.py --mode mirror --profile physx_cpu_hybrid
    {{uv_dev}} python scripts/validate_mode_config.py --mode mirror --profile newton_cpu
    {{uv_dev}} python scripts/validate_mode_config.py --mode mirror --profile newton_cuda
    {{uv_dev}} python scripts/validate_mode_config.py --mode kaleidoscope --profile physx_cuda
    {{uv_dev}} python scripts/validate_mode_config.py --mode kaleidoscope --profile newton_cuda

quality: whitespace format-check lint check-docs validate-config test-architecture test-pure

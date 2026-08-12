#!/usr/bin/env bash
# 仿真/GPU 门禁与普通质量门禁分进程执行，防止 Kit、CUDA 和 usd-core 污染彼此环境。
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"

exec just test-simulation

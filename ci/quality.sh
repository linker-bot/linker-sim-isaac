#!/usr/bin/env bash
# CPU 质量门禁：只使用仓库声明的 Justfile 入口，避免本地与自动化环境维护两套命令。
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"

exec just quality

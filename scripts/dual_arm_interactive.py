#!/usr/bin/env python3
"""双 AR5+L6 实时交互式 JSON motion runtime 的薄 CLI 入口。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 仓库采用 src-layout。直接运行本脚本时，Python 只会自动把 scripts/ 放进 sys.path；
# 因此这里显式把 src/ 加入搜索路径，让导入始终指向当前工作区代码。
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.app.interactive.dual_arm import main  # noqa: E402


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"DUAL_ARM_INTERACTIVE_FAILED {type(exc).__name__}: {exc}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

#!/usr/bin/env python3
"""Tiled Scene interactive step-control 的薄 CLI 入口。

真实实现放在 ``linkerbot_sim.app.interactive.tiled_scene``，这里仅负责让用户可以继续
直接执行 ``scripts/tiled_scene_interactive.py``。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.app.interactive.tiled_scene import main  # noqa: E402


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"TILED_SCENE_INTERACTIVE_FAILED {type(exc).__name__}: {exc}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

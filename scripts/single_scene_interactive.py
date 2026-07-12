#!/usr/bin/env python3
"""Single Scene interactive simulation entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.app.interactive.single_scene import main  # noqa: E402


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(
            f"SINGLE_SCENE_INTERACTIVE_FAILED {type(exc).__name__}: {exc}", flush=True
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

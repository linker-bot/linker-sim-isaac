#!/usr/bin/env python3
"""Validate and inspect a runtime profile without starting Isaac Sim."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.configs.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

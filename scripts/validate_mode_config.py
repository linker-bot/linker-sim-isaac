#!/usr/bin/env python3
"""在启动 Kit 前加载并审计 Mirror/Kaleidoscope canonical 配置图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.configuration import (  # noqa: E402
    load_kaleidoscope_config,
    load_mirror_config,
    semantic_config_fingerprint,
    semantic_config_payload,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mirror", "kaleidoscope"), required=True)
    parser.add_argument("--profile")
    arguments = parser.parse_args(argv)
    loader = (
        load_mirror_config if arguments.mode == "mirror" else load_kaleidoscope_config
    )
    profile = arguments.profile or (
        "physx_cpu" if arguments.mode == "mirror" else "physx_cuda"
    )
    config = loader(profile)
    print(
        json.dumps(
            {
                "event": "mode_config_validated",
                "mode": arguments.mode,
                "profile": profile,
                "fingerprint": semantic_config_fingerprint(config),
                "sources": {
                    name: str(path) for name, path in sorted(config.sources.items())
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _json_value(value: object) -> object:
    """保留旧测试/脚本私有导入；canonical 实现只有 configuration facade 一份。"""

    return semantic_config_payload(value)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""根据对象配置生成 capsule rope USD 资产。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from manipulation_project.app.launch import launch_simulation_app
from manipulation_project.objects.capsule_rope import (
    CapsuleRopeConfig,
    write_capsule_rope_asset,
)
from manipulation_project.utils.config import load_yaml
from manipulation_project.utils.paths import repo_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/objects/capsule_rope.yaml")
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """脚本主入口。"""

    args = parse_args()
    config = CapsuleRopeConfig.from_mapping(load_yaml(args.config))
    simulation_app = launch_simulation_app(gui=False)
    try:
        output = write_capsule_rope_asset(
            config, repo_path(args.output) if args.output is not None else None
        )
        print(
            "BUILD_CAPSULE_ROPE_ASSET_OK "
            f"asset={output} prim_path={config.prim_path} root_path={config.root_path} "
            f"segments={config.segments} shape={config.shape}",
            flush=True,
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

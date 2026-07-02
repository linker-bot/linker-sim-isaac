#!/usr/bin/env python3
"""根据 tools 侧生成配置生成 T block USD 资产。

该脚本是离线资产生成入口，而不是仿真运行入口。它会启动一个 headless SimulationApp，
原因是 USD/PhysX schema 写入依赖 Isaac/Omni 扩展已经加载；生成完成后只保存 USD 文件。
运行时 static/material/import 配置放在 configs/objects，不由本脚本写入。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.app.launch import launch_simulation_app
from linkerbot_sim.utils.config import load_yaml
from linkerbot_sim.utils.paths import repo_path
from tools.object_assets.rigid.tblock.builder import (
    TBlockAssetConfig,
    write_tblock_asset,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tools/object_assets/rigid/tblock/config.yaml"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """脚本主入口。"""

    args = parse_args()
    config = TBlockAssetConfig.from_mapping(load_yaml(args.config))
    simulation_app = launch_simulation_app(gui=False)
    try:
        output = write_tblock_asset(
            config, repo_path(args.output) if args.output is not None else None
        )
        print(
            "BUILD_T_BLOCK_ASSET_OK "
            f"asset={output} root_path={config.root_path} "
            f"stem_size={config.stem_size} cap_size={config.cap_size}",
            flush=True,
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

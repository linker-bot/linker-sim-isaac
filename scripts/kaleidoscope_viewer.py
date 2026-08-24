#!/usr/bin/env python3
"""启动可交互 Kaleidoscope viewport，并在窗口关闭前持续推进 GPU 环境。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
import json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("physx_cuda", "newton_cuda"),
        default="physx_cuda",
    )
    parser.add_argument("--viewport-profile", default="kaleidoscope")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--selected-env", type=int, default=0)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="0 means run continuously until the window is closed or Ctrl+C is pressed",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-source", choices=("hold", "random"), default="hold")
    parser.add_argument("--action-scale", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.selected_env < 0 or args.selected_env >= args.num_envs:
        parser.error("--selected-env must be within [0, num_envs)")
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    if not 0.0 <= args.action_scale <= 1.0:
        parser.error("--action-scale must be within [0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from linkerbot_sim.configuration import (
        load_kaleidoscope_config,
        load_kaleidoscope_viewport_config,
    )
    from linkerbot_sim.kaleidoscope import make_viewport_env

    config = load_kaleidoscope_config(args.profile)
    viewport = replace(
        load_kaleidoscope_viewport_config(args.viewport_profile),
        selected_env=args.selected_env,
    )
    env = make_viewport_env(
        config=config,
        viewport=viewport,
        num_envs=args.num_envs,
    )
    completed = 0
    successful = False
    try:
        import torch

        generator = torch.Generator(device=env.device)
        generator.manual_seed(args.seed)
        env.reset(seed=args.seed)
        env.render()
        while env.is_running() and (args.steps == 0 or completed < args.steps):
            if args.action_source == "random":
                actions = (
                    torch.rand(
                        (env.num_envs, env.action_dim),
                        device=env.device,
                        dtype=torch.float32,
                        generator=generator,
                    )
                    * 2.0
                    - 1.0
                ) * args.action_scale
            else:
                actions = torch.zeros(
                    (env.num_envs, env.action_dim),
                    device=env.device,
                    dtype=torch.float32,
                )
            token = env.begin_same_step()
            env.step_same_step(token, actions)
            completed += 1
            if completed % env.render_every_n_steps == 0:
                env.render()
            env.complete_same_step(token)
    except KeyboardInterrupt:
        successful = True
    else:
        successful = True
    finally:
        # standalone Kit 的 fast shutdown 不会把控制流交还给 Python。成功 marker 必须
        # 在 close 前 flush，供 shell/CI 与人工日志区分正常 viewer 结束和 native crash。
        if successful:
            print(
                "LINKERBOT_KALEIDOSCOPE_VIEWPORT_VALID "
                + json.dumps(
                    {
                        "num_envs": args.num_envs,
                        "physics_engine": str(config.physics.engine),
                        "physics_execution": str(config.physics.execution),
                        "profile": args.profile,
                        "selected_env": args.selected_env,
                        "steps": completed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

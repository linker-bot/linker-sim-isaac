"""Cross-process portable reset smoke using real, independently constructed Kits."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from linkerbot_sim.configuration import load_kaleidoscope_config
from linkerbot_sim.kaleidoscope import (
    KaleidoscopeResetDescription,
    make_torch_env,
    reset_json,
    sample_reset_description,
)
from scripts.runtime_worker_supervisor import in_runtime_worker, run_supervised_worker

MARKER = "LINKERBOT_PORTABLE_RESET_OK"


def config_for(profile, hand):
    if hand == "l6":
        return load_kaleidoscope_config(profile)
    with tempfile.TemporaryDirectory(prefix="linkerbot-portable-reset-") as temp:
        root = Path(temp) / "configs"
        shutil.copytree(ROOT / "configs", root)
        path = root / "scenes/kaleidoscope/tblock_push.yaml"
        data = yaml.safe_load(path.read_text())
        for robot in data["scene"]["robots"]:
            robot["robot_profile"] = robot["robot_profile"].replace("l6", "o6")
            robot["label"] = robot["label"].replace("l6", "o6")
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return load_kaleidoscope_config(profile, configs_root=root)


def run(args):
    # Different actual configuration batch sizes, not only a factory override.
    config = config_for(args.profile, args.hand)
    config = replace(
        config,
        environments=replace(
            config.environments, num_envs=2 if args.phase == "produce" else 3
        ),
    )
    env = make_torch_env(config=config)
    prefix = args.output / f"{args.profile}-{args.hand}"
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        def state(ids):
            return {
                name: value.detach().cpu().numpy().copy()
                for name, value in env.get_state(ids).items()
            }

        def check(actual, expected):
            assert actual.keys() == expected.keys()
            for name in actual:
                np.testing.assert_allclose(
                    actual[name], expected[name], rtol=0, atol=3e-6, err_msg=name
                )

        clock = env.runtime.physics_runtime.simulation_time
        if args.phase == "produce":
            env.reset(seed=91)
            template = env.get_reset_template()
            prefix.with_suffix(".template.json").write_text(reset_json(template))
            description = sample_reset_description(
                template, seed=91, source_env_ids=(1,)
            )
            prefix.with_suffix(".json").write_text(reset_json(description))
            ids = torch.tensor([1], device=env.device)
            initial = state(ids)
            # Match the actual ordinary GPU reset, not only two portable restores.
            observed = env.runtime.views.refresh(ids)
            np.testing.assert_allclose(
                observed.joint_positions.cpu().numpy()[0],
                description.rows[0].joint_positions,
                rtol=0,
                atol=3e-6,
            )
            np.savez_compressed(prefix.with_suffix(".initial.npz"), **initial)
            env.reset_idx(ids)
            np.savez_compressed(prefix.with_suffix(".next.npz"), **state(ids))
        else:
            env.reset(seed=567)
            for _ in range(2):
                env.step(torch.zeros((env.num_envs, env.action_dim), device=env.device))
            clock = env.runtime.physics_runtime.simulation_time
            ids = torch.tensor([2], device=env.device)
            others = torch.tensor([0, 1], device=env.device)
            before = state(others)
            description = KaleidoscopeResetDescription.from_mapping(
                json.loads(prefix.with_suffix(".json").read_text())
            )
            bad = replace(description, template_fingerprint="0" * 64)
            all_before = state(None)
            try:
                env.restore_reset_description(bad, target_env_ids=ids)
            except ValueError:
                pass
            else:
                raise AssertionError("incompatible description was accepted")
            assert not env.runtime._failed
            for name, values in state(None).items():
                np.testing.assert_array_equal(values, all_before[name], err_msg=name)
            env.restore_reset_description(description, target_env_ids=ids)
            with np.load(
                prefix.with_suffix(".initial.npz"), allow_pickle=False
            ) as archive:
                check(state(ids), dict(archive))
            for name, values in state(others).items():
                np.testing.assert_array_equal(values, before[name], err_msg=name)
            env.reset_idx(ids)
            with np.load(
                prefix.with_suffix(".next.npz"), allow_pickle=False
            ) as archive:
                check(state(ids), dict(archive))
        assert env.runtime.physics_runtime.simulation_time == clock
        result = dict(
            profile=args.profile,
            hand=args.hand,
            phase=args.phase,
            num_envs=env.num_envs,
            clock=clock,
            status="passed",
        )
        prefix.with_suffix(f".{args.phase}-result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(MARKER + " " + json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("physx_cuda", "newton_cuda"), required=True
    )
    parser.add_argument("--hand", choices=("l6", "o6"), default="o6")
    parser.add_argument("--phase", choices=("produce", "consume"))
    parser.add_argument("--output", type=Path, default=ROOT / "logs/portable-reset")
    args = parser.parse_args()
    if in_runtime_worker():
        run(args)
    else:
        for phase in (args.phase,) if args.phase else ("produce", "consume"):
            argv = [
                "--profile",
                args.profile,
                "--hand",
                args.hand,
                "--phase",
                phase,
                "--output",
                str(args.output),
            ]
            result = run_supervised_worker(
                script_path=Path(__file__),
                argv=argv,
                required_markers=[MARKER],
                success_marker=MARKER + "_SUPERVISED",
                timeout_s=600,
            )
            if result:
                raise SystemExit(result)

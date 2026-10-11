#!/usr/bin/env python3
"""Validate portable Kaleidoscope reset descriptions on a real physics owner."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.configuration import load_kaleidoscope_config  # noqa: E402
from linkerbot_sim.kaleidoscope import (  # noqa: E402
    load_reset_scene_description,
    sample_reset_scene_description,
    save_reset_scene_description,
)


SUCCESS_MARKER = "LINKERBOT_KALEIDOSCOPE_PORTABLE_RESET_SMOKE_OK"
FAILURE_MARKER = "LINKERBOT_KALEIDOSCOPE_PORTABLE_RESET_SMOKE_FAILED"
PRODUCED_MARKER = "LINKERBOT_KALEIDOSCOPE_PORTABLE_RESET_PRODUCED"
CROSS_PROCESS_MARKER = "LINKERBOT_KALEIDOSCOPE_PORTABLE_RESET_CROSS_PROCESS_OK"
PROFILES = ("physx_cuda", "newton_cuda")
PHASES = ("orchestrate", "produce", "restore")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-id", type=int, default=12)
    parser.add_argument("--reset-index", type=int, default=4)
    parser.add_argument("--phase", choices=PHASES, default="orchestrate")
    parser.add_argument("--description-path", type=Path)
    return parser.parse_args(argv)


def _require_cuda_tensor(value: object, *, label: str) -> None:
    import torch

    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"{label} is not a Torch tensor")
    if value.device.type != "cuda":
        raise RuntimeError(f"{label} must be CUDA-resident, got {value.device}")


def _expected_reset_values(description: object, *, device: object) -> dict[str, object]:
    import torch

    robots = getattr(description, "robots")
    dynamic_object = getattr(description, "dynamic_object")
    task = getattr(description, "task")
    return {
        "robot.target": torch.tensor(
            [[value for robot in robots for value in robot.joint_targets]],
            device=device,
            dtype=torch.float32,
        ),
        "object.pose_local_wxyz": torch.tensor(
            [[*dynamic_object.position_xyz, *dynamic_object.orientation_wxyz]],
            device=device,
            dtype=torch.float32,
        ),
        "object.com_velocity": torch.zeros((1, 6), device=device),
        "task.goal_position": torch.tensor(
            [task.goal_position],
            device=device,
            dtype=torch.float32,
        ),
        "task.goal_yaw": torch.tensor(
            [task.goal_yaw],
            device=device,
            dtype=torch.float32,
        ),
    }


def _assert_applied_state(
    state: Mapping[str, object],
    description: object,
    *,
    device: object,
) -> None:
    import torch

    expected = _expected_reset_values(description, device=device)
    for name, value in state.items():
        _require_cuda_tensor(value, label=name)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not bool(torch.all(torch.isfinite(value)).item()):
                raise RuntimeError(f"{name} contains non-finite values")

    torch.testing.assert_close(state["robot.target"], expected["robot.target"])
    torch.testing.assert_close(
        state["object.pose_local_wxyz"],
        expected["object.pose_local_wxyz"],
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        state["object.com_velocity"],
        expected["object.com_velocity"],
    )
    torch.testing.assert_close(
        state["task.goal_position"],
        expected["task.goal_position"],
    )
    torch.testing.assert_close(
        state["task.goal_yaw"],
        expected["task.goal_yaw"],
    )

    generator = getattr(description, "generator")
    if int(state["rng.key"].item()) != generator.rng_key:
        raise RuntimeError("restored RNG key differs from the description")
    if int(state["rng.counter"].item()) != generator.rng_counter_after:
        raise RuntimeError("restored RNG counter differs from the description")
    if int(state["task.episode_length"].item()) != 0:
        raise RuntimeError("portable reset advanced the episode decision counter")
    if int(state["task.episode_physics_steps"].item()) != 0:
        raise RuntimeError("portable reset advanced physics time")


def produce_description(
    *,
    profile: str,
    seed: int,
    sample_id: int,
    reset_index: int,
    path: Path,
) -> dict[str, object]:
    config = load_kaleidoscope_config(profile)
    description = sample_reset_scene_description(
        config,
        seed=seed,
        sample_id=sample_id,
        reset_index=reset_index,
    )
    saved = save_reset_scene_description(description, path)
    forbidden = ("torch", "gymnasium", "skrl", "newton", "omni", "isaacsim")
    heavy_modules = sorted(
        name for name in sys.modules if name.split(".", 1)[0] in forbidden
    )
    if heavy_modules:
        raise RuntimeError(f"pure producer loaded heavy modules: {heavy_modules}")
    result = {
        "profile": profile,
        "path": str(saved),
        "seed": seed,
        "sample_id": sample_id,
        "reset_index": reset_index,
        "rng_key": description.generator.rng_key,
        "rng_counter_after": description.generator.rng_counter_after,
        "heavy_modules_loaded": False,
    }
    print(PRODUCED_MARKER + " " + json.dumps(result, sort_keys=True), flush=True)
    return result


def restore_description(*, profile: str, path: Path) -> dict[str, object]:
    import torch

    from linkerbot_sim.kaleidoscope import make_torch_env

    config = load_kaleidoscope_config(profile)
    loaded = load_reset_scene_description(path)

    env = make_torch_env(config=config, num_envs=1)
    failed = False
    try:
        observations, info = env.reset_from_description(loaded, env_id=0)
        _require_cuda_tensor(observations, label="observations")
        if not bool(torch.all(torch.isfinite(observations)).item()):
            raise RuntimeError("reset observations contain non-finite values")

        env_ids = info.get("env_ids")
        _require_cuda_tensor(env_ids, label="info.env_ids")
        if env_ids.tolist() != [0]:
            raise RuntimeError(f"reset returned unexpected env ids: {env_ids}")

        selected = torch.tensor([0], device=env.device, dtype=torch.int64)
        state = env.get_state(
            selected,
            fields=(
                "robot.target",
                "object.pose_local_wxyz",
                "object.com_velocity",
                "task.goal_position",
                "task.goal_yaw",
                "rng.key",
                "rng.counter",
                "task.episode_length",
                "task.episode_physics_steps",
            ),
        )
        _assert_applied_state(state, loaded, device=env.device)

        env.reset_idx(selected)
        continued = env.get_state(selected, fields=("rng.counter",))
        continued_counter = continued["rng.counter"]
        _require_cuda_tensor(continued_counter, label="continued.rng.counter")
        draws_per_reset = sum(len(robot.joint_names) for robot in loaded.robots) + 6
        expected_counter = loaded.generator.rng_counter_after + draws_per_reset
        if int(continued_counter.item()) != expected_counter:
            raise RuntimeError(
                "ordinary reset did not continue the restored RNG sequence: "
                f"actual={int(continued_counter.item())}, "
                f"expected={expected_counter}"
            )

        result = {
            "profile": profile,
            "device": str(env.device),
            "num_envs": env.num_envs,
            "seed": loaded.generator.seed,
            "sample_id": loaded.generator.sample_id,
            "reset_index": loaded.generator.reset_index,
            "rng_key": loaded.generator.rng_key,
            "rng_counter_after": loaded.generator.rng_counter_after,
            "rng_counter_after_next_reset": expected_counter,
            "json_round_trip_verified": True,
            "physical_reset_verified": True,
            "rng_continuation_verified": True,
        }
        print(SUCCESS_MARKER + " " + json.dumps(result, sort_keys=True), flush=True)
        return result
    except BaseException as exc:
        failed = True
        traceback.print_exception(exc)
        sys.stderr.flush()
        print(
            f"{FAILURE_MARKER} {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    finally:
        env.close(exit_code=1 if failed else 0)


def run_cross_process_smoke(
    *,
    profile: str,
    seed: int,
    sample_id: int,
    reset_index: int,
) -> None:
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="linkerbot-portable-reset-") as directory:
        path = Path(directory) / "reset.json"
        common = [
            sys.executable,
            str(script),
            "--profile",
            profile,
            "--description-path",
            str(path),
        ]
        producer = subprocess.run(
            [
                *common,
                "--phase",
                "produce",
                "--seed",
                str(seed),
                "--sample-id",
                str(sample_id),
                "--reset-index",
                str(reset_index),
            ],
            check=False,
        )
        if producer.returncode != 0:
            raise RuntimeError(
                f"portable reset producer failed with exit code {producer.returncode}"
            )
        consumer = subprocess.run(
            [*common, "--phase", "restore"],
            check=False,
        )
        if consumer.returncode != 0:
            raise RuntimeError(
                f"portable reset physics consumer failed with exit code {consumer.returncode}"
            )
    result = {
        "profile": profile,
        "producer_exit_code": producer.returncode,
        "consumer_exit_code": consumer.returncode,
        "producer_and_consumer_are_distinct_processes": True,
    }
    print(CROSS_PROCESS_MARKER + " " + json.dumps(result, sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile = str(args.profile)
    if args.phase == "orchestrate":
        run_cross_process_smoke(
            profile=profile,
            seed=int(args.seed),
            sample_id=int(args.sample_id),
            reset_index=int(args.reset_index),
        )
    else:
        if args.description_path is None:
            raise ValueError("--description-path is required for internal phases")
        if args.phase == "produce":
            produce_description(
                profile=profile,
                seed=int(args.seed),
                sample_id=int(args.sample_id),
                reset_index=int(args.reset_index),
                path=args.description_path,
            )
        else:
            restore_description(profile=profile, path=args.description_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

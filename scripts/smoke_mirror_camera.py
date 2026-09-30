#!/usr/bin/env python3
"""Verify native camera freshness, paused edits and 120/60 Hz sampling in Mirror."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from linkerbot_sim.configuration import load_mirror_config  # noqa: E402
from linkerbot_sim.mirror import create_mirror_runtime  # noqa: E402
from scripts.runtime_worker_supervisor import (  # noqa: E402
    in_runtime_worker,
    run_supervised_worker,
)

MARKER = "LINKERBOT_MIRROR_CAMERA_RUNTIME_OK"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("physx_cpu", "newton_cpu", "newton_cuda"),
        default="physx_cpu",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--resolution", choices=("320x240", "1920x1080"), default="320x240"
    )
    parser.add_argument("--cameras", type=int, choices=(1, 3), default=1)
    parser.add_argument("--record-root", type=Path)
    return parser.parse_args(argv)


def camera_config(args):
    config = load_mirror_config(args.profile)
    width, height = (int(value) for value in args.resolution.split("x"))
    base = config.scene.cameras[0]
    k = base.intrinsics
    assert k is not None
    sx, sy = width / base.resolution[0], height / base.resolution[1]
    cameras = tuple(
        replace(
            base,
            camera_id=f"probe_{index}",
            prim_path=f"/World/ProbeCamera{index}",
            pose=replace(
                base.pose,
                xyz=(
                    base.pose.xyz[0],
                    base.pose.xyz[1] + index * 0.025,
                    base.pose.xyz[2],
                ),
            ),
            resolution=(width, height),
            frequency_hz=60.0,
            intrinsics=replace(
                k, fx=k.fx * sx, fy=k.fy * sy, cx=k.cx * sx, cy=k.cy * sy
            ),
        )
        for index in range(args.cameras)
    )
    output = config.outputs
    return replace(
        config,
        scene=replace(
            config.scene,
            physics_frequency_hz=120.0,
            render_frequency_hz=60.0,
            cameras=cameras,
            planning_startup="lazy",
        ),
        outputs=replace(
            output,
            render=replace(output.render, gui=args.gui),
            camera=replace(
                output.camera,
                enabled=args.record_root is not None,
                save_root=None
                if args.record_root is None
                else str(args.record_root.resolve()),
                foxglove_live_port=None,
                foxglove_mcap_path=None,
                existing_data_policy="error",
                shutdown_timeout_s=30.0,
            ),
            telemetry=replace(output.telemetry, enabled=False),
        ),
    )


def capture(runtime):
    before = float(runtime.physics_runtime.simulation_time)
    frames = runtime.render()
    after = float(runtime.physics_runtime.simulation_time)
    assert before == after, (before, after)
    metadata = {
        camera.name: camera.get_capture_metadata()
        for camera in runtime.scene_resources.sensor_cameras
    }
    for name, item in metadata.items():
        assert item["native_frame_source"] in ("kit_swh", "kit_render_product_frame"), (
            item
        )
        assert item["physics_time_s"] == before, item
        assert np.isfinite(frames[name]["depth"]).any()
    return frames, metadata


def changed_pixels(before, after):
    return {
        name: int(
            np.count_nonzero(
                ~np.isclose(
                    before[name]["depth"], after[name]["depth"], atol=1e-4, rtol=0
                )
            )
        )
        for name in before
    }


def run(args):
    runtime = create_mirror_runtime(camera_config(args))
    try:
        baseline = runtime.get_state()
        first, initial_meta = capture(runtime)  # no warmup physics steps
        moved = deepcopy(baseline)
        obj = moved["objects"]["Tblock"]
        obj["positions_local"][1] += 0.15
        # A rigid pose edit intentionally uses the maximal-coordinate contract.
        # Newton solver/generalized integration state belongs to the old snapshot.
        for key in (
            "generalized_signature",
            "generalized_q_names",
            "generalized_qd_names",
        ):
            obj[key] = []
        for key in ("generalized_q", "generalized_qd", "generalized_world_origin"):
            obj[key] = None
        moved["metadata"]["info"].pop(
            "linkerbot.snapshot.newton_solver_integration_state", None
        )
        runtime.set_state(moved)
        second, moved_meta = capture(runtime)
        differences = changed_pixels(first, second)
        assert all(count > 10 for count in differences.values()), differences
        for name in initial_meta:
            assert (
                moved_meta[name]["native_frame_id"]
                > initial_meta[name]["native_frame_id"]
            )
        runtime.restore_snapshot(baseline)
        restored, restored_meta = capture(runtime)
        restore_error = changed_pixels(first, restored)
        # Depth should be deterministic; tiny backend transform rounding is tolerated.
        assert all(
            count < max(10, np.asarray(first[name]["depth"]).size * 0.001)
            for name, count in restore_error.items()
        ), restore_error
        samples = []
        last = restored_meta
        for decision in range(12):
            runtime.step(render=(decision % 2 == 1))
            if decision % 2:
                _, meta = capture(runtime)
                for name in meta:
                    assert meta[name]["native_frame_id"] > last[name]["native_frame_id"]
                samples.append(meta)
                last = meta
        times = [
            float(sample[next(iter(sample))]["physics_time_s"]) for sample in samples
        ]
        assert np.allclose(np.diff(times), 1.0 / 60.0, rtol=0, atol=1e-9), times
        runtime.reset(hold_after_reset=False)
        _, reset_meta = capture(runtime)
        report = {
            "profile": args.profile,
            "gui": args.gui,
            "resolution": args.resolution,
            "camera_count": args.cameras,
            "paused_move_changed_pixels": differences,
            "restore_changed_pixels": restore_error,
            "initial": initial_meta,
            "samples": samples,
            "reset": reset_meta,
        }
    except BaseException:
        traceback.print_exc()
        runtime.close()
        raise
    runtime.close(
        before_session_close=lambda _report: print(
            MARKER + " " + json.dumps(report), flush=True
        )
    )


def main():
    args = parse_args()
    if not in_runtime_worker():
        result = run_supervised_worker(
            script_path=Path(__file__),
            argv=sys.argv[1:],
            required_markers=(MARKER,),
            success_marker="LINKERBOT_MIRROR_CAMERA_SMOKE_OK",
        )
        if result == 0 and args.record_root is not None:
            verify_recording(args.record_root, camera_count=args.cameras)
        return result
    run(args)
    return 0


def verify_recording(root: Path, *, camera_count: int) -> None:
    """Check drained files in the supervisor after native fast shutdown."""

    for index in range(camera_count):
        directory = root / f"probe_{index}"
        with (directory / "metadata.jsonl").open() as stream:
            entries = [json.loads(line) for line in stream]
        for modality in ("rgb", "depth"):
            frames = [entry for entry in entries if entry["modality"] == modality]
            assert len(frames) == 6, (directory, modality, len(frames))
            times = [frame["time_s"] for frame in frames]
            assert np.allclose(np.diff(times), 1.0 / 60.0, rtol=0, atol=1e-9)
            ids = [frame["capture"]["native_frame_id"] for frame in frames]
            assert all(right > left for left, right in zip(ids, ids[1:]))
            assert all(
                (directory / frame["relative_path"]).is_file() for frame in frames
            )
            assert all(
                abs(frame["capture"]["physics_time_s"] - frame["time_s"]) < 1e-9
                for frame in frames
            )
    print(
        "LINKERBOT_MIRROR_CAMERA_RECORDING_OK "
        + json.dumps({"cameras": camera_count, "frames_per_modality": 6}),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

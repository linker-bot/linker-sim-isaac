#!/usr/bin/env python3
"""Verify native camera freshness, paused edits and 120/60 Hz sampling in Mirror."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
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
        "--resolution", choices=("320x240", "640x320", "1920x1080"), default="320x240"
    )
    parser.add_argument("--cameras", type=int, choices=(1, 3, 4), default=1)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument(
        "--frequencies",
        type=float,
        nargs="+",
        help="One simulation-time Hz value per camera",
    )
    parser.add_argument(
        "--modalities", choices=("rgb", "depth"), nargs="+", default=("rgb", "depth")
    )
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.frequencies is not None and (
        len(args.frequencies) != args.cameras
        or any(not np.isfinite(hz) or hz <= 0 for hz in args.frequencies)
    ):
        parser.error("--frequencies requires one positive finite value per camera")
    return args


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
            frequency_hz=(args.frequencies[index] if args.frequencies else 60.0),
            modalities=tuple(args.modalities),
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
        for pixels in frames[name].values():
            assert np.isfinite(pixels).any()
    return frames, metadata


def changed_pixels(before, after):
    return {
        name: int(
            np.count_nonzero(
                ~np.isclose(
                    before[name]["depth" if "depth" in before[name] else "rgb"],
                    after[name]["depth" if "depth" in before[name] else "rgb"],
                    atol=1e-4,
                    rtol=0,
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
        if "depth" in args.modalities:
            assert all(
                count < max(10, np.asarray(first[name]["depth"]).size * 0.001)
                for name, count in restore_error.items()
            ), restore_error
        activation = exercise_product_selection(runtime)
        cameras = runtime.scene_resources.sensor_cameras
        counts = {camera.name: 0 for camera in cameras}
        for camera in cameras:
            begin = camera.begin_render_capture

            def counted_begin(camera=camera, begin=begin):
                counts[camera.name] += 1
                begin()

            camera.begin_render_capture = counted_begin
        samples = {camera.name: [] for camera in cameras}
        last_ids = {
            camera.name: camera.get_capture_metadata().get("native_frame_id")
            for camera in cameras
        }
        clock = float(runtime.physics_runtime.simulation_time)
        started = time.perf_counter()
        for decision in range(args.steps):
            runtime.step(render=True)
            for camera in cameras:
                meta = camera.get_capture_metadata()
                native_id = meta.get("native_frame_id")
                if native_id != last_ids[camera.name]:
                    assert (
                        meta["physics_time_s"]
                        == runtime.physics_runtime.simulation_time
                    )
                    samples[camera.name].append(meta)
                    last_ids[camera.name] = native_id
        sampled_wall_s = time.perf_counter() - started
        assert (
            abs(runtime.physics_runtime.simulation_time - clock - args.steps / 120.0)
            < 1e-6
        )
        for camera in cameras:
            expected = (
                expected_frames(args, camera.settings.frequency)
                if args.record_root
                else 0
            )
            assert counts[camera.name] == len(samples[camera.name]) == expected, (
                counts,
                expected,
            )
        automatic_counts = dict(counts)
        observer = getattr(runtime.rendering.cameras.output, "observer", None)
        deadlines = {} if observer is None else dict(observer._next_sample_time)
        capture(runtime)
        assert observer is None or observer._next_sample_time == deadlines
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
            "automatic_capture_counts": automatic_counts,
            "sampled_wall_s": sampled_wall_s,
            "product_selection": activation,
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
            verify_recording(args)
        return result
    run(args)
    return 0


def expected_frames(args, frequency: float) -> int:
    # First sample is the first completed tick; rates above physics sample once per tick.
    return min(args.steps, int(np.floor((args.steps - 1) * frequency / 120 + 1e-7)) + 1)


def exercise_product_selection(runtime):
    """Prove native inactivity, subset freshness and owner-loop service without physics."""
    cameras = runtime.scene_resources.sensor_cameras
    physics = runtime.physics_runtime
    clock = physics.simulation_time

    def ids():
        return {camera.name: camera._frame_tracker.frame_id for camera in cameras}

    # Drain submitted work, then verify that GUI updates cannot keep sensors running.
    for _ in range(8):
        physics.render_update()
    before = ids()
    import omni.kit.viewport.utility as viewport

    main = viewport.get_active_viewport()
    resolution = main.resolution
    try:
        main.resolution = (800, 600)
        for _ in range(8):
            physics.render_update()
    finally:
        main.resolution = resolution
    assert ids() == before, (before, ids())
    for selection in ((), ("missing",), (cameras[0].name, cameras[0].name)):
        try:
            runtime.render(selection)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid camera selection: {selection}")
    selected = cameras[0].name
    first = runtime.render((selected,))
    assert tuple(first) == (selected,)
    frozen = {name: pixels.copy() for name, pixels in first[selected].items()}
    first_id = cameras[0].get_capture_metadata()["native_frame_id"]
    runtime.render((selected,))
    assert cameras[0].get_capture_metadata()["native_frame_id"] > first_id
    for name in before:
        if name != selected:
            assert ids()[name] == before[name]
    for name, pixels in frozen.items():
        np.testing.assert_array_equal(first[selected][name], pixels)
    assert physics.simulation_time == clock
    return {
        "inactive_native_ids": before,
        "selected": selected,
        "after_subset_ids": ids(),
    }


def verify_recording(args) -> None:
    """Decode drained files and verify the actual automatic recording cadence."""
    from PIL import Image

    width, height = (int(value) for value in args.resolution.split("x"))
    for index in range(args.cameras):
        directory = args.record_root / f"probe_{index}"
        with (directory / "metadata.jsonl").open() as stream:
            entries = [json.loads(line) for line in stream]
        frequency = args.frequencies[index] if args.frequencies else 60.0
        for modality in args.modalities:
            frames = [entry for entry in entries if entry["modality"] == modality]
            assert len(frames) == expected_frames(args, frequency), (
                directory,
                modality,
                len(frames),
            )
            times = [frame["time_s"] for frame in frames]
            if frequency <= 120 and 120 % frequency == 0:
                assert np.allclose(np.diff(times), 1.0 / frequency, rtol=0, atol=1e-6)
            ids = [frame["capture"]["native_frame_id"] for frame in frames]
            assert all(right > left for left, right in zip(ids, ids[1:]))
            assert [f["frame_index"] for f in frames] == list(range(len(frames)))
            for frame in frames:
                path = directory / frame["relative_path"]
                assert abs(frame["capture"]["physics_time_s"] - frame["time_s"]) < 1e-9
                if modality == "rgb":
                    with Image.open(path) as pixels:
                        assert pixels.size == (width, height)
                        assert np.asarray(pixels).max() > np.asarray(pixels).min()
                elif path.suffix == ".npz":
                    with np.load(path) as archive:
                        assert archive[archive.files[0]].shape == (height, width)
                else:
                    assert np.load(path).shape == (height, width)
    print(
        "LINKERBOT_MIRROR_CAMERA_RECORDING_OK "
        + json.dumps({"cameras": args.cameras, "steps": args.steps}),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

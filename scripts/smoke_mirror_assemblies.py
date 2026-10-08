#!/usr/bin/env python3
"""Exercise flanged assemblies and physical optical mounts on the selected backend."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import traceback

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from linkerbot_sim.configuration import load_mirror_config  # noqa: E402
from linkerbot_sim.configuration.scenes import CameraSettings  # noqa: E402
from linkerbot_sim.configuration.robots import RobotProfileSettings  # noqa: E402
from linkerbot_sim.configuration.catalog import load_yaml_mapping  # noqa: E402
from linkerbot_sim.mirror import create_mirror_runtime  # noqa: E402
from scripts.runtime_worker_supervisor import in_runtime_worker, run_supervised_worker  # noqa: E402

MARKER = "LINKERBOT_MIRROR_ASSEMBLIES_RUNTIME_OK"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="physx_cpu",
        choices=("physx_cpu", "newton_cpu", "newton_cuda"),
    )
    parser.add_argument("--hand", default="o6", choices=("l6", "o6"))
    parser.add_argument(
        "--wrists", default="both", choices=("none", "left", "right", "both")
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--depth-hz", type=int, choices=(30, 60), default=30)
    parser.add_argument("--top-hz", type=int, choices=(30, 60), default=60)
    parser.add_argument("--development-cameras", action="store_true")
    parser.add_argument("--task-flow", action="store_true")
    return parser.parse_args()


def assembly_config(args):
    config = load_mirror_config("camera_workstation")
    backend = load_mirror_config(args.profile)
    robots = []
    selected = {"left", "right"} if args.wrists == "both" else {args.wrists}
    for instance, side in zip(config.scene.robots, ("left", "right"), strict=True):
        camera = "_gemini335l" if side in selected else ""
        name = f"ar5_08_{args.hand}{camera}_{side[0]}"
        path = REPO_ROOT / "configs" / "robots" / f"{name}.yaml"
        profile = RobotProfileSettings.from_mapping(
            load_yaml_mapping(path), source=str(path)
        )
        robots.append(replace(instance, robot_profile=name, resolved_profile=profile))
    cameras = tuple(
        c
        for c in config.scene.cameras
        if c.camera_id in {"top", "side"} or c.camera_id.split("_")[0] in selected
    )
    adjusted = []
    for camera in cameras:
        if args.development_cameras and camera.camera_id.endswith("_depth"):
            continue
        preset = camera.camera_profile
        if args.development_cameras:
            preset = "development_1080p_60"
        elif camera.camera_id.endswith("_depth"):
            preset = f"gemini335l_depth_{args.depth_hz}"
        elif camera.camera_id == "top":
            preset = "zed2i_1080p_30" if args.top_hz == 30 else "zed2i_720p_60"
        payload = asdict(camera)
        payload["id"] = payload.pop("camera_id")
        payload = {key: value for key, value in payload.items() if value is not None}
        payload.update(
            load_yaml_mapping(REPO_ROOT / "configs" / "cameras" / f"{preset}.yaml")[
                "camera"
            ]
        )
        payload["camera_profile"] = preset
        adjusted.append(
            CameraSettings.from_mapping(payload, label=f"camera {camera.camera_id}")
        )
    cameras = tuple(adjusted)
    return replace(
        config,
        physics=backend.physics,
        controller_bundles=backend.controller_bundles,
        scene=replace(config.scene, robots=tuple(robots), cameras=cameras),
        control=replace(config.control, sync_simulation_to_wall_clock=False),
        outputs=replace(
            config.outputs,
            render=replace(config.outputs.render, gui=args.gui),
            telemetry=replace(config.outputs.telemetry, enabled=False),
            camera=replace(
                config.outputs.camera,
                enabled=args.record_root is not None,
                save_root=str(args.record_root.resolve()) if args.record_root else None,
                foxglove_live_port=None,
                foxglove_mcap_path=None,
                existing_data_policy="error",
                shutdown_timeout_s=60.0,
            ),
        ),
    )


def exercise_task_flow(runtime):
    """Use public state, image, FK/IK and collision-aware motion boundaries."""
    from linkerbot_sim.planning.requests import IKRequest

    baseline = runtime.get_state()
    first = runtime.render()
    moved = deepcopy(baseline)
    block = moved["objects"]["Tblock"]
    block["positions_local"][1] += 0.08
    for key in ("generalized_signature", "generalized_q_names", "generalized_qd_names"):
        block[key] = []
    for key in ("generalized_q", "generalized_qd", "generalized_world_origin"):
        block[key] = None
    moved["metadata"]["info"].pop(
        "linkerbot.snapshot.newton_solver_integration_state", None
    )
    runtime.set_state(moved)
    second = runtime.render()
    changed = int(np.count_nonzero(first["side"]["depth"] != second["side"]["depth"]))
    assert changed > 10, changed
    runtime.restore_snapshot(baseline)
    resources = runtime.scene_resources
    robot = resources.robots_by_id[0]
    planning = resources.planning_registry
    with planning.lease(0) as context:
        planning.sync_before_plan(0, resources.collision_registry.snapshot())
        names = context.joint_names()
        q = np.asarray(robot.articulation.get_joint_positions())[
            [list(robot.articulation.dof_names).index(name) for name in names]
        ]
        pose = context.make_forward_kinematics().compute_pose(
            q, context.default_tcp_frame
        )
        ik = context.make_inverse_kinematics().solve(
            IKRequest(
                target_position=pose.position,
                target_orientation=pose.orientation,
                warm_start_ik_cspace_seed=q,
                avoid_collisions=False,
            )
        )
        assert ik.success, ik
        assert ik.position_error < 0.005
    goal = q.copy()
    goal[-1] += 0.03
    before = runtime.physics_runtime.simulation_time
    previous_contacts = resources.collision_registry.snapshot().allowed_contacts
    # Exercise a phase-scoped public declaration without disabling other links.
    with runtime.planning_contact_scope(0, (("hand_lh_index_distal", "Tblock"),)):
        runtime.motion.execute(
            "motion.plan_cspace_goal",
            {
                "robot_id": 0,
                "joint_positions": goal.tolist(),
                "duration_s": 0.25,
                "avoid_collisions": True,
            },
            request_id="assembly-task-flow",
            should_cancel=lambda: False,
            protocol="linkerbot.mirror.v2",
        )
    assert resources.collision_registry.snapshot().allowed_contacts == previous_contacts
    actual = np.asarray(robot.articulation.get_joint_positions())[
        [list(robot.articulation.dof_names).index(name) for name in names]
    ]
    assert np.max(np.abs(actual - goal)) < 0.02, (actual, goal)
    assert runtime.physics_runtime.simulation_time > before
    return {
        "set_block_changed_depth_pixels": changed,
        "ik_position_error_m": ik.position_error,
        "execution_max_joint_error_rad": float(np.max(np.abs(actual - goal))),
        "phase_contacts_restored": True,
    }


def run(args):
    runtime = create_mirror_runtime(assembly_config(args))
    try:
        resources = runtime.scene_resources
        cameras = resources.sensor_cameras
        flow = exercise_task_flow(runtime) if args.task_flow else None

        def capture():
            before = runtime.physics_runtime.simulation_time
            frames = runtime.render()
            assert runtime.physics_runtime.simulation_time == before
            poses = {}
            for camera in cameras:
                for modality in camera.settings.modalities:
                    data = np.asarray(frames[camera.name][modality])
                    assert data.shape[:2] == camera.settings.resolution[::-1], (
                        camera.name,
                        data.shape,
                    )
                    assert np.isfinite(data).any(), camera.name
                p, q = camera.get_world_pose()
                poses[camera.name] = [np.asarray(p).tolist(), np.asarray(q).tolist()]
            return frames, poses

        baseline = runtime.get_state()
        first, initial_poses = capture()
        report = {
            "profile": args.profile,
            "hand": args.hand,
            "wrists": args.wrists,
            "gui": args.gui,
            "task_flow": flow,
            "initial_poses": initial_poses,
            "mounts": {c.name: c.get_capture_metadata() for c in cameras},
        }
        moved = {}
        for robot in resources.robots_by_id.values():
            articulation = robot.execution.articulation
            q = np.asarray(articulation.get_joint_positions()).copy()
            index = list(articulation.dof_names).index(
                robot.joint_groups.names("arm")[-1]
            )
            q[index] += 0.15
            articulation.set_joint_positions(q)
        second, moved_poses = capture()
        for camera in cameras:
            if camera.name.startswith(("left_", "right_")):
                assert not np.allclose(
                    initial_poses[camera.name][1],
                    moved_poses[camera.name][1],
                    atol=1e-4,
                ), camera.name
                moved[camera.name] = moved_poses[camera.name]
        runtime.restore_snapshot(baseline)
        restored, restored_poses = capture()
        for name in initial_poses:
            assert np.allclose(
                initial_poses[name][0], restored_poses[name][0], atol=1e-5, rtol=0
            ), (
                name,
                initial_poses[name],
                restored_poses[name],
                baseline["objects"].get("top_camera"),
            )
            initial_q = np.asarray(initial_poses[name][1])
            restored_q = np.asarray(restored_poses[name][1])
            assert any(
                np.allclose(initial_q, sign * restored_q, atol=1e-6, rtol=0)
                for sign in (1, -1)
            ), (name, initial_q, restored_q)
        report["followed_poses"] = moved
        report["restore_depth_error"] = {
            name: int(
                np.count_nonzero(
                    ~np.isclose(
                        first[name]["depth"], restored[name]["depth"], atol=1e-4, rtol=0
                    )
                )
            )
            for name in first
            if "depth" in first[name]
        }
        # Let real contacts settle; compare against the initial named joint state.
        q0 = {
            r.label: np.asarray(r.execution.articulation.get_joint_positions()).copy()
            for r in resources.robots_by_id.values()
        }
        for decision in range(12):
            # A preceding timeline can end on either sampling parity. Render
            # every tick here; the observer keeps each camera's configured rate.
            runtime.step(render=args.task_flow or decision % 2 == 1)
        report["joint_max_delta_after_12_steps"] = {
            r.label: float(
                np.max(
                    np.abs(
                        np.asarray(r.execution.articulation.get_joint_positions())
                        - q0[r.label]
                    )
                )
            )
            for r in resources.robots_by_id.values()
        }
        assert max(report["joint_max_delta_after_12_steps"].values()) < 0.01, report[
            "joint_max_delta_after_12_steps"
        ]
        # Exact restored camera poses can still move a few edge rays after float
        # transform publication. Unchanged native captures also show this noise.
        # Keep a tight pixel budget (10 ppm, floor 10), independent of pose checks.
        report["restore_depth_pixel_budget"] = {
            name: max(10, int(np.asarray(first[name]["depth"]).size * 1e-5))
            for name in report["restore_depth_error"]
        }
        assert all(
            count <= report["restore_depth_pixel_budget"][name]
            for name, count in report["restore_depth_error"].items()
        ), report["restore_depth_error"]
        report["joint_deltas"] = {
            r.label: {
                str(n): float(v)
                for n, v in zip(
                    r.execution.articulation.dof_names,
                    np.asarray(r.execution.articulation.get_joint_positions())
                    - q0[r.label],
                    strict=True,
                )
                if abs(v) > 1e-4
            }
            for r in resources.robots_by_id.values()
        }
        if args.profile == "newton_cpu":
            solver = runtime.physics_runtime.solver
            model, data = solver.mj_model, solver.mj_data
            report["contacts"] = [
                {
                    "body1": model.body(int(model.geom_bodyid[c.geom1])).name,
                    "body2": model.body(int(model.geom_bodyid[c.geom2])).name,
                    "distance": float(c.dist),
                }
                for c in data.contact[: data.ncon]
            ]
        report["gravity_disabled"] = {}
        from pxr import PhysxSchema, Usd, UsdPhysics

        for robot in resources.robots_by_id.values():
            root = resources.session.stage.GetPrimAtPath(
                robot.scene_instance.effective_prim_path
            )
            bodies = [
                p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)
            ]
            flags = {
                str(p.GetPath()): (
                    p.GetAttribute("mjc:gravcomp").Get() == 1.0
                    if args.profile.startswith("newton")
                    else PhysxSchema.PhysxRigidBodyAPI(p).GetDisableGravityAttr().Get()
                )
                for p in bodies
            }
            assert flags and all(flags.values()), flags
            report["gravity_disabled"][robot.label] = len(flags)
        runtime.reset(hold_after_reset=False)
        capture()
        report["reset_capture"] = True
    except BaseException:
        traceback.print_exc()
        runtime.close()
        raise
    runtime.close(
        before_session_close=lambda _: print(
            MARKER + " " + json.dumps(report), flush=True
        )
    )


def main():
    args = parse_args()
    if not in_runtime_worker():
        if args.task_flow and args.record_root is None:
            with TemporaryDirectory(prefix="linkerbot-assembly-record-") as directory:
                args.record_root = Path(directory)
                return supervise(args, [*sys.argv[1:], "--record-root", directory])
        return supervise(args, sys.argv[1:])
    run(args)
    return 0


def supervise(args, argv):
    result = run_supervised_worker(
        script_path=Path(__file__),
        argv=argv,
        required_markers=(MARKER,),
        success_marker="LINKERBOT_MIRROR_ASSEMBLIES_SMOKE_OK",
    )
    if result == 0 and args.record_root is not None:
        verify_recording(args)
    return result


def verify_recording(args):
    """Read drained payloads after native shutdown, including actual sample times."""
    from PIL import Image

    report = {}
    for camera in assembly_config(args).scene.cameras:
        directory = args.record_root / camera.camera_id
        entries = [
            json.loads(line)
            for line in (directory / "metadata.jsonl").read_text().splitlines()
        ]
        for modality in camera.modalities:
            frames = [entry for entry in entries if entry["modality"] == modality]
            assert len(frames) >= 3, (camera.camera_id, modality, len(frames))
            times = [frame["time_s"] for frame in frames]
            assert np.allclose(
                np.diff(times), 1 / camera.frequency_hz, rtol=0, atol=2e-7
            ), times
            ids = [frame["capture"]["native_frame_id"] for frame in frames]
            assert all(b > a for a, b in zip(ids, ids[1:]))
            for frame in frames:
                path = directory / frame["relative_path"]
                if modality == "rgb":
                    with Image.open(path) as picture:
                        assert picture.size == tuple(camera.resolution)
                        picture.load()
                else:
                    with np.load(path) as data:
                        assert data["data"].shape[:2] == tuple(camera.resolution[::-1])
                assert abs(frame["capture"]["physics_time_s"] - frame["time_s"]) < 1e-9
            report[f"{camera.camera_id}/{modality}"] = {
                "frames": len(frames),
                "frequency_hz": camera.frequency_hz,
                "resolution": camera.resolution,
            }
    print("LINKERBOT_MIRROR_ASSEMBLY_RECORDING_OK " + json.dumps(report), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

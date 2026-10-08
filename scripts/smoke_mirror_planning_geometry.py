#!/usr/bin/env python3
"""Exercise mounted geometry, scoped contacts and payloads with real cuRobo models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.runtime_worker_supervisor import in_runtime_worker, run_supervised_worker  # noqa: E402
from scripts.smoke_mirror_assemblies import assembly_config  # noqa: E402
from linkerbot_sim.mirror import create_mirror_runtime  # noqa: E402
from linkerbot_sim.planning.collision_objects import CollisionObject  # noqa: E402
from linkerbot_sim.planning.requests import MotionRequest  # noqa: E402
from linkerbot_sim.planning.results import MotionResult  # noqa: E402
from linkerbot_sim.utils.math_utils import make_rpy_transform  # noqa: E402
from linkerbot_sim.mirror.collision.urdf_kinematics import _UrdfKinematics  # noqa: E402

MARKER = "LINKERBOT_MIRROR_PLANNING_GEOMETRY_RUNTIME_OK"


def run(args):
    config = assembly_config(
        SimpleNamespace(
            hand=args.hand,
            wrists="both",
            profile=args.profile,
            gui=False,
            record_root=None,
            depth_hz=30,
            top_hz=60,
            development_cameras=False,
        )
    )
    runtime = create_mirror_runtime(config)
    try:
        resources = runtime.scene_resources
        registry = resources.collision_registry
        planning = resources.planning_registry
        robot = resources.robots_by_id[0]
        baseline = runtime.capture_snapshot()
        clock = runtime.physics_runtime.simulation_time
        snapshot = registry.snapshot()
        root = make_rpy_transform(
            robot.scene_instance.root_pose.xyz, robot.scene_instance.root_pose.rpy
        )
        report = {
            "hand": args.hand,
            "profile": args.profile,
            "initial_mounted_spheres": len(snapshot.mounted_models[0].spheres),
        }
        with planning.lease(0) as context:

            def sync():
                planning.sync_before_plan(0, registry.snapshot())

            sync()
            q = np.asarray(robot.articulation.get_joint_positions())[
                [
                    list(robot.articulation.dof_names).index(name)
                    for name in context.joint_names()
                ]
            ]
            goal = q.copy()
            goal[-1] += 0.03

            def plan():
                return context.make_motion_planner().plan(
                    MotionRequest(
                        current_q=q,
                        goal_q=goal,
                        avoid_collisions=True,
                        duration_s=0.25,
                        sample_dt_s=1 / 120,
                    )
                )

            result = plan()
            assert result.success, result.diagnostics
            report["free_plan_samples"] = result.diagnostics.coverage["path"]["samples"]
            state = context.kinematics.compute_kinematics(
                context.joint_state_from_positions(q[None])
            )
            spheres = state.robot_spheres.detach().cpu().numpy().reshape(-1, 4)
            links = np.asarray(context._sphere_link_names())
            arm = spheres[np.char.startswith(links, "arm_")]
            for label, selected in (
                ("camera", links == "camera_camera_link"),
                ("hand", np.char.startswith(links, "hand_")),
            ):
                candidates = spheres[selected]
                clearance = np.min(
                    np.linalg.norm(candidates[:, None, :3] - arm[None, :, :3], axis=-1)
                    - arm[None, :, 3],
                    axis=1,
                )
                index = int(np.argmax(clearance))
                assert clearance[index] > 0.01
                pose = np.eye(4)
                pose[:3, 3] = root[:3, :3] @ candidates[index, :3] + root[:3, 3]
                name = f"probe_{label}"
                registry.register_provider(
                    name,
                    lambda pose=pose, name=name: (
                        CollisionObject(name, "cuboid", pose, (0.006, 0.006, 0.006)),
                    ),
                    source="object",
                )
                sync()
                assert not plan().success, label
                pose[:3, 3] += 3
                registry.mark_dirty()
                sync()
                assert plan().success, f"{label}: moved obstacle remained stale"
                registry.unregister_provider(name)
                report[f"{label}_only_obstacle_rejected"] = True

            before = registry.snapshot().mounted_models[0].fingerprint
            changed = np.asarray(robot.articulation.get_joint_positions()).copy()
            hand_index = list(robot.articulation.dof_names).index(
                robot.joint_groups.hand[0]
            )
            changed[hand_index] += 0.2
            robot.articulation.set_joint_positions(changed)
            registry.mark_dirty()
            assert registry.snapshot().mounted_models[0].fingerprint != before
            sync()
            assert context._motion_planner is None
            runtime.restore_snapshot(baseline)
            assert registry.snapshot().mounted_models[0].fingerprint == before
            sync()
            report["hand_shape_refresh"] = True

            urdf_fk = _UrdfKinematics(context.config.robot.urdf_path)

            def flange(qvalue):
                links = urdf_fk.link_transforms(
                    dict(zip(context.joint_names(), qvalue, strict=True))
                )
                return root @ links[context.config.robot.flange_frame]

            local = np.array((0.24, 0.0, 0.35, 1.0))
            payload_pose = np.eye(4)
            payload_pose[:3, 3] = (flange(q) @ local)[:3]
            registry.register_provider(
                "payload_probe",
                lambda: (
                    CollisionObject("payload_probe", "sphere", payload_pose, (0.012,)),
                ),
                source="object",
            )
            runtime.attach_planning_object("payload_probe", 0)
            carried = runtime.capture_snapshot()
            end = q.copy()
            end[-1] += 0.8
            middle = (q + end) / 2
            blocker_pose = np.eye(4)
            blocker_pose[:3, 3] = (flange(middle) @ local)[:3]
            registry.register_provider(
                "payload_blocker",
                lambda: (
                    CollisionObject(
                        "payload_blocker", "cuboid", blocker_pose, (0.012, 0.012, 0.012)
                    ),
                ),
                source="object",
            )
            sync()

            def validate(path):
                return context.validate_motion_collision(
                    MotionResult(
                        path=np.asarray(path),
                        trajectory=None,
                        success=True,
                        status="SUCCESS",
                    )
                )

            assert validate([q]).success
            assert validate([end]).success
            blocked = validate([q, end])
            assert not blocked.success
            assert (
                blocked.diagnostics.coverage["path"]["link_name"]
                == "payload:payload_probe"
            ), blocked.diagnostics
            report["carried_midpath_rejected"] = blocked.diagnostics.coverage["path"]
            runtime.detach_planning_object("payload_probe")
            assert any(
                obj.name == "payload_probe"
                for obj in registry.snapshot().collision_objects_for(0)
            )
            runtime.restore_snapshot(carried)
            assert len(registry.capture_attachments()) == 1
            assert runtime.physics_runtime.simulation_time == clock
            runtime.reset(hold_after_reset=False)
            assert registry.capture_attachments() == []
            report["attachment_restore_release_reset"] = True
            registry.unregister_provider("payload_probe")
            registry.unregister_provider("payload_blocker")
            runtime.restore_snapshot(baseline)
        # All geometry/IK/plan queries are observational; reset itself is separate.
        report["physics_time_before_reset_s"] = clock
    except BaseException:
        runtime.close()
        raise
    runtime.close(
        before_session_close=lambda _: print(
            MARKER + " " + json.dumps(report), flush=True
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("l6", "o6"), default="l6")
    parser.add_argument(
        "--profile",
        choices=("physx_cpu", "newton_cpu", "newton_cuda"),
        default="physx_cpu",
    )
    args = parser.parse_args()
    if not in_runtime_worker():
        return run_supervised_worker(
            script_path=Path(__file__),
            argv=sys.argv[1:],
            required_markers=(MARKER,),
            success_marker="LINKERBOT_MIRROR_PLANNING_GEOMETRY_SMOKE_OK",
        )
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

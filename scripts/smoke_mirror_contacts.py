#!/usr/bin/env python3
"""Check real O6 thumb/camera and object contacts without changing physical filters."""

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import sys
import json
import traceback
from copy import deepcopy
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.smoke_mirror_assemblies import assembly_config
from scripts.runtime_worker_supervisor import in_runtime_worker, run_supervised_worker
from linkerbot_sim.mirror import create_mirror_runtime

MARKER = "LINKERBOT_MIRROR_CONTACTS_RUNTIME_OK"


def main():
    if not in_runtime_worker():
        return run_supervised_worker(
            script_path=Path(__file__),
            argv=sys.argv[1:],
            required_markers=(MARKER,),
            success_marker="LINKERBOT_MIRROR_CONTACTS_SMOKE_OK",
        )
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("physx_cpu", "newton_cpu", "newton_cuda"),
        default="physx_cpu",
    )
    profile = parser.parse_args().profile
    cfg = assembly_config(
        SimpleNamespace(
            hand="o6",
            wrists="both",
            profile=profile,
            gui=False,
            record_root=None,
            depth_hz=30,
            top_hz=60,
            development_cameras=False,
        )
    )
    cfg = replace(cfg, scene=replace(cfg.scene, cameras=()))
    if profile == "physx_cpu":
        import linkerbot_sim.mirror.scene_assembly as assembly

        original_camera_creator = assembly.create_sensor_camera_runtimes

        def prepare_reports(**kwargs):
            from pxr import UsdPhysics, PhysxSchema

            count = 0
            for prim in kwargs["stage"].Traverse():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0)
                    count += 1
            assert count > 0, "no physical bodies received contact reporting"
            return original_camera_creator(**kwargs)

        assembly.create_sensor_camera_runtimes = prepare_reports
    try:
        runtime = create_mirror_runtime(cfg)
    finally:
        if profile == "physx_cpu":
            assembly.create_sensor_camera_runtimes = original_camera_creator
    try:
        resources = runtime.scene_resources
        robots = resources.robots_by_id
        original = {
            i: np.asarray(r.articulation.get_joint_positions()).copy()
            for i, r in robots.items()
        }
        equalities = {}
        for i, r in robots.items():
            tree = ET.parse(ROOT / r.profile_config.asset_path).getroot()
            equalities[i] = [
                (
                    j.attrib["joint1"],
                    j.attrib["joint2"],
                    np.fromstring(j.attrib["polycoef"], sep=" "),
                )
                for j in tree.findall("equality/joint")
            ]
        seen = []
        subscription = None
        if profile == "physx_cpu":
            from pxr import PhysicsSchemaTools
            import omni.physx

            def callback(headers, data):
                for h in headers:
                    pair = [
                        str(PhysicsSchemaTools.intToSdfPath(x))
                        for x in (h.collider0, h.collider1)
                    ]
                    for c in data[
                        h.contact_data_offset : h.contact_data_offset
                        + h.num_contact_data
                    ]:
                        if c.separation < -1e-5:
                            seen.append((pair, float(c.separation)))

            subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(
                callback
            )

        def contacts():
            seen.clear()
            runtime.physics_runtime.step(render=False)
            if profile == "physx_cpu":
                return list(seen)
            solver = runtime.physics_runtime.solver
            model = solver.mj_model
            if profile == "newton_cpu":
                data = solver.mj_data
                return [
                    (
                        [model.body(int(model.geom_bodyid[g])).name for g in c.geom],
                        float(c.dist),
                    )
                    for c in data.contact[: data.ncon]
                    if c.dist < -1e-5
                ]
            runtime.physics_runtime._synchronize_owner_stream()
            data = solver.mjw_data
            n = int(data.nacon.numpy()[0])
            geom = data.contact.geom.numpy()[:n]
            dist = data.contact.dist.numpy()[:n]
            return [
                ([model.body(int(model.geom_bodyid[g])).name for g in pair], float(d))
                for pair, d in zip(geom, dist)
                if d < -1e-5
            ]

        def pose(yaw, pitch):
            for i, r in robots.items():
                names = list(r.articulation.dof_names)
                side = "l" if i == 0 else "r"
                q = original[i].copy()
                q[names.index(f"hand_{side}h_thumb_cmc_yaw")] = yaw[i]
                q[names.index(f"hand_{side}h_thumb_cmc_pitch")] = pitch[i]
                for follower, master, coef in equalities[i]:
                    q[names.index(follower)] = np.polynomial.polynomial.polyval(
                        q[names.index(master)], coef
                    )
                r.articulation.set_joint_positions(q)
                r.articulation.set_joint_velocities(np.zeros_like(q))

        def wrist_pairs(items):
            return [
                (names, d)
                for names, d in items
                if any("hand_" in n for n in names)
                and any("camera_camera_link" in n for n in names)
            ]

        pose((0, 0), (0, 0))
        initial = contacts()
        assert not wrist_pairs(initial), initial
        pose((0.3, 0.3), (0, 0))
        raw = contacts()
        dangerous = wrist_pairs(raw)
        assert dangerous, raw
        for side in ("left_arm", "right_arm"):
            assert any(any(side in name for name in pair) for pair, _ in dangerous), (
                side
            )
        safe = []
        for t in np.linspace(0, 1, 16):
            pose((0, 0), (0.5 * t, 0.57 * t))
            safe += wrist_pairs(contacts())
        for t in np.linspace(0, 1, 25):
            pose((0.26 * t, 1.2 * t), (0.5, 0.57))
            safe += wrist_pairs(contacts())
        assert not safe
        # Move a real dynamic object into a camera/hand collider, then release it.
        pose((0, 0), (0, 0))
        base = runtime.get_state()
        fixture_contacts = {}
        provider = resources.collision_registry._mounted_provider(0)
        for kind, token in [
            ("camera", "camera_camera_link/"),
            ("hand", "hand_lh_hand_base_link/hand_hand_base_col/"),
        ]:
            obj = next(c for c in provider.collision_objects() if token in c.name)
            moved = deepcopy(base)
            block = moved["objects"]["Tblock"]
            block["positions_local"] = obj.pose[:3, 3].tolist()
            for key in (
                "generalized_signature",
                "generalized_q_names",
                "generalized_qd_names",
            ):
                block[key] = []
            for key in ("generalized_q", "generalized_qd", "generalized_world_origin"):
                block[key] = None
            moved["metadata"]["info"].pop(
                "linkerbot.snapshot.newton_solver_integration_state", None
            )
            runtime.set_state(moved)
            result = contacts()
            selected = [
                (names, d)
                for names, d in result
                if any("Tblock" in n or "TBlock" in n for n in names)
                and any(
                    ("camera_camera_link" if kind == "camera" else "hand_") in n
                    for n in names
                )
            ]
            assert selected, (kind, result)
            fixture_contacts[kind] = selected
            runtime.restore_snapshot(base)
        # Keep subscription alive until the simulation is done.
        report = {
            "profile": profile,
            "dangerous_contacts": dangerous,
            "safe_samples": 41,
            "initial_contacts": initial,
            "fixture_contacts": fixture_contacts,
        }
        del subscription
    except BaseException:
        traceback.print_exc()
        runtime.close()
        raise
    runtime.close(
        before_session_close=lambda _: print(
            MARKER + " " + json.dumps(report), flush=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

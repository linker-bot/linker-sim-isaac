"""Do Isaac Lab's body and joint velocity readouts agree with how poses actually change?

Run as a script (Isaac Lab needs a fresh process); prints one JSON line of errors.

Scene, 480 Hz, 2 s, 4 environments:
- a free box with its centre of mass off the body origin, thrown spinning onto the
  ground so it lands, bounces and tumbles;
- a fixed-base two-link pendulum, swinging.

MuJoCo-Warp's implicit integrator advances positions with the end-of-step velocity, so
the readout after a step must equal the backward difference of poses over that step.
This holds exactly for joint coordinates and free-body motion. For a point on a
rotating link, the backward difference is a chord, so it differs from the readout by
O(dt); that error is reported separately and must halve when dt halves.
"""

import json
import os
import sys
import tempfile

import linker_lab

linker_lab.bootstrap()


def author(directory):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    def stage_at(name):
        stage = Usd.Stage.CreateNew(os.path.join(directory, name))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        return stage

    def body(stage, path, pos, mass, com, inertia):
        x = UsdGeom.Xform.Define(stage, path)
        x.AddTranslateOp().Set(Gf.Vec3d(*pos))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        m = UsdPhysics.MassAPI.Apply(x.GetPrim())
        m.CreateMassAttr(mass)
        m.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        return x

    stage = stage_at("box.usda")
    box = body(stage, "/box", (0, 0, 0), 0.2, (0.03, -0.02, 0.01), (2e-4, 3e-4, 4e-4))
    stage.SetDefaultPrim(box.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/box/geom")
    cube.CreateSizeAttr(0.08)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.Save()

    stage = stage_at("pendulum.usda")
    root = UsdGeom.Xform.Define(stage, "/pendulum")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    z = 0.45
    base = body(stage, "/pendulum/base", (0, 0, z), 1.0, (0, 0, 0), (0.01,) * 3)
    link1 = body(stage, "/pendulum/link1", (0, 0, z), 1.0, (0.2, 0, 0), (0.01,) * 3)
    link2 = body(stage, "/pendulum/link2", (0.4, 0, z), 1.0, (0.15, 0.05, 0), (0.01,) * 3)
    tip = UsdGeom.Sphere.Define(stage, "/pendulum/link2/tip")
    tip.CreateRadiusAttr(0.04)
    tip.AddTranslateOp().Set(Gf.Vec3d(0.3, 0, 0))
    UsdPhysics.CollisionAPI.Apply(tip.GetPrim())
    UsdPhysics.FixedJoint.Define(stage, "/pendulum/fix").CreateBody1Rel().SetTargets([base.GetPath()])
    for name, parent, child, anchor in (("j1", base, link1, (0, 0, 0)), ("j2", link1, link2, (0.4, 0, 0))):
        j = UsdPhysics.RevoluteJoint.Define(stage, f"/pendulum/{name}")
        j.CreateAxisAttr("Y")
        j.CreateBody0Rel().SetTargets([parent.GetPath()])
        j.CreateBody1Rel().SetTargets([child.GetPath()])
        j.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    stage.Save()


def main(hz: float) -> dict:
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.app import launch_simulation
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass, instantiate
    from isaaclab.utils.math import quat_box_minus
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    directory = tempfile.mkdtemp()
    author(directory)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(directory, "box.usda")),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.3)),
        )
        pendulum = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Pendulum",
            spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(directory, "pendulum.usda")),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={"j1": 0.3, "j2": -0.5}),
            actuators={},
        )

    dt = 1 / hz
    sim_cfg = sim_utils.SimulationCfg(
        dt=dt, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(integrator="implicitfast"))
    )
    with launch_simulation(sim_cfg, {"headless": True}):
        sim = SimulationContext(sim_cfg)
        scene = instantiate(SceneCfg(num_envs=4, env_spacing=3.0))
        sim.reset()
        box, pendulum = scene["box"], scene["pendulum"]
        pose = box.data.default_root_pose.torch.clone()
        pose[:, :3] += scene.env_origins
        box.write_root_pose_to_sim_index(root_pose=pose)
        throw = torch.tensor([0.3, -0.2, 2.0, 4.0, -2.0, 6.0], device=sim.device)
        box.write_root_velocity_to_sim_index(root_velocity=throw.repeat(scene.num_envs, 1))
        swing = torch.tensor([2.0, -3.0], device=sim.device)
        pendulum.write_joint_velocity_to_sim_index(velocity=swing.repeat(scene.num_envs, 1))
        scene.reset()

        keys = ("pos", "quat", "lin", "ang", "q", "qd")
        rec = {"box": {k: [] for k in keys[:4]}, "pendulum": {k: [] for k in keys}}
        for _ in range(round(2.0 * hz)):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            for name, asset in (("box", box), ("pendulum", pendulum)):
                d = asset.data
                r = rec[name]
                r["pos"].append(d.body_link_pos_w.torch.clone())
                r["quat"].append(d.body_link_quat_w.torch.clone())
                r["lin"].append(d.body_link_lin_vel_w.torch.clone())
                r["ang"].append(d.body_link_ang_vel_w.torch.clone())
                if name == "pendulum":
                    r["q"].append(d.joint_pos.torch.clone())
                    r["qd"].append(d.joint_vel.torch.clone())

        def err(readout, fd):
            return float((readout - fd).norm(dim=-1).max()) if fd.dim() > 2 else float((readout - fd).abs().max())

        out = {"hz": hz}
        for name, r in rec.items():
            s = {k: torch.stack(v) for k, v in r.items() if v}
            fd_lin = (s["pos"][1:] - s["pos"][:-1]) / dt
            q0, q1 = s["quat"][:-1].reshape(-1, 4), s["quat"][1:].reshape(-1, 4)
            fd_ang = (quat_box_minus(q1, q0) / dt).reshape(fd_lin.shape)
            out[name] = {
                "linear": err(s["lin"][1:], fd_lin),
                "angular": err(s["ang"][1:], fd_ang),
                "peak_speed": float(fd_lin.norm(dim=-1).max()),
                "lowest_z": float(s["pos"][..., 2].min()),
            }
            if "q" in s:
                out[name]["joint"] = err(s["qd"][1:], (s["q"][1:] - s["q"][:-1]) / dt)
        return out


if __name__ == "__main__":
    print("RESULT " + json.dumps(main(float(sys.argv[1]) if len(sys.argv) > 1 else 480.0)), flush=True)

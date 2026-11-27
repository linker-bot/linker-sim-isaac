"""Run LW-v0 with the arm and hand actions and report what happened (one JSON line).

    python tests/gpu/arm_hand_check.py reference
    python tests/gpu/arm_hand_check.py limits [num_envs] [seconds] [amplitude]
    python tests/gpu/arm_hand_check.py press
    python tests/gpu/arm_hand_check.py pose
    python tests/gpu/arm_hand_check.py hold

reference  Replays ``tests/data/drive_reference.npz`` closed-loop: the left arm gets the
           recorded per-step references (``stream="samples"``), the right arm the recorded
           Hermite points with velocities (``stream="points_velocity"``), both without
           limits; the hands get the recorded shaft targets. Reports the largest deviation
           of the measured joints and of the written targets from the recording, before
           and during (``_pressing``) the last phase, where the hands press toward their
           torque guard.
limits     The default environment (joint goals through the trajectory generator, real
           limits); every half second each arm gets a random goal within ``amplitude``
           (default 0.15 rad) of home, the left wrist always beyond the octagon. Reports
           reference and measured speeds against the limits, the wrist against the
           octagon, limit hits and simulation speed. (With larger goals the fingers reach
           the table; a blocked arm lags its reference, and when it slips free at the
           torque cap, faster than the limits.)
press      The left hand's six driven joints stop at 40 % of their range (joint limits
           written into the simulation); the hand closes onto the stops and presses for
           12 s, then opens; then the thumb yaw alone. Reports the largest drive torque
           against its force limit (the hand's torque guard keeps it 1 N·m below) and
           whether every joint returns to its open target.
pose       Palm-point pose commands: each palm point is commanded 5 cm from where it is
           (forward, sideways, up in the arm base frame); reports the time to come within
           2 mm and the final error.
hold       The default environment, every action value non-finite (NaN, "keep the previous
           command") from the reset on, for 1 s. Reports the largest deviation of the driven
           joints from home.

Run as a separate process: Isaac Lab allows one simulation per process.
"""

import json
import pathlib
import sys
import time

import gymnasium as gym
import numpy as np
import torch

from linker_lab import arm, workstation  # workstation bootstraps the engine

REFERENCE = pathlib.Path(__file__).parents[1] / "data" / "drive_reference.npz"


def make_env(cfg):
    env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
    env.reset()
    return env


def joint_ids(env, side, names):
    ids, _ = env.scene[side].find_joints(list(names), preserve_order=True)
    return ids


def reference() -> dict:
    ref = np.load(REFERENCE)
    cfg = workstation.LWv0EnvCfg()
    cfg.scene.num_envs = ref["left/arm_q"].shape[0]
    cfg.actions.left_arm = workstation.arm_action_cfg("left", interpolation=False, stream="samples", limits=None)
    cfg.actions.right_arm = workstation.arm_action_cfg(
        "right", interpolation=False, stream="points_velocity", limits=None
    )
    from isaaclab.app import launch_simulation

    with launch_simulation(cfg, {"headless": True}):
        env = make_env(cfg)
        dev = env.device
        decisions = ref["left/hand_command"].shape[1]
        press = int(ref["press_from_step"])  # from here the hands press toward their torque guard
        deviation = {f"{p}_{x}{a}": 0.0 for p in ("arm", "hand") for x in ("q", "target") for a in ("", "_pressing")}
        for k in range(decisions):
            q_ref = ref["left/arm_q_ref"][:, 8 * k : 8 * k + 8]
            v_ref = ref["left/arm_v_ref"][:, 8 * k : 8 * k + 8]
            parts = [
                np.concatenate((q_ref.reshape(len(q_ref), -1), v_ref.reshape(len(v_ref), -1)), -1),
                np.concatenate((ref["right/arm_endpoint_q"][:, k], ref["right/arm_endpoint_v"][:, k]), -1),
                ref["left/hand_command"][:, k],
                ref["right/hand_command"][:, k],
            ]
            env.step(torch.tensor(np.concatenate(parts, -1), device=dev, dtype=torch.float32))
            t = 8 * (k + 1)
            for side in workstation.SIDES:
                robot = env.scene[side]
                for part, names in (("arm", workstation.ARM_JOINTS[side]), ("hand", workstation.HAND_JOINTS[side])):
                    ids = joint_ids(env, side, names)
                    if t < ref[f"{side}/{part}_q"].shape[1]:
                        measured = robot.data.joint_pos.torch[:, ids].cpu().numpy()
                        key = f"{part}_q" + ("_pressing" if t > press else "")
                        deviation[key] = max(deviation[key], float(np.abs(measured - ref[f"{side}/{part}_q"][:, t]).max()))
                    target = robot.data.joint_pos_target.torch[:, ids].cpu().numpy()
                    key = f"{part}_target" + ("_pressing" if t - 1 >= press else "")
                    deviation[key] = max(deviation[key], float(np.abs(target - ref[f"{side}/{part}_target"][:, t - 1]).max()))
        env.close()
    return {"max_deviation": deviation}


def limits(num_envs: int, seconds: float, amplitude: float) -> dict:
    cfg = workstation.LWv0EnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = seconds + 10
    from isaaclab.app import launch_simulation

    with launch_simulation(cfg, {"headless": True}):
        env = make_env(cfg)
        dev = env.device
        gen = torch.Generator(device=dev).manual_seed(1)
        arms = [env.action_manager.get_term(f"{side}_arm") for side in workstation.SIDES]
        hands = [env.scene[s].data.default_joint_pos.torch[:, joint_ids(env, s, workstation.HAND_JOINTS[s])]
                 for s in workstation.SIDES]
        ids = [joint_ids(env, s, workstation.ARM_JOINTS[s]) for s in workstation.SIDES]
        home = [env.scene[s].data.default_joint_pos.torch[:, i] for s, i in zip(workstation.SIDES, ids)]
        vmax = torch.tensor(arm.VELOCITY, device=dev)
        speed_ratio = torch.zeros(7, device=dev)
        ref_speed_ratio = torch.zeros(7, device=dev)
        wrist_outside = 0.0
        decisions = round(seconds / env.step_dt)
        start = None
        for k in range(decisions):
            if k % 30 == 0:  # a new goal every half second
                goals = [h + amplitude * (2 * torch.rand(h.shape, device=dev, generator=gen) - 1) for h in home]
                goals[0][:, 5] = 1.2  # J6 beyond the octagon in the left arm ...
                goals[0][:, 6] = 1.5  # ... in its corner region
            if k == 1:
                torch.cuda.synchronize()
                start = time.perf_counter()
            env.step(torch.cat([*goals, *hands], -1))
            for side, i, term in zip(workstation.SIDES, ids, arms):
                qd = env.scene[side].data.joint_vel.torch[:, i]
                speed_ratio = torch.maximum(speed_ratio, (qd.abs() / vmax).amax(0))
                ref_speed_ratio = torch.maximum(ref_speed_ratio, (term.reference[1].abs().float() / vmax).amax(0))
                q = env.scene[side].data.joint_pos.torch[:, i].double()
                wide = torch.full((7,), 10.0, device=dev, dtype=torch.float64)
                p, _ = arm.project(q, -wide, wide)  # nearest point in the octagon
                wrist_outside = max(wrist_outside, float((q - p).norm(dim=-1).max()))
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        q = torch.cat([env.scene[s].data.joint_pos.torch for s in workstation.SIDES])
        hits = env.extras["arm_limit_hits"]
        result = {
            "num_envs": num_envs,
            "env_steps_per_s": num_envs * (decisions - 1) / wall,
            "max_measured_speed_over_limit": speed_ratio.tolist(),
            "max_reference_speed_over_limit": ref_speed_ratio.tolist(),
            "max_wrist_outside_octagon_rad": wrist_outside,
            "limit_hits_mean": {s: float(hits[s].float().mean()) for s in workstation.SIDES},
            "finite": bool(torch.isfinite(q).all()),
        }
        env.close()
    return result


def press() -> dict:
    cfg = workstation.LWv0EnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 120
    from isaaclab.app import launch_simulation

    with launch_simulation(cfg, {"headless": True}):
        env = make_env(cfg)
        dev = env.device
        robot = env.scene["left"]
        hand = joint_ids(env, "left", workstation.HAND_JOINTS["left"])
        cap = robot.data.joint_effort_limits.torch[0, hand]
        limits = robot.data.joint_pos_limits.torch[:, hand].clone()
        limits[..., 1] *= 0.4
        robot.write_joint_position_limit_to_sim(limits, joint_ids=hand)
        arms = [env.scene[s].data.default_joint_pos.torch[:, joint_ids(env, s, workstation.ARM_JOINTS[s])]
                for s in workstation.SIDES]
        right_hand = env.scene["right"].data.default_joint_pos.torch[:, joint_ids(env, "right", workstation.HAND_JOINTS["right"])]
        torque_over_cap = -1e9
        result = {}
        for name, target, seconds in (("close", (0.45, 1.2, 1.45, 1.45, 1.45, 1.45), 12), ("open", (0.0,) * 6, 12),
                                      ("thumb yaw", (0.0, 1.36, 0.0, 0.0, 0.0, 0.0), 12), ("open again", (0.0,) * 6, 12)):
            goal = torch.tensor(target, device=dev)[None]
            for _ in range(round(seconds / env.step_dt)):
                env.step(torch.cat([*arms, goal, right_hand], -1))
                torque = robot.data.applied_torque.torch[0, hand].abs()
                torque_over_cap = max(torque_over_cap, float((torque - cap).max()))
            q = robot.data.joint_pos.torch[0, hand]
            result[name] = {"error": float((q - goal[0]).abs().max()),
                            "speed": float(robot.data.joint_vel.torch[0, hand].abs().max())}
        env.close()
    return {"max_torque_over_cap": torque_over_cap, **result}


def pose() -> dict:
    cfg = workstation.LWv0EnvCfg()
    cfg.scene.num_envs = 3
    for side in workstation.SIDES:
        setattr(cfg.actions, f"{side}_arm", workstation.arm_action_cfg(side, command="pose", frame="palm"))
    from isaaclab.app import launch_simulation
    from isaaclab.utils.math import subtract_frame_transforms

    with launch_simulation(cfg, {"headless": True}):
        env = make_env(cfg)
        hands = [env.scene[s].data.default_joint_pos.torch[:, joint_ids(env, s, workstation.HAND_JOINTS[s])]
                 for s in workstation.SIDES]

        def palm_b(side):
            robot = env.scene[side]
            pos_w, quat_w = workstation.palm_pose_w(robot)
            root = robot.data.root_link_pose_w.torch
            return subtract_frame_transforms(root[:, :3], root[:, 3:7], pos_w, quat_w)

        goals = []
        for side in workstation.SIDES:
            pos, quat = palm_b(side)
            pos = pos.clone()
            pos[torch.arange(3), torch.arange(3)] += 0.05  # env i moves 5 cm along base axis i
            goals.append(torch.cat((pos, quat), -1))
        settle = {}
        error = {}
        for k in range(240):  # 4 s
            env.step(torch.cat([*goals, *hands], -1))
            for side, goal in zip(workstation.SIDES, goals):
                err = (palm_b(side)[0] - goal[:, :3]).norm(dim=-1)
                error[side] = err.tolist()
                if side not in settle and bool((err < 0.002).all()):
                    settle[side] = (k + 1) * env.step_dt
        env.close()
    return {"settle_time_2mm_s": settle, "final_error_m": error}


def hold() -> dict:
    cfg = workstation.LWv0EnvCfg()
    cfg.scene.num_envs = 2
    from isaaclab.app import launch_simulation

    with launch_simulation(cfg, {"headless": True}):
        env = make_env(cfg)
        ids = {s: joint_ids(env, s, [*workstation.ARM_JOINTS[s], *workstation.HAND_JOINTS[s]]) for s in workstation.SIDES}
        home = {s: env.scene[s].data.default_joint_pos.torch[:, ids[s]].clone() for s in workstation.SIDES}
        nan = torch.full((env.num_envs, env.action_manager.total_action_dim), float("nan"), device=env.device)
        drift = 0.0
        for _ in range(60):
            env.step(nan)
            for s in workstation.SIDES:
                q = env.scene[s].data.joint_pos.torch[:, ids[s]]
                drift = max(drift, float((q - home[s]).abs().max()))
        env.close()
    return {"max_drift_from_home_rad": drift}


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "reference":
        result = reference()
    elif mode == "limits":
        result = limits(
            int(args[0]) if args else 400,
            float(args[1]) if len(args) > 1 else 4.0,
            float(args[2]) if len(args) > 2 else 0.15,
        )
    elif mode == "press":
        result = press()
    elif mode == "hold":
        result = hold()
    else:
        result = pose()
    print("RESULT " + json.dumps(result))

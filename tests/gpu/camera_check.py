"""Render every LW-v0 camera once at the home pose and report what it sees (one JSON line).

    python tests/gpu/camera_check.py

Per camera: the share of pixels closer than 3 cm (its own housing, if it is blocked) and
the median depth. Images at a quarter of the real resolution.
"""

import json

import torch

from linker_lab import assets, workstation  # bootstraps the engine


def main() -> dict:
    import gymnasium as gym
    from isaaclab.app import launch_simulation
    from isaaclab.utils import replace

    cfg = workstation.LWv0EnvCfg()
    for name in workstation.CAMERAS:
        width, height = assets.camera(*workstation.CAMERAS[name][:2])["resolution"]
        camera = workstation.camera_cfg(name, width=width // 4, height=height // 4)
        if "depth" not in camera.data_types:  # measure what the RGB cameras see
            camera = replace(camera, data_types=[*camera.data_types, "depth"])
        setattr(cfg.scene, name, camera)
    result = {}
    with launch_simulation(cfg, {"headless": True}):
        env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
        env.reset()
        for _ in range(4):
            env.step(workstation.home_action(env))
        for name in workstation.CAMERAS:
            depth = env.scene[name].data.output["depth"].torch[0, ..., 0]
            hit = depth > 0
            result[name] = {
                "near_3cm": float((hit & (depth < 0.03)).float().mean()),
                "median_depth_m": float(depth[hit].median()) if hit.any() else None,
                "finite": bool(torch.isfinite(depth).all()),
            }
        env.close()
    return result


if __name__ == "__main__":
    print("RESULT " + json.dumps(main()))

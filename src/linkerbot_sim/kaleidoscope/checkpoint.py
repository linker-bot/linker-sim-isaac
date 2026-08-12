"""显式持久化冷边界；训练 step/reset 永远不会调用本模块。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from linkerbot_sim.kaleidoscope.snapshot import KaleidoscopeEpisodeSnapshot

if TYPE_CHECKING:
    import torch


def save_kaleidoscope_checkpoint(
    snapshot: KaleidoscopeEpisodeSnapshot,
    path: str | Path,
) -> Path:
    """把 GPU snapshot 显式下载并保存为无 pickle 的压缩 NPZ。"""

    destination = Path(path).expanduser().resolve()
    if destination.suffix != ".npz":
        raise ValueError("Kaleidoscope checkpoint path must use the .npz suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = tuple(sorted(snapshot.fields))
    payload: dict[str, np.ndarray] = {
        "env_ids": snapshot.env_ids.detach().cpu().numpy(),
        "metadata": np.asarray(
            json.dumps(
                {
                    "schema_version": snapshot.schema_version,
                    "compatibility_fingerprint": snapshot.compatibility_fingerprint,
                    "control_mode": snapshot.control_mode,
                    "control_generation": snapshot.control_generation,
                    "fields": names,
                },
                sort_keys=True,
            )
        ),
    }
    for index, name in enumerate(names):
        payload[f"field_{index}"] = snapshot.fields[name].detach().cpu().numpy()
    np.savez_compressed(destination, **payload)
    return destination


def load_kaleidoscope_checkpoint(
    path: str | Path, *, device: str | "torch.device"
) -> KaleidoscopeEpisodeSnapshot:
    """读取冷 checkpoint，并一次性上传为 GPU-owned snapshot。"""

    import torch

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        schema_version = metadata.get("schema_version")
        if schema_version not in {1, 2}:
            raise ValueError("unsupported Kaleidoscope checkpoint schema")
        names = metadata.get("fields")
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise ValueError("invalid Kaleidoscope checkpoint field metadata")
        compatibility_fingerprint = metadata.get("compatibility_fingerprint")
        if (
            not isinstance(compatibility_fingerprint, str)
            or not compatibility_fingerprint
        ):
            raise ValueError("invalid Kaleidoscope checkpoint fingerprint metadata")
        env_ids = torch.as_tensor(
            archive["env_ids"], device=device, dtype=torch.int64
        ).clone()
        fields = {
            name: torch.as_tensor(archive[f"field_{index}"], device=device).clone()
            for index, name in enumerate(names)
        }
    return KaleidoscopeEpisodeSnapshot(
        env_ids=env_ids,
        fields=fields,
        compatibility_fingerprint=compatibility_fingerprint,
        control_mode=(None if schema_version == 1 else metadata.get("control_mode")),
        control_generation=(
            0 if schema_version == 1 else metadata.get("control_generation")
        ),
        schema_version=schema_version,
    )


__all__ = ["load_kaleidoscope_checkpoint", "save_kaleidoscope_checkpoint"]

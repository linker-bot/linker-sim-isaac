"""skrl 2.1 的全 CUDA SAME_STEP 集成。"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CudaRolloutMemory": ("linkerbot_sim.training.skrl.memory", "CudaRolloutMemory"),
    "FinalObservationPPO": (
        "linkerbot_sim.training.skrl.final_observation_ppo",
        "FinalObservationPPO",
    ),
    "SkrlTorchAdapter": ("linkerbot_sim.training.skrl.env", "SkrlTorchAdapter"),
    "make_skrl_trainer": (
        "linkerbot_sim.training.skrl.factory",
        "make_skrl_trainer",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value

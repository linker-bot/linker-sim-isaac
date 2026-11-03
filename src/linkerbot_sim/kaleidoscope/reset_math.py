"""Shared counter arithmetic and reset ranges; works on NumPy or Torch arrays.

No RNG state, device selection, or physics ownership lives here. Integer arrays
use signed int64 wraparound; only the low 31 bits enter the float32 draw.
"""

from typing import Any


def reset_rng_keys(env_ids: Any, seed: int) -> Any:
    return env_ids * 6364136223846793005 + seed


def reset_rng_bits(key: Any, counter: Any, lane: Any) -> Any:
    state = key + (counter + lane * 747796405) * 2891336453
    return (state * 1103515245 + 12345) & 0x7FFFFFFF


def uniform_range(unit: Any, bounds: tuple[float, float]) -> Any:
    low, high = float(bounds[0]), float(bounds[1])
    if low > high:
        raise ValueError("randomization range minimum must not exceed maximum")
    return low + unit * (high - low)

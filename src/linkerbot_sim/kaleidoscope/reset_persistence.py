"""Explicit JSON cold-storage boundary for Kaleidoscope reset descriptions."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from linkerbot_sim.kaleidoscope.reset_description import (
    KaleidoscopeResetSceneDescription,
    validate_reset_scene_description,
)


def load_reset_scene_description(
    path: str | Path,
) -> KaleidoscopeResetSceneDescription:
    """Load and fully validate one UTF-8 JSON reset description."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid Kaleidoscope reset description JSON: {source}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Kaleidoscope reset description JSON must contain an object")
    return validate_reset_scene_description(payload)


def save_reset_scene_description(
    description: KaleidoscopeResetSceneDescription | Mapping[str, object],
    path: str | Path,
    *,
    replace: bool = False,
) -> Path:
    """Validate and atomically save one reset description as canonical JSON."""

    parsed = validate_reset_scene_description(description)
    destination = Path(path)
    if destination.exists() and not replace:
        raise FileExistsError(
            f"Kaleidoscope reset description already exists: {destination}"
        )
    encoded = (
        json.dumps(
            parsed.as_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and not replace:
            raise FileExistsError(
                f"Kaleidoscope reset description already exists: {destination}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


__all__ = [
    "load_reset_scene_description",
    "save_reset_scene_description",
]

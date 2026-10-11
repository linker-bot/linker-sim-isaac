"""Minimal Isaac PhysX scene-query adapter.

This module keeps Isaac imports behind construction so its callback and
lifecycle contracts can be tested without starting Kit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


EncodedSdfPath = tuple[int, int]
PathEncoder = Callable[[str], Sequence[int]]


@dataclass(frozen=True, slots=True)
class PhysxOverlapHit:
    """Exact PhysX identities reported for one overlapping scene shape."""

    collision_path: str
    rigid_body_path: str
    prototype_index: int


class PhysxSceneQueryAdapter:
    """Own the Python entry point to PhysX ``overlap_shape``.

    The adapter does not own or mutate the USD prim being queried. Callers are
    responsible for creating and positioning any query-only prim before this
    method is invoked.
    """

    def __init__(
        self,
        scene_query_interface: object | None = None,
        *,
        path_encoder: PathEncoder | None = None,
    ) -> None:
        if scene_query_interface is None or path_encoder is None:
            default_interface, default_encoder = self._load_isaac_dependencies()
            if scene_query_interface is None:
                scene_query_interface = default_interface
            if path_encoder is None:
                path_encoder = default_encoder

        overlap_shape = getattr(scene_query_interface, "overlap_shape", None)
        if not callable(overlap_shape):
            raise TypeError("scene_query_interface must expose callable overlap_shape")
        if not callable(path_encoder):
            raise TypeError("path_encoder must be callable")

        self._scene_query_interface: object | None = scene_query_interface
        self._path_encoder: PathEncoder | None = path_encoder
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def overlap_shape(
        self,
        prim_path: str,
        *,
        stop_after_first: bool = False,
    ) -> tuple[PhysxOverlapHit, ...]:
        """Return exact scene identities overlapping one USD GPrim."""

        interface, path_encoder = self._require_open()
        encoded_path = self._encode_prim_path(prim_path, path_encoder)
        hits: list[PhysxOverlapHit] = []

        def report_hit(hit: Any) -> bool:
            hits.append(
                PhysxOverlapHit(
                    collision_path=str(hit.collision),
                    rigid_body_path=str(hit.rigid_body),
                    prototype_index=int(hit.protoIndex),
                )
            )
            return not stop_after_first

        # Keep anyHit=False because native any-hit mode only guarantees a
        # boolean result; the callback identities are required by later exact
        # env/current-self/target filtering.
        interface.overlap_shape(
            encoded_path[0],
            encoded_path[1],
            report_hit,
            False,
        )
        return tuple(hits)

    def close(self) -> None:
        """Idempotently invalidate this Python entry point.

        The global native interface is owned by Isaac/SimulationApp, so this
        method deliberately does not release or shut down that interface.
        """

        self._closed = True
        self._scene_query_interface = None
        self._path_encoder = None

    def _require_open(self) -> tuple[Any, PathEncoder]:
        if self._closed:
            raise RuntimeError("PhysX scene-query adapter is closed")
        if self._scene_query_interface is None or self._path_encoder is None:
            raise RuntimeError("PhysX scene-query adapter is unavailable")
        return self._scene_query_interface, self._path_encoder

    @staticmethod
    def _encode_prim_path(
        prim_path: str,
        path_encoder: PathEncoder,
    ) -> EncodedSdfPath:
        if not isinstance(prim_path, str) or not prim_path.startswith("/"):
            raise ValueError("prim_path must be an absolute USD prim path")
        encoded = tuple(path_encoder(prim_path))
        if len(encoded) != 2 or any(type(value) is not int for value in encoded):
            raise RuntimeError("PhysicsSchemaTools.encodeSdfPath returned invalid data")
        return encoded[0], encoded[1]

    @staticmethod
    def _load_isaac_dependencies() -> tuple[object, PathEncoder]:
        try:
            from omni.physx import get_physx_scene_query_interface
            from pxr import PhysicsSchemaTools, Sdf
        except ImportError as exc:
            raise RuntimeError(
                "PhysX scene queries require an active Isaac Sim runtime"
            ) from exc

        interface = get_physx_scene_query_interface()

        def encode_path(prim_path: str) -> Sequence[int]:
            return PhysicsSchemaTools.encodeSdfPath(Sdf.Path(prim_path))

        return interface, encode_path


__all__ = ["PhysxOverlapHit", "PhysxSceneQueryAdapter"]

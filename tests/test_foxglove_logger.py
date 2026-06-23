from __future__ import annotations

import sys

import numpy as np

from manipulation_project.visualization import foxglove_logger


def test_foxglove_time_helpers() -> None:
    assert foxglove_logger._ns_time(None) == 0
    assert foxglove_logger._ns_time(1.25) == 1_250_000_000


def test_foxglove_optional_dependency_error() -> None:
    original_module = sys.modules.get("foxglove")
    sys.modules["foxglove"] = None
    try:
        foxglove_logger._load_foxglove()
    except ImportError as exc:
        assert "foxglove-sdk" in str(exc)
    else:
        raise AssertionError("expected ImportError when foxglove-sdk is unavailable")
    finally:
        if original_module is None:
            sys.modules.pop("foxglove", None)
        else:
            sys.modules["foxglove"] = original_module


def test_foxglove_vector_shape_validation() -> None:
    class Messages:
        class Vector3:
            def __init__(self, *, x=0.0, y=0.0, z=0.0):
                self.x = x
                self.y = y
                self.z = z

    vec = foxglove_logger._vector3(np.asarray([1.0, 2.0, 3.0]), Messages)
    assert (vec.x, vec.y, vec.z) == (1.0, 2.0, 3.0)

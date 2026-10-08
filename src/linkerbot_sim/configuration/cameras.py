"""Reusable Mirror sensor leaves; mounting remains owned by the scene."""

from .common import require_keys, strict_mapping


def camera_profile_from_mapping(value: object, *, label: str) -> dict[str, object]:
    """Validate leaf ownership; CameraSettings validates the expanded typed values."""

    mapping = strict_mapping(value, label=label)
    require_keys(
        mapping,
        required={"resolution", "frequency_hz", "modalities", "clipping_range_m"},
        optional={"intrinsics", "calibration_source", "sensor_model"},
        label=label,
    )
    return mapping

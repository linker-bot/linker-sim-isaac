from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.planning.batch_ik import BatchIKResult, apply_ik_failure_fallback


def test_batched_ik_result_normalizes_and_fills_status() -> None:
    result = BatchIKResult(
        joint_positions=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        success=np.asarray([True, False]),
        position_error=np.asarray([0.01, 1.0]),
    )

    assert result.status == ("SUCCESS", "FAILED")
    assert result.success.dtype == np.bool_


def test_apply_ik_failure_fallback_keeps_failed_rows() -> None:
    result = BatchIKResult(
        joint_positions=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        success=np.asarray([True, False]),
        position_error=np.asarray([0.0, 1.0]),
    )
    fallback = np.asarray([[10.0, 20.0], [30.0, 40.0]])

    np.testing.assert_allclose(
        apply_ik_failure_fallback(result, fallback),
        [[1.0, 2.0], [30.0, 40.0]],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "joint_positions": np.asarray([1.0, 2.0]),
            "success": np.asarray([True]),
            "position_error": np.asarray([0.0]),
        },
        {
            "joint_positions": np.asarray([[1.0, 2.0]]),
            "success": np.asarray([True, False]),
            "position_error": np.asarray([0.0]),
        },
    ],
)
def test_batched_ik_result_rejects_shape_mismatches(kwargs) -> None:
    with pytest.raises(ValueError):
        BatchIKResult(**kwargs)

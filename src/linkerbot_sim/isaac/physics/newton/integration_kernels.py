"""Fixed TIME/ACT/WARMSTART packing; imported only by initialized Warp stores."""

import warp as wp


@wp.kernel(enable_backward=False)
def _pack_persistent_state(
    time: wp.array(dtype=wp.float32),
    activation: wp.array2d(dtype=wp.float32),
    warmstart: wp.array2d(dtype=wp.float32),
    active: wp.array(dtype=wp.bool),
    output: wp.array2d(dtype=wp.float32),
):
    world, column = wp.tid()
    if active[world]:
        if column == 0:
            output[world, column] = time[world]
        elif column <= activation.shape[1]:
            output[world, column] = activation[world, column - 1]
        else:
            output[world, column] = warmstart[world, column - 1 - activation.shape[1]]


def pack_persistent_state(data, output, active, *, stream=None):
    """Preserve unselected rows and enqueue on the store's owning stream."""
    wp.launch(
        _pack_persistent_state,
        dim=output.shape,
        inputs=[data.time, data.act, data.qacc_warmstart, active, output],
        device=output.device,
        stream=stream,
    )

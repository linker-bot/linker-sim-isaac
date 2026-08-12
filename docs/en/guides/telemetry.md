# Telemetry

Language: [English](telemetry.md) | [中文](../../zh-CN/guides/telemetry.md)

Telemetry belongs to Mirror. It observes a completed reality-mapped world and hands
owned samples to bounded sinks. Kaleidoscope has no telemetry publisher in its
runtime closure; trainers should aggregate CUDA metrics downstream and cross the host
boundary only at an explicit reporting cadence.

## Configuration

```yaml
telemetry:
  enabled: true
  rate_hz: 60.0
  buffer_size: 1
  include_hybrid_control: true
  topics:
    hybrid_control: /linkerbot/mirror/hybrid_control
  shutdown_timeout_s: 2.0
```

- `rate_hz` is nonnegative and must be positive when enabled.
- `buffer_size` is a positive bounded handoff capacity.
- `shutdown_timeout_s` is positive and bounds sink teardown.

When `enabled: true`, configure at least one live port or MCAP path and enable at
least one message modality. When `enabled: false`, valid endpoint, topic, and policy
settings may remain in YAML; they are still schema-checked, but runtime does not
preflight the MCAP path, bind the port, allocate the publisher buffer, or create a
sink. Likewise, `include_efforts: false` may retain a valid `joint_effort_field`, but
runtime projects it to `none`; enabling efforts requires a non-`none` source.

The telemetry cadence is an output fact. It does not change physics or render
frequency.

`include_hybrid_control` creates a separate JSON modality. The Foxglove channel is
lazy: when disabled, it is never created. When enabled but no hybrid motion is active,
the topic publishes exactly `{"active": false}`. During a hybrid motion it publishes
the latest finite owner-thread diagnostic, including request/robot identity, step,
parameter and tare generations, force-axis selection, target and measured pose/wrench,
joint effort, saturation/contact flags, and Jacobian conditioning.

## Sampling Boundary

Sample after a completed physics step, when robot/object state is coherent. The Isaac
owner thread may construct a pure immutable snapshot, but network or file sinks must
not retain articulation, USD, camera, or physics handles.

Hybrid telemetry follows the same rule. The controller replaces one cached owned
diagnostic mapping at each control tick. The sampler deep-copies that cache into
`StateSnapshot`; the publisher thread never calls a PhysX wrench, Jacobian, or
articulation getter.

Telemetry is read-only. A sink failure must never write simulation state or adjust
control targets.

## Bounded Handoff

A slow consumer cannot be allowed to grow memory without limit. The stream capacity
defines the maximum pending sample count. For live visualization, a latest-value
policy is usually preferable; for durable capture, select a policy that makes loss or
backpressure explicit and size the sink for expected throughput.

Status should expose at least buffer depth, dropped/replaced sample count, and sink
lifecycle state. Treat a shutdown timeout as a live resource, not as successful
closure.

## Foxglove

Foxglove publishing is one possible Mirror sink. Keep topic schemas stable and use
the scene snapshot/telemetry DTO rather than serializing engine objects. See
[Foxglove](foxglove.md).

## Reinforcement-Learning Metrics

Kaleidoscope returns dense CUDA `info` tensors from task and action execution. The
skrl adapter preserves them on device. A trainer may reduce values on CUDA, then copy
small aggregates at a chosen logging interval. Copying complete observations or state
for every environment every decision is not telemetry support; it is a performance
regression.

## Troubleshooting

If samples stop:

1. verify Mirror output telemetry is enabled and has a positive rate;
2. check that physics is advancing rather than paused or emergency-stopped;
3. inspect bounded-buffer status and sink errors;
4. confirm the sink did not retain an engine handle across threads;
5. on shutdown, identify the named live worker before retrying close.

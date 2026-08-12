# Foxglove Integration

Language: [English](foxglove.md) | [中文](../../zh-CN/guides/foxglove.md)

Foxglove is a Mirror visualization/output integration. It is not part of the
Kaleidoscope environment or its CUDA training loop.

## Data Flow

```text
Mirror owner thread
  -> immutable state or camera payload
  -> bounded output handoff
  -> Foxglove live server and/or MCAP sink
```

The sink receives pure owned data. It must never call Isaac, USD, articulation,
planner, or camera APIs from its worker thread.

## Typical Channels

Projects commonly publish:

- JSON scene state for identity-rich inspection;
- joint-state messages for standard robot tooling;
- frame transforms and scene markers;
- RGB/depth camera messages when enabled.

Treat the exact topic names and encodings as an output schema. Change them through a
versioned sink/profile update, not by silently renaming topics in a worker.

## Live Server Security

A visualization listener is an operational network service. Bind it to loopback by
default, do not assume it provides authentication or TLS, and use an authenticated
encrypted proxy for remote access. Camera and state streams may expose sensitive
workspace information.

## MCAP

MCAP output is persistent experiment data. Apply the selected existing-data policy,
write metadata that includes the configuration fingerprint, and close the writer
before destroying the Mirror session. A partial or timed-out close must be reported
as incomplete output.

## Kaleidoscope Metrics

For training, reduce task metrics on CUDA and send only small aggregates to an
external logger at a controlled cadence. Do not attach a Foxglove state or camera sink
to the environment runtime; doing so would violate the headless GPU-residency
contract.

See [Telemetry](telemetry.md), [Cameras](cameras.md), and
[Outputs](../reference/outputs.md).

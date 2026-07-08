# Documentation Index

Language: [English](index.md) | [中文](../zh-CN/文档索引.md)

This is the entry point for `docs/en/`. The English tree mirrors the Chinese customer-facing documentation. Necessary product and API terms such as `cuMotion`, `Tiled`, `Foxglove`, `Isaac`, `USD`, and `TCP Frame` are kept in English.

## Configuration And Naming

- [Configuration Guide](configuration-and-naming/configuration-guide.md): responsibilities of `configs/` profiles, references between profiles, and common run commands.
- [Asset Naming](configuration-and-naming/asset-naming.md): naming rules for assets, joints, links/bodies, and configuration profiles.

## Interaction And Runtime

- [Interactive Simulation](interaction-and-runtime/interactive-simulation.md): single-arm, dual-arm, and tiled interactive entrypoints, JSON protocols, and common commands.
- [Realtime State Stream](interaction-and-runtime/realtime-state-stream.md): Foxglove live, MCAP, and state snapshot output for single-arm and dual-arm interactive runtimes.

## Tiled Environments

- [Tiled Usage And Command Format](tiled-environments/tiled-usage-and-command-format.md): tiled environment configuration, interaction protocol, trajectory buffers, and async planner usage.

## Motion Planning

- [cuMotion Backend API](motion-planning/cumotion-backend-api.md): cuMotion backend interfaces, requests/results, path conversion, and trajectory adapters.
- [cuMotion Motion Modes And Parameters](motion-planning/cumotion-motion-modes-and-parameters.md): Python/JSON examples for different motion modes and their boundaries.

## Telemetry And Sensors

- [Foxglove Data](telemetry-and-sensors/foxglove-data.md): Foxglove live, MCAP, state streams, camera images, and effort fields.
- [Camera Types And Sensors](telemetry-and-sensors/camera-types-and-sensors.md): GUI viewport versus simulated sensor cameras, configuration, and output conventions.

## Assets And Scenes

- [Object Asset Generation](assets-and-scenes/object-asset-generation.md): offline generation of capsule rope, T block, and related USD/USDA assets.
- [Isaac Collision Approximation](assets-and-scenes/isaac-collision-approximation.md): Isaac importer collision approximation fields and USD/PhysX semantics.
- [USD Asset Preview](assets-and-scenes/usd-asset-preview.md): commands and checks for previewing `.usd` / `.usda` assets in Isaac Sim.

## Risks And Constraints

- [Known Risks And Design Constraints](risks-and-constraints/known-risks-and-design-constraints.md): known simulation risks, design constraints, and regression guards.

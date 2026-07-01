"""SimulationApp lifecycle helpers."""

from __future__ import annotations


def close_simulation_app(app) -> None:
    """Close Isaac ``SimulationApp`` without turning shutdown into process exit."""

    try:
        app.close()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise

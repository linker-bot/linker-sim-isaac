"""linker-lab: the Linker LW-v0 workstation for Isaac Lab.

Call :func:`bootstrap` before importing Isaac Lab. Importing ``linker_lab`` itself
starts nothing and imports neither Isaac Lab nor the physics engine.
"""

__all__ = ["bootstrap", "__version__"]

__version__ = "1.0.0"


def bootstrap() -> None:
    """Prepare the pinned physics engine for Isaac Lab (see ``linker_lab.engine``)."""
    from linker_lab import engine

    engine.apply()

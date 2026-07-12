"""PEP 517 backend that enforces this repository's workspace-only boundary."""

from __future__ import annotations

from typing import Any, NoReturn


_BUILD_ERROR = (
    "linkerbot-sim is a workspace application, not a distributable Python "
    "package; run it from the repository root with PYTHONPATH=src"
)


def _reject_build() -> NoReturn:
    raise RuntimeError(_BUILD_ERROR)


def get_requires_for_build_wheel(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    _reject_build()


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    _reject_build()


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _reject_build()


def get_requires_for_build_sdist(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    _reject_build()


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    _reject_build()


def get_requires_for_build_editable(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    _reject_build()


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    _reject_build()


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _reject_build()

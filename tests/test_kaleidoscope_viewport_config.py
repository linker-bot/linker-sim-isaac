from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from linkerbot_sim.configuration import (
    KaleidoscopeViewportSettings,
    load_kaleidoscope_config,
    load_kaleidoscope_viewport_config,
    semantic_config_fingerprint,
    semantic_config_payload,
)
from linkerbot_sim.configuration.common import ConfigurationError


def _write_viewport(root: Path, payload: str) -> Path:
    path = root / "visualization" / "kaleidoscope.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _canonical_text() -> str:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "visualization"
        / "kaleidoscope.yaml"
    )
    return path.read_text(encoding="utf-8")


def test_default_kaleidoscope_viewport_profile_is_strictly_typed() -> None:
    config = load_kaleidoscope_viewport_config()

    assert isinstance(config, KaleidoscopeViewportSettings)
    assert config.selected_env == 0
    assert config.render_every_n_steps == 1
    assert (config.width, config.height) == (1280, 720)
    assert (config.window_width, config.window_height) == (1440, 900)
    assert config.renderer == "RaytracedLighting"
    assert config.anti_aliasing == 0
    assert config.samples_per_pixel_per_frame == 1
    assert config.denoiser is False
    assert config.visuals.viewport.enabled is True
    assert config.visuals.key_light.path == "/World/KeyLight"
    assert config.visuals.fill_light.path == "/World/FillLight"


def test_viewport_config_accepts_explicit_path_within_config_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "configs"
    path = _write_viewport(root, _canonical_text())

    config = load_kaleidoscope_viewport_config(path, configs_root=root)

    assert config.selected_env == 0


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (
            "  selected_env: 0",
            "  selected_env: false",
            "selected_env must be an integer",
        ),
        ("  render_every_n_steps: 1", "  render_every_n_steps: 0", "must be >= 1"),
        ("  denoiser: false", "  denoiser: 0", "denoiser must be a YAML boolean"),
        ("  renderer: RaytracedLighting", "  renderer: ' '", "renderer must be"),
    ],
)
def test_viewport_config_rejects_invalid_scalar_types_and_ranges(
    tmp_path: Path,
    old: str,
    new: str,
    expected: str,
) -> None:
    root = tmp_path / "configs"
    text = _canonical_text()
    assert old in text
    _write_viewport(root, text.replace(old, new, 1))

    with pytest.raises(ConfigurationError, match=expected):
        load_kaleidoscope_viewport_config(configs_root=root)


def test_viewport_config_rejects_unknown_root_and_nested_visual_keys(
    tmp_path: Path,
) -> None:
    root = tmp_path / "configs"
    text = _canonical_text()
    _write_viewport(
        root,
        text.replace("  selected_env: 0", "  typo: true\n  selected_env: 0", 1),
    )
    with pytest.raises(ConfigurationError, match="contains unknown fields: typo"):
        load_kaleidoscope_viewport_config(configs_root=root)

    _write_viewport(
        root,
        text.replace(
            "    viewport:",
            "    unsupported: true\n    viewport:",
            1,
        ),
    )
    with pytest.raises(
        ConfigurationError,
        match="visuals contains unknown fields: unsupported",
    ):
        load_kaleidoscope_viewport_config(configs_root=root)


def test_viewport_config_stays_outside_episode_semantic_fingerprint() -> None:
    training = load_kaleidoscope_config()
    before = semantic_config_fingerprint(training)
    viewport = load_kaleidoscope_viewport_config()

    changed_viewport = replace(viewport, selected_env=7, render_every_n_steps=3)

    assert semantic_config_fingerprint(training) == before
    assert "visualization" not in semantic_config_payload(training)
    assert "viewport" not in semantic_config_payload(training)
    assert changed_viewport.selected_env == 7

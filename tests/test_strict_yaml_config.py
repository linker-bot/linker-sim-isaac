from __future__ import annotations

from pathlib import Path

import pytest

from linkerbot_sim.utils.config import load_yaml


def _yaml_file(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_yaml_preserves_normal_safe_mapping_behavior(tmp_path: Path) -> None:
    path = _yaml_file(
        tmp_path,
        """
name: example
enabled: true
limits:
  count: 3
  values: [1.5, null]
""",
    )

    assert load_yaml(path) == {
        "name": "example",
        "enabled": True,
        "limits": {"count": 3, "values": [1.5, None]},
    }


@pytest.mark.parametrize(
    "content", ["", "# comment only\n", "[]\n", "false\n", "0\n", '""\n']
)
def test_load_yaml_rejects_empty_or_non_mapping_documents(
    tmp_path: Path,
    content: str,
) -> None:
    path = _yaml_file(tmp_path, content)

    with pytest.raises(ValueError, match="expected a mapping|Expected a mapping"):
        load_yaml(path)


@pytest.mark.parametrize(
    ("content", "key", "duplicate_line"),
    [
        ("name: first\nname: second\n", "name", 2),
        ("outer:\n  nested:\n    value: 1\n    value: 2\n", "value", 4),
        ("items:\n  - id: first\n    id: second\n", "id", 3),
    ],
)
def test_load_yaml_rejects_duplicate_keys_at_every_mapping_depth(
    tmp_path: Path,
    content: str,
    key: str,
    duplicate_line: int,
) -> None:
    path = _yaml_file(tmp_path, content)

    with pytest.raises(ValueError) as caught:
        load_yaml(path)

    message = str(caught.value)
    assert f"duplicate mapping key {key!r}" in message
    assert str(path) in message
    assert f"line {duplicate_line}, column" in message
    assert "first occurrence at line" in message


def test_load_yaml_retains_safe_loader_tag_restrictions(tmp_path: Path) -> None:
    path = _yaml_file(
        tmp_path,
        "value: !!python/object/apply:builtins.str [unsafe]\n",
    )

    with pytest.raises(ValueError, match="could not determine a constructor"):
        load_yaml(path)

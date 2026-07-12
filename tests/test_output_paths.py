from __future__ import annotations

from pathlib import Path

import pytest

from linkerbot_sim.utils.output_paths import (
    apply_output_path_plans,
    plan_output_directory,
    plan_output_file,
)


def test_error_policy_rejects_existing_target_during_preflight(
    tmp_path: Path,
) -> None:
    target = tmp_path / "frames"
    target.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        plan_output_directory(target, policy="error")


def test_truncate_policy_does_not_mutate_until_all_plans_apply(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    first.mkdir()
    payload = first / "old.bin"
    payload.write_bytes(b"old")
    invalid = tmp_path / "not-a-directory"
    invalid.write_text("file", encoding="utf-8")

    first_plan = plan_output_directory(first, policy="truncate")
    with pytest.raises(ValueError, match="must be a directory"):
        plan_output_directory(invalid, policy="truncate")

    assert payload.read_bytes() == b"old"
    apply_output_path_plans((first_plan,))
    assert first.is_dir()
    assert list(first.iterdir()) == []


def test_resume_preserves_target_and_timestamped_dir_redirects_output(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "frames"
    existing.mkdir()
    payload = existing / "old.bin"
    payload.write_bytes(b"old")

    resume = plan_output_directory(existing, policy="resume")
    timestamped = plan_output_file(
        tmp_path / "capture.mcap",
        policy="timestamped_dir",
        run_name="20260711T120000.000000Z",
    )
    apply_output_path_plans((resume, timestamped))

    assert resume.resolved_path == existing
    assert payload.read_bytes() == b"old"
    assert timestamped.resolved_path == (
        tmp_path / "20260711T120000.000000Z" / "capture.mcap"
    )
    assert timestamped.resolved_path.parent.is_dir()
    assert not timestamped.resolved_path.exists()


def test_apply_revalidates_every_plan_before_mutating_any_target(
    tmp_path: Path,
) -> None:
    truncate_target = tmp_path / "truncate"
    truncate_target.mkdir()
    payload = truncate_target / "old.bin"
    payload.write_bytes(b"old")
    error_target = tmp_path / "new.csv"

    truncate = plan_output_directory(truncate_target, policy="truncate")
    exclusive = plan_output_file(error_target, policy="error")
    error_target.write_text("raced", encoding="utf-8")

    with pytest.raises(FileExistsError, match="changed after preflight"):
        apply_output_path_plans((truncate, exclusive))

    assert payload.read_bytes() == b"old"


def test_apply_rejects_overlapping_output_namespaces_before_mutation(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "capture"
    directory.mkdir()
    old = directory / "old.bin"
    old.write_bytes(b"old")
    directory_plan = plan_output_directory(directory, policy="truncate")
    nested_file_plan = plan_output_file(
        directory / "camera.mcap",
        policy="truncate",
    )

    with pytest.raises(ValueError, match="overlapping output paths"):
        apply_output_path_plans((directory_plan, nested_file_plan))

    assert old.read_bytes() == b"old"


def test_apply_rejects_duplicate_targets_reached_through_parent_symlink(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    direct = plan_output_file(real_parent / "capture.mcap", policy="error")
    aliased = plan_output_file(alias_parent / "capture.mcap", policy="error")

    with pytest.raises(ValueError, match="same path"):
        apply_output_path_plans((direct, aliased))

    assert not (real_parent / "capture.mcap").exists()


@pytest.mark.parametrize("policy", ("append", "timestamped", "overwrite", ""))
def test_output_path_policy_is_strict(policy: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing_data_policy"):
        plan_output_file(tmp_path / "output.csv", policy=policy)

#!/usr/bin/env python3
"""Verify that an active GitHub ruleset enforces the maintained branch policy."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import re
import sys
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO_ROOT / ".github" / "rulesets" / "master.json"
API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
JsonFetcher = Callable[[str], object]


class RulesetError(RuntimeError):
    """Raised when the desired or active repository policy is not trustworthy."""


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise RulesetError(f"{label} must be a JSON object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise RulesetError(f"{label} must be a JSON array")
    return value


def _rules_by_type(
    payload: Mapping[str, object], *, label: str
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(
        _sequence(payload.get("rules"), label=f"{label}.rules")
    ):
        rule = _mapping(value, label=f"{label}.rules[{index}]")
        rule_type = rule.get("type")
        if not isinstance(rule_type, str) or not rule_type:
            raise RulesetError(f"{label}.rules[{index}].type must be non-empty text")
        if rule_type in result:
            raise RulesetError(f"{label} repeats the {rule_type!r} rule")
        result[rule_type] = rule
    return result


def load_policy(path: Path = DEFAULT_POLICY) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RulesetError(f"cannot read ruleset policy {path}: {exc}") from exc
    return _mapping(value, label="ruleset policy")


def validate_policy(policy: Mapping[str, object]) -> None:
    if policy.get("target") != "branch" or policy.get("enforcement") != "active":
        raise RulesetError("policy must actively enforce a branch ruleset")
    if policy.get("bypass_actors") != []:
        raise RulesetError("policy must not declare bypass actors")

    conditions = _mapping(policy.get("conditions"), label="policy.conditions")
    ref_name = _mapping(conditions.get("ref_name"), label="policy.conditions.ref_name")
    if ref_name.get("include") != ["~DEFAULT_BRANCH"] or ref_name.get("exclude") != []:
        raise RulesetError("policy must target only the default branch")

    rules = _rules_by_type(policy, label="policy")
    required_types = {
        "deletion",
        "non_fast_forward",
        "pull_request",
        "required_status_checks",
    }
    if set(rules) != required_types:
        raise RulesetError(
            f"policy rule types must be exactly {sorted(required_types)}"
        )

    pull_request = _mapping(
        rules["pull_request"].get("parameters"), label="policy.pull_request.parameters"
    )
    if pull_request.get("required_approving_review_count") != 1:
        raise RulesetError("policy must require one approving review")
    for name in (
        "dismiss_stale_reviews_on_push",
        "require_last_push_approval",
        "required_review_thread_resolution",
    ):
        if pull_request.get(name) is not True:
            raise RulesetError(f"policy pull request parameter {name} must be true")
    if pull_request.get("require_code_owner_review") is not False:
        raise RulesetError(
            "policy must not claim an unconfigured CODEOWNERS requirement"
        )
    merge_methods = pull_request.get("allowed_merge_methods")
    if not isinstance(merge_methods, list) or not merge_methods:
        raise RulesetError("policy must allow at least one merge method")
    if not set(merge_methods) <= {"merge", "squash", "rebase"}:
        raise RulesetError("policy contains an unknown merge method")

    status = _mapping(
        rules["required_status_checks"].get("parameters"),
        label="policy.required_status_checks.parameters",
    )
    if status.get("strict_required_status_checks_policy") is not True:
        raise RulesetError("policy must require branches to be current before merging")
    checks = _sequence(
        status.get("required_status_checks"),
        label="policy.required_status_checks",
    )
    if len(checks) != 1:
        raise RulesetError("policy must require exactly one always-on status check")
    quality = _mapping(checks[0], label="policy.required_status_checks[0]")
    if (
        quality.get("context") != "CPU quality"
        or quality.get("integration_id") != 15368
    ):
        raise RulesetError(
            "policy must require 'CPU quality' from the GitHub Actions app"
        )


def _fetch_json(url: str) -> object:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "linker-sim-isaac-ruleset-audit",
        "X-GitHub-Api-Version": API_VERSION,
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=15) as response:  # noqa: S310
            return json.load(response)
    except HTTPError as exc:
        raise RulesetError(f"GitHub API returned HTTP {exc.code} for {url}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RulesetError(f"GitHub API request failed for {url}: {exc}") from exc


def _active_ruleset_satisfies(
    actual: Mapping[str, object], desired: Mapping[str, object]
) -> bool:
    if actual.get("target") != "branch" or actual.get("enforcement") != "active":
        return False
    bypass_actors = actual.get("bypass_actors")
    if bypass_actors is not None and bypass_actors != []:
        return False
    conditions = _mapping(actual.get("conditions"), label="active.conditions")
    ref_name = _mapping(conditions.get("ref_name"), label="active.conditions.ref_name")
    include = ref_name.get("include")
    exclude = ref_name.get("exclude")
    if not isinstance(include, list) or "~DEFAULT_BRANCH" not in include:
        return False
    if exclude not in (None, []):
        return False

    actual_rules = _rules_by_type(actual, label="active")
    desired_rules = _rules_by_type(desired, label="policy")
    if not set(desired_rules) <= set(actual_rules):
        return False

    actual_pull = _mapping(
        actual_rules["pull_request"].get("parameters"),
        label="active.pull_request.parameters",
    )
    desired_pull = _mapping(
        desired_rules["pull_request"].get("parameters"),
        label="policy.pull_request.parameters",
    )
    required_reviews = actual_pull.get("required_approving_review_count")
    desired_reviews = desired_pull.get("required_approving_review_count")
    if not isinstance(required_reviews, int) or not isinstance(desired_reviews, int):
        return False
    if required_reviews < desired_reviews:
        return False
    for name in (
        "dismiss_stale_reviews_on_push",
        "require_last_push_approval",
        "required_review_thread_resolution",
    ):
        if desired_pull.get(name) is True and actual_pull.get(name) is not True:
            return False

    actual_status = _mapping(
        actual_rules["required_status_checks"].get("parameters"),
        label="active.required_status_checks.parameters",
    )
    if actual_status.get("strict_required_status_checks_policy") is not True:
        return False
    actual_checks = _sequence(
        actual_status.get("required_status_checks"),
        label="active.required_status_checks",
    )
    return any(
        check.get("context") == "CPU quality" and check.get("integration_id") == 15368
        for check in (
            _mapping(value, label="active.required_status_checks[]")
            for value in actual_checks
        )
    )


def audit_repository(
    repository: str,
    policy: Mapping[str, object],
    *,
    fetch_json: JsonFetcher = _fetch_json,
) -> Mapping[str, object]:
    if not REPOSITORY_RE.fullmatch(repository):
        raise RulesetError("repository must use the owner/name form")
    owner, name = repository.split("/", 1)
    if owner in {".", ".."} or name in {".", ".."}:
        raise RulesetError("repository owner and name must be concrete")
    validate_policy(policy)

    summaries = _sequence(
        fetch_json(
            f"{API_ROOT}/repos/{repository}/rulesets?includes_parents=true&per_page=100"
        ),
        label="GitHub ruleset list",
    )
    candidates: list[Mapping[str, object]] = []
    for index, value in enumerate(summaries):
        summary = _mapping(value, label=f"GitHub ruleset list[{index}]")
        if summary.get("target") == "branch" and summary.get("enforcement") == "active":
            ruleset_id = summary.get("id")
            if not isinstance(ruleset_id, int) or ruleset_id <= 0:
                raise RulesetError("active GitHub ruleset has an invalid id")
            detail = _mapping(
                fetch_json(
                    f"{API_ROOT}/repos/{repository}/rulesets/{ruleset_id}?includes_parents=true"
                ),
                label=f"GitHub ruleset {ruleset_id}",
            )
            candidates.append(detail)

    for actual in candidates:
        if _active_ruleset_satisfies(actual, policy):
            return actual
    raise RulesetError(
        "no active ruleset protects the default branch with review, strict CPU quality, "
        "deletion, and force-push constraints"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", required=True, help="GitHub repository as owner/name"
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--validate-policy-only",
        action="store_true",
        help="validate the maintained JSON without contacting GitHub",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = load_policy(args.policy)
        validate_policy(policy)
        if args.validate_policy_only:
            print(
                json.dumps({"event": "repository_ruleset_policy_valid"}, sort_keys=True)
            )
            return 0
        actual = audit_repository(args.repository, policy)
    except RulesetError as exc:
        print(f"repository ruleset error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "event": "repository_ruleset_validated",
                "repository": args.repository,
                "ruleset_id": actual.get("id"),
                "ruleset_name": actual.get("name"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

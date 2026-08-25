# Repository Governance

Language: [English](repository-governance.md) | [中文](../../zh-CN/operations/repository-governance.md)

The CPU quality workflow runs on every pull request, but a workflow is only evidence
until the default branch requires it. The maintained policy is
`.github/rulesets/master.json`. It targets the repository's default branch and requires:

- changes through pull requests;
- at least one approving review, with stale approvals dismissed after new pushes;
- approval of the latest push by someone other than its author;
- resolution of review conversations;
- the strict, always-on `CPU quality` check;
- rejection of branch deletion and non-fast-forward pushes.

The policy intentionally has no bypass actors. It does not require the path-filtered
dependency audit because GitHub does not create that check for unrelated changes. It
also does not send pull-request code to the self-hosted Simulation runner; relevant
changes still need the reviewed manual run described in [Simulation CI](simulation-ci.md).

## Apply The Ruleset

Merging the JSON file does not change repository settings. A repository administrator
must create the ruleset once, either through **Settings → Rules → Rulesets** or with a
fine-grained token that has repository Administration write permission:

```bash
gh api --method POST \
  repos/linker-bot/linker-sim-isaac/rulesets \
  --input .github/rulesets/master.json
```

Do not repeat `POST`: list the existing rulesets and use `PATCH` with its exact numeric
ID when policy changes. Compare the active settings with the reviewed JSON before
updating them. Keep the required context spelled exactly as `CPU quality`, matching the
job name in `.github/workflows/quality.yml`, and keep its source restricted to the
GitHub Actions app (`integration_id` 15368).

## Audit Drift

After applying the ruleset, run:

```bash
just check-repository-policy
```

The `Repository Policy` workflow performs the same read-only check weekly and on
manual dispatch. It verifies active default-branch targeting, reviews, current-branch
status checks, deletion protection, and force-push protection. It never creates or
updates repository settings.

GitHub does not expose bypass actors to a public metadata reader without ruleset write
access. Review the settings page after ownership or administrator changes and confirm
that the active ruleset still has no bypass actors.

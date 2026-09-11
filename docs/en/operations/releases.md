# Releases

Language: [English](releases.md) | [中文](../../zh-CN/operations/releases.md)

linker-sim-isaac publishes a versioned source workspace archive through GitHub
Releases. It does not publish a wheel, sdist, container image, or PyPI distribution;
runtime profiles, Kit experiences, configuration, scripts, and assets must remain in
one checkout.

## One-Time Repository Setup

Create a GitHub environment named `release`. Restrict deployment access to project
maintainers and add required reviewers when the repository plan supports them. The
workflow receives only `actions: read` and `contents: write`; no checkout credential is
persisted.

## Prepare A Version

1. Update `project.version` in `pyproject.toml` and `linkerbot_sim.__version__` to the
   same semantic version.
2. Move user-visible entries from `[Unreleased]` into dated sections with the exact
   version in `CHANGELOG.md` and `CHANGELOG_zh.md`.
3. Merge the version change after `just quality` passes.
4. On the merged commit, create and push an annotated tag:

   ```bash
   git tag -a v0.3.0 -m "linker-sim-isaac 0.3.0"
   git push origin v0.3.0
   ```

Do not move or replace a published tag. Correct a released defect with a new patch
version.

## Produce Acceptance Evidence

From **Actions → Simulation → Run workflow**, select the tag or its exact reviewed
repository branch. Wait for the complete matrix to succeed and record the numeric run
ID from its URL. A failed, cancelled, skipped, or timed-out run is not release evidence.

The GPU workflow is currently manual-only while runner stability issues are being
resolved. The release workflow therefore requires an explicit successful Simulation
run whose `headSha` equals the annotated tag commit; it cannot reuse evidence from a
different revision.

## Publish

From **Actions → Release → Run workflow**, enter the annotated tag, Simulation run ID,
and prerelease choice. The workflow then:

1. checks out the exact tag and verifies that it is annotated and matches both version
   declarations;
2. verifies the changelog entry and successful Simulation evidence from the same
   commit;
3. runs the locked CPU `just quality` gate;
4. creates `linker-sim-isaac-VERSION.tar.gz` with `git archive` and `SHA256SUMS`;
5. creates the GitHub Release only for the existing tag.

The workflow will not create a missing tag. If validation or upload fails, keep the
tag unchanged, correct the release tooling in a reviewed change, and rerun. If the
tagged source itself is wrong, publish a new patch version instead of retargeting the
tag.

Verify a downloaded archive with:

```bash
sha256sum --check SHA256SUMS
```

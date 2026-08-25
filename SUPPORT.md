# Support

Language: [English](SUPPORT.md) | [中文](SUPPORT_zh.md)

Use the public tracker for reproducible project questions and defects. Search existing
issues first, then choose the matching template:

- [Bug report](https://github.com/linker-bot/linker-sim-isaac/issues/new?template=bug_report.md)
  for reproducible incorrect behavior.
- [Feature request](https://github.com/linker-bot/linker-sim-isaac/issues/new?template=feature_request.md)
  for a proposed public capability or contract change.
- [Question](https://github.com/linker-bot/linker-sim-isaac/issues/new?template=question.md)
  for documented setup and usage questions.
- [Security policy](SECURITY.md) for vulnerabilities. Do not disclose them in a public
  issue.

Include the output of `python scripts/mirror.py --version`, `git rev-parse HEAD`, the
selected product and profile, OS/Python/Isaac/GPU versions when relevant, the config
validation fingerprint, a minimal reproduction, and complete error text. Remove
credentials, tokens, customer data, internal hostnames, and private filesystem paths
before posting.

The public tracker covers this repository's maintained source, documented
configuration, and reproducible runtime behavior. It does not promise private
deployment operation, hardware procurement, custom scene authoring, or support for
unpublished local modifications. A maintainer may close requests that cannot be
reproduced or that omit the information required by the selected template.

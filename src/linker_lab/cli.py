"""The ``linker-lab`` command."""

from __future__ import annotations

import argparse
import sys


def doctor() -> int:
    """Check the installation: versions, GPU, engine patches, and a short simulation."""
    import linker_lab
    from linker_lab import engine

    linker_lab.bootstrap()
    checks = [*engine.check_versions(), engine.check_gpu(), engine.check_patches()]
    if all(c.ok for c in checks):
        checks.append(engine.check_simulation())
    print(f"\nlinker-lab {linker_lab.__version__} doctor")
    for c in checks:
        print(f"  [{'ok' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
    ok = all(c.ok for c in checks)
    print("All checks passed." if ok else "Some checks failed; see above.")
    return 0 if ok else 1


def play(selftest: bool = False, out: str | None = None) -> int:
    """Drive LW-v0 by hand in the viewer (``--selftest``: scripted keys, then a report)."""
    import json
    from pathlib import Path

    from linker_lab import playground

    if not selftest:
        return playground.run()
    report = playground.selftest(Path(out) if out else None)
    print(json.dumps(report, indent=1))
    print("Self-test passed." if report["passed"] else "Self-test FAILED; see above.")
    return 0 if report["passed"] else 1


def gallery(command: str, name: str | None) -> int:
    """Pretrained policies on LW-v0: list the entries, show one, or open one in the viewer."""
    from linker_lab import gallery as g

    if command == "list":
        for entry in g.entries():
            data = g.manifest(entry)
            supported = "" if g.supported(data) else "  (not supported)"
            print(f"{entry:16s} {data['title']}{supported}")
    elif command == "info":
        print(g.info(name))
    else:
        g.run(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="linker-lab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help=doctor.__doc__)
    play_parser = commands.add_parser("play", help=play.__doc__)
    play_parser.add_argument("--selftest", action="store_true", help="scripted keys, then a report")
    play_parser.add_argument("--out", help="with --selftest: write the report and screenshots here")
    gallery_parser = commands.add_parser("gallery", help=gallery.__doc__)
    gallery_commands = gallery_parser.add_subparsers(dest="gallery_command", required=True)
    gallery_commands.add_parser("list", help="the gallery's entries")
    gallery_commands.add_parser("info", help="an entry's description and license").add_argument(
        "entry"
    )
    gallery_commands.add_parser("run", help="open an entry in the viewer").add_argument("entry")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "play":
        return play(args.selftest, args.out)
    if args.command == "gallery":
        return gallery(args.gallery_command, getattr(args, "entry", None))
    return 2


if __name__ == "__main__":
    sys.exit(main())

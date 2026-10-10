"""Prepare a temporary MJCF with an explicit, overridable contact fallback."""

from math import isfinite
from pathlib import Path
import xml.etree.ElementTree as ET


def prepare_mjcf_contact_defaults(
    source: Path, destination: Path, *, time_constant_s: float
) -> Path:
    """Let MuJoCo resolve geom/default inheritance, without rewriting authored values.

    Isaac 6.1 serializes even the built-in MuJoCo solref default into USD. Supply
    the project's fallback before conversion instead of guessing which resulting
    USD attributes were authored. Includes are expanded only to find the main
    default; all class, childclass and per-geom inheritance remains MuJoCo's job.
    The source and its dependencies are read-only.
    """
    if not isfinite(time_constant_s) or time_constant_s <= 0:
        raise ValueError("contact time constant must be finite and positive")
    source = source.resolve()
    root = _read_with_includes(source, ancestors=())
    default = root.find("default")
    if default is None:
        default = ET.Element("default")
        root.insert(0, default)
    geom = default.find("geom")
    if geom is None:
        geom = ET.Element("geom")
        default.insert(0, geom)
    if "solref" not in geom.attrib:
        geom.set("solref", f"{time_constant_s!r} 1")

    # Asset paths are relative to the main MJCF, not the temporary copy. Keep
    # MuJoCo's compiler path resolution (including mesh/texture strippath) intact.
    compiler_defaults = ET.Element(
        "compiler",
        meshdir=str(source.parent),
        texturedir=str(source.parent),
    )
    root.insert(0, compiler_defaults)
    for compiler in root.findall("compiler"):
        for attribute in ("assetdir", "meshdir", "texturedir"):
            if attribute in compiler.attrib:
                compiler.set(
                    attribute,
                    str((source.parent / compiler.attrib[attribute]).resolve()),
                )
    # Attached MJCF models use the model-file directory rather than meshdir.
    for model in root.findall("./asset/model"):
        if "file" in model.attrib:
            model.set("file", str((source.parent / model.attrib["file"]).resolve()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def _read_with_includes(path: Path, *, ancestors: tuple[Path, ...]) -> ET.Element:
    if path in ancestors:
        raise ValueError(f"recursive MJCF include: {path}")
    root = ET.parse(path).getroot()

    def expand(parent: ET.Element) -> None:
        for child in list(parent):
            if child.tag != "include":
                expand(child)
                continue
            included = _read_with_includes(
                (path.parent / child.attrib["file"]).resolve(),
                ancestors=(*ancestors, path),
            )
            index = list(parent).index(child)
            parent.remove(child)
            for offset, element in enumerate(included):
                parent.insert(index + offset, element)

    expand(root)
    return root

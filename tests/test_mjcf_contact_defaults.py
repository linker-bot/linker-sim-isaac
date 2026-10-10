import xml.etree.ElementTree as ET

import pytest

from linkerbot_sim.assets.mjcf_defaults import prepare_mjcf_contact_defaults


def test_default_preserves_authored_values_and_expands_nested_include(tmp_path):
    source = tmp_path / "model.xml"
    part = tmp_path / "parts"
    part.mkdir()
    (part / "classes.xml").write_text(
        '<mujocoinclude><default><default class="soft">'
        '<geom solref=".02 1"/><default class="nested"/>'
        "</default></default></mujocoinclude>"
    )
    (part / "body.xml").write_text(
        '<mujocoinclude><body childclass="nested">'
        '<geom name="inherited" type="sphere" size=".1"/>'
        '<geom name="explicit" type="sphere" size=".1" solref=".03 2"/>'
        "</body></mujocoinclude>"
    )
    source.write_text(
        '<mujoco><include file="parts/classes.xml"/><worldbody>'
        '<geom name="implicit" type="sphere" size=".1"/>'
        '<include file="parts/body.xml"/></worldbody></mujoco>'
    )
    original = source.read_bytes()
    output = prepare_mjcf_contact_defaults(
        source, tmp_path / "output/model.xml", time_constant_s=0.004
    )
    root = ET.parse(output).getroot()
    assert root.find("./default/geom").get("solref") == "0.004 1"
    assert root.find("./default/default/geom").get("solref") == ".02 1"
    assert root.find(".//geom[@name='explicit']").get("solref") == ".03 2"
    assert root.find(".//body").get("childclass") == "nested"
    assert not root.findall(".//include")
    assert source.read_bytes() == original


def test_authored_main_default_equal_to_mujoco_default_is_preserved(tmp_path):
    source = tmp_path / "model.xml"
    source.write_text('<mujoco><default><geom solref=".02 1"/></default></mujoco>')
    output = prepare_mjcf_contact_defaults(
        source, tmp_path / "output.xml", time_constant_s=0.004
    )
    assert ET.parse(output).find("./default/geom").get("solref") == ".02 1"


def test_new_default_and_source_asset_paths(tmp_path):
    source = tmp_path / "model.xml"
    source.write_text(
        '<mujoco><compiler meshdir="meshes" texturedir="../textures"/>'
        '<asset><model name="child" file="child.xml"/></asset></mujoco>'
    )
    output = prepare_mjcf_contact_defaults(
        source, tmp_path / "output/model.xml", time_constant_s=0.008
    )
    root = ET.parse(output).getroot()
    assert root.find("./default/geom").get("solref") == "0.008 1"
    compiler = root.findall("compiler")[-1]
    assert compiler.get("meshdir") == str(tmp_path / "meshes")
    assert compiler.get("texturedir") == str(tmp_path.parent / "textures")
    assert root.find("./asset/model").get("file") == str(tmp_path / "child.xml")


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_time_constant(tmp_path, value):
    with pytest.raises(ValueError, match="finite and positive"):
        prepare_mjcf_contact_defaults(
            tmp_path / "missing.xml", tmp_path / "output.xml", time_constant_s=value
        )


def test_recursive_include_is_rejected(tmp_path):
    source = tmp_path / "model.xml"
    source.write_text('<mujoco><include file="model.xml"/></mujoco>')
    with pytest.raises(ValueError, match="recursive MJCF include"):
        prepare_mjcf_contact_defaults(
            source, tmp_path / "out.xml", time_constant_s=0.004
        )

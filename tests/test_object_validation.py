"""Invalid physical parameters fail before USD creation, including explicit inertia inputs."""

import numpy as np
import pytest
import trimesh
from pxr import Usd, UsdPhysics

from linker_lab import objects


@pytest.fixture
def parts(tmp_path, monkeypatch):
    monkeypatch.setattr(objects, "cache_dir", lambda: tmp_path)
    mesh = trimesh.creation.box((0.05, 0.05, 0.05))
    return [(mesh.vertices, mesh.faces)]


@pytest.mark.parametrize("mass", [0, -1, np.nan, np.inf, -np.inf, [1], 1 + 1j, 1e100, 1e-100])
@pytest.mark.parametrize("explicit_inertia", [False, True])
def test_write_rejects_invalid_mass(parts, tmp_path, mass, explicit_inertia):
    kwargs = {"com": (0, 0, 0), "inertia": np.eye(3) * 1e-3} if explicit_inertia else {}
    with pytest.raises(ValueError, match="mass"):
        objects.write_usd(parts, mass, **kwargs)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mass", [0, -1, np.nan, np.inf, [1], 1 + 1j])
def test_mass_properties_rejects_invalid_mass(parts, mass):
    with pytest.raises(ValueError, match="mass"):
        objects.mass_properties(parts, mass)


@pytest.mark.parametrize(
    "com",
    [
        (0, 0),
        ((0, 0, 0),),
        (0, np.nan, 0),
        (0, np.inf, 0),
        (0, 1j, 0),
        (1e100, 0, 0),
    ],
)
def test_write_rejects_invalid_center_of_mass(parts, tmp_path, com):
    with pytest.raises(ValueError, match="com"):
        objects.write_usd(parts, 0.1, com, np.eye(3) * 1e-3)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "inertia, message",
    [
        (np.ones(3), "shape"),
        (np.eye(2), "shape"),
        (np.diag([np.nan, 1, 1]), "finite real"),
        (np.diag([np.inf, 1, 1]), "finite real"),
        (np.eye(3, dtype=complex), "finite real"),
        (np.array([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]]), "symmetric"),
        (np.diag([-1, 1, 1]), "physical inertia"),
        (np.diag([0, 1, 1]), "physical inertia"),
        (np.diag([1, 1, 3]), "physical inertia"),
        (np.eye(3) * 1e100, "USD float precision"),
        (np.eye(3) * 1e-100, "USD float precision"),
    ],
)
def test_write_rejects_invalid_inertia(parts, tmp_path, inertia, message):
    with pytest.raises(ValueError, match=message):
        objects.write_usd(parts, 0.1, (0, 0, 0), inertia)
    assert list(tmp_path.iterdir()) == []


def test_roundoff_asymmetry_is_normalized_before_caching(parts):
    inertia = np.array([[3e-4, 1e-5, 0], [1e-5, 3.5e-4, 0], [0, 0, 2e-4]])
    inertia[0, 1] += 1e-12
    symmetric = 0.5 * inertia + 0.5 * inertia.T
    path = objects.write_usd(parts, 0.1, (0, 0, 0), inertia)
    assert objects.write_usd(parts, 0.1, (0, 0, 0), symmetric) == path


def test_triangle_equality_and_numpy_scalar_mass_are_valid(parts):
    path = objects.write_usd(parts, np.float64(0.1), (0, 0, 0), np.diag([1e-3, 1e-3, 2e-3]))
    stage = Usd.Stage.Open(str(path))
    mass = UsdPhysics.MassAPI(stage.GetDefaultPrim())
    assert mass.GetMassAttr().Get() == pytest.approx(0.1)
    assert np.allclose(mass.GetDiagonalInertiaAttr().Get(), (1e-3, 1e-3, 2e-3))


def test_mass_properties_rejects_empty_parts():
    with pytest.raises(ValueError, match="at least one mesh part"):
        objects.mass_properties([], 0.1)

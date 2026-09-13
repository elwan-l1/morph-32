"""Research protocol checks, prepared for later execution."""

import pytest

from morph32.profiles import recipe
from morph32.statistics import accuracy_interval, timing_summary


def test_control_uses_same_three_plane_layout():
    full = recipe("morph32-3s", 0, "up_proj")
    independent = recipe("independent-3s", 0, "up_proj")
    assert independent == {**full, "planes": 3}
    assert (3 * 32 + 16) / 32 == 3.5
    assert recipe("affine-3bit", 0, "up_proj") == dict(bits=3, group_size=64)
    assert (64 * 3 + 2 * 16) / 64 == 3.5


def test_wilson_boundary_and_counts():
    zero = accuracy_interval(0, 10)
    full = accuracy_interval(10, 10)
    assert zero["ci95"][0] == pytest.approx(0, abs=1e-15)
    assert full["ci95"][1] == pytest.approx(1)
    assert zero["ci95"][1] == pytest.approx(1 - full["ci95"][0])
    with pytest.raises(ValueError):
        accuracy_interval(1, 0)


def test_timing_variation_not_just_mean():
    result = timing_summary([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    assert result["count"] == 10
    assert result["median"] == 5.5
    assert result["q1"] == 3.25
    assert result["q3"] == 7.75
    assert result["maximum"] == 100
    with pytest.raises(ValueError):
        timing_summary([float("nan")])

# ruff: noqa: E402
"""Unexecuted numerical tests for the controls and opt-in decoder candidate."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif("M5" not in mx.device_info()["device_name"], reason="M5 required"),
]

from morph32.codec import fit, matvec_fast, reference
from morph32.profiles import recipe
from morph32.scalar import Affine3Linear
from morph32.scalar import fit as scalar_fit
from morph32.scalar import reference as scalar_reference


@pytest.mark.parametrize("profile", ["morph32-3s", "independent-3s", "morph32-c"])
def test_split_accumulator_against_dense_reference(profile):
    mx.random.seed(21)
    weight = mx.random.normal((64, 1024)) * 0.02
    spec = recipe(profile, 16, "up_proj")
    words, scales = fit(weight, **spec)
    x = mx.random.normal((1, 1024))
    expected = np.array(x) @ reference(words, scales, **spec).T
    for optimized in (False, True):
        y = matvec_fast(x, words, scales, **spec, values=32, optimized=optimized)
        np.testing.assert_allclose(np.array(y), expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("batch", [1, 3, 33])
def test_affine3_direct_decode_and_storage(batch):
    mx.random.seed(21)
    weight = mx.random.normal((64, 128)) * 0.02
    words, scales, biases = scalar_fit(weight, mx.ones(128))
    assert (words.nbytes + scales.nbytes + biases.nbytes) * 8 / weight.size == 3.5
    module = Affine3Linear(words, scales, biases)
    x = mx.random.normal((batch, 128))
    expected = np.array(x) @ scalar_reference(words, scales, biases).T
    np.testing.assert_allclose(np.array(module(x)), expected, rtol=2e-4, atol=2e-5)


def test_affine3_constant_and_zero_groups_remain_finite():
    weight = mx.concatenate([mx.zeros((1, 64)), mx.full((1, 64), 0.25)], axis=1)
    packed = scalar_fit(weight, mx.ones(128))
    decoded = scalar_reference(*packed)
    np.testing.assert_allclose(decoded, np.array(weight), atol=1e-4)

# ruff: noqa: E402
# Skip before importing GPU modules on machines without MLX.
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif("M5" not in mx.device_info()["device_name"], reason="M5 NAX required"),
]

from morph32.codec import fit, matvec_fast, reference, unpack
from morph32.conversion import encode_metadata
from morph32.metadata import CorrelatedLinear, MetadataLinear
from morph32.prefill import matmul
from morph32.profiles import recipe


@pytest.mark.parametrize("profile", ["morph32-3s", "morph32-c"])
def test_direct_orbit_kernels(profile):
    mx.random.seed(42)
    weight = mx.random.normal((64, 1024)) * 0.04
    spec = recipe(profile, 16, "up_proj")
    words, scales = fit(weight, **spec)
    dense = unpack(words, scales, **spec)
    mx.eval(words, scales, dense)
    np.testing.assert_array_equal(np.array(dense), reference(words, scales, **spec))
    expected_bits = 3.0 if profile == "morph32-c" else 3.5
    assert (words.nbytes + scales.nbytes) * 8 / weight.size == expected_bits
    for batch in (1, 3, 33, 129):
        x = mx.random.normal((batch, 1024)).astype(mx.bfloat16)
        actual = (
            matvec_fast(x, words, scales, **spec, values=32)
            if batch == 1
            else matmul(x, words, scales, **spec)
        )
        # Prefill rounds reconstructed weights to activation dtype before matrix multiplication.
        expected = (
            x.astype(mx.float32)
            @ (dense if batch == 1 else dense.astype(x.dtype)).astype(mx.float32).T
        ).astype(x.dtype)
        mx.eval(actual, expected)
        relative = mx.linalg.norm(
            actual.astype(mx.float32) - expected.astype(mx.float32)
        ) / mx.linalg.norm(expected.astype(mx.float32))
        assert relative.item() < 0.004
        assert mx.all(mx.isfinite(actual)).item()


def test_retained_metadata_and_execution():
    mx.random.seed(7)
    dense = (mx.random.normal((64, 1024)) * 0.04).astype(mx.bfloat16)
    weight, scales, biases = mx.quantize(dense, group_size=64, bits=4)
    encoded = encode_metadata(scales, biases)
    assert encoded is not None
    arrays, spec = encoded
    if spec["kind"] == "correlated_bf16_metadata":
        module = CorrelatedLinear(weight, **arrays, bits=spec["bits"], decode="packet")
    else:
        module = MetadataLinear(weight, **arrays, bits=spec["bits"])
    for batch in (1, 3):
        x = mx.random.normal((batch, 1024)).astype(mx.bfloat16)
        expected = mx.quantized_matmul(
            x, weight, scales, biases, transpose=True, group_size=64, bits=4
        )
        actual = module(x)
        mx.eval(actual, expected)
        np.testing.assert_array_equal(
            np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32))
        )

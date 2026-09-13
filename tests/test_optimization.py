# ruff: noqa: E402
"""Numerical and decision checks for opt-in optimization experiments."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif("M5" not in mx.device_info()["device_name"], reason="M5 required"),
]
from morph32.codec import fit, matvec_fast, reference, unpack
from morph32.optimization import allocate, joint_scale_fit, refine, second_moments
from morph32.prefill import matmul
from morph32.profiles import recipe


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("covariance", [False, True])
def test_refinement_retains_candidate_and_roundtrips(partial, covariance):
    mx.random.seed(13)
    w = mx.random.normal((4, 64)) * 0.05
    w = mx.concatenate([mx.zeros((1, 64)), w], axis=0)
    spec = recipe("morph32-c" if partial else "morph32-3s", 16, "up_proj")
    x = mx.random.normal((40, 64))
    h = second_moments(x)
    initial = fit(w, **spec)
    args = dict(spec=spec, sweeps=2, restarts=2, seed=7, moments=h if covariance else None)
    actual = refine(w, initial, **args)
    again = refine(w, initial, **args)
    for a, b in zip(actual, again):
        np.testing.assert_array_equal(np.array(a), np.array(b))
    np.testing.assert_array_equal(np.array(unpack(*actual, **spec)), reference(*actual, **spec))

    def loss(pair):
        e = (w - unpack(*pair, **spec)).reshape(5, 2, 32)
        return mx.sum(e * ((e[:, :, None, :] @ h).squeeze(2) if covariance else e), axis=-1)

    assert mx.all(loss(actual) <= loss(initial) + 1e-7).item()
    assert mx.all(mx.isfinite(actual[1])).item()


def test_second_moment_identity_and_diagonal_equivalence():
    mx.random.seed(4)
    x = mx.random.normal((12, 64))
    e = mx.random.normal((2, 32))
    h = second_moments(x)
    lhs = mx.sum(e * mx.matmul(e[:, None, :], h, stream=mx.cpu).squeeze(1), axis=-1)
    rhs = mx.mean(mx.sum(x.reshape(12, 2, 32) * e, axis=-1) ** 2, axis=0)
    np.testing.assert_allclose(np.array(lhs), np.array(rhs), rtol=1e-5)
    spec = recipe("morph32-3s", 0, "up_proj")
    w = mx.random.normal((2, 64)) * 0.02
    initial = fit(w, **spec)
    diagonal = mx.broadcast_to(mx.eye(32), (2, 32, 32))
    a = refine(w, initial, spec=spec, moments=diagonal)
    b = refine(w, initial, spec=spec)
    np.testing.assert_allclose(
        np.array(unpack(*a, **spec)), np.array(unpack(*b, **spec)), atol=1e-7
    )


@pytest.mark.parametrize(
    "coeff",
    [
        (1.0, 0.5, 0.25),
        (1.0, 0.48, 0.23),
        (1.0, 0.5625, 0.28125),
    ],
)
@pytest.mark.parametrize("partial", [False, True])
def test_coefficients_all_paths(coeff, partial):
    mx.random.seed(22)
    spec = recipe("morph32-c" if partial else "morph32-3s", 16, "up_proj")
    spec["planes"] = len(coeff)
    spec["coefficients"] = list(coeff)
    w = mx.random.normal((64, 1024)) * 0.02
    q, s = fit(w, **spec)
    dense = unpack(q, s, **spec)
    np.testing.assert_allclose(np.array(dense), reference(q, s, **spec), atol=1e-7)
    for batch in (1, 3, 33):
        x = mx.random.normal((batch, 1024)).astype(mx.float16)
        y = matvec_fast(x, q, s, **spec, values=32) if batch == 1 else matmul(x, q, s, **spec)
        expected = (
            x.astype(mx.float32)
            @ (dense if batch == 1 else dense.astype(x.dtype)).astype(mx.float32).T
        )
        np.testing.assert_allclose(
            np.array(y.astype(mx.float32)), np.array(expected), atol=0.002, rtol=0.01
        )


def test_allocation_budget_and_infeasible():
    rows = {
        "a": [dict(name="full", bytes=14, loss=1), dict(name="compact", bytes=12, loss=1.1)],
        "b": [dict(name="full", bytes=14, loss=1), dict(name="compact", bytes=12, loss=3)],
    }
    chosen = allocate(rows, 26)
    assert chosen["a"]["name"] == "compact" and chosen["b"]["name"] == "full"
    with pytest.raises(ValueError):
        allocate(rows, 23)


def test_joint_fit_exported_loss_nonincreasing(tmp_path):
    import mlx.nn as nn

    mx.random.seed(5)
    spec = recipe("morph32-3s", 0, "up_proj")
    g = mx.random.normal((32, 64)) * 0.1
    u = mx.random.normal((32, 64)) * 0.1
    x = mx.random.normal((32, 64))
    target = nn.silu(x @ g.T) * (x @ u.T)
    a, b, history = joint_scale_fit(
        x, fit(g, **spec), fit(u, **spec), gate_spec=spec, up_spec=spec, target=target
    )
    assert history[-1] <= history[0]
    p = tmp_path / "joint.safetensors"
    mx.save_safetensors(str(p), dict(gq=a[0], gs=a[1], uq=b[0], us=b[1]))
    loaded = mx.load(str(p))
    out = nn.silu(x @ unpack(loaded["gq"], loaded["gs"], **spec).T) * (
        x @ unpack(loaded["uq"], loaded["us"], **spec).T
    )
    assert mx.mean((out - target) ** 2).item() == pytest.approx(history[-1], rel=1e-5)


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("batch", [3, 33, 129])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_register_prefill(partial, batch, dtype):
    mx.random.seed(25)
    spec = recipe("morph32-c" if partial else "morph32-3s", 16, "up_proj")
    q, s = fit(mx.random.normal((64, 1024)) * 0.02, **spec)
    x = mx.random.normal((batch, 1024)).astype(dtype)
    a = matmul(x, q, s, **spec, variant="registers")
    b = matmul(x, q, s, **spec, variant="original")
    np.testing.assert_array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)))


def test_invalid_recipes_and_budgets_rejected():
    from morph32.codec import header
    from morph32.optimization import refine

    with pytest.raises(ValueError):
        header(3, 5, 0.5)
    with pytest.raises(ValueError):
        header(3, 3, 0.5, coefficients=[1.0, 0.5])
    with pytest.raises(ValueError):
        header(3, 3, 0.5, coefficients=[1.0, 0.5, 0.0])
    with pytest.raises(ValueError):
        refine(
            mx.zeros((1, 32)),
            (mx.zeros((1, 1, 3), dtype=mx.uint32), mx.zeros((1, 1), dtype=mx.float16)),
            spec=recipe("morph32-3s", 0, "up_proj"),
            sweeps=2.5,
        )
    # Repeated powers cannot use bitwise OR: their contributions overlap.
    assert (
        " | "
        not in header(3, 3, 0.5, coefficients=[1.0, 0.25, 0.25]).split("inline float value")[1]
    )


@pytest.mark.parametrize("full_output", [False, True])
def test_joint_seed_selection_preserves_original_objective(full_output):
    import mlx.nn as nn

    from morph32.optimization import joint_seed_fit

    mx.random.seed(17)
    spec = recipe("morph32-3s", 0, "up_proj")
    g = mx.random.normal((32, 64)) * 0.08
    u = mx.random.normal((32, 64)) * 0.08
    x = mx.random.normal((32, 64))
    gp = fit(g, **spec)
    up = fit(u, **spec)
    down = mx.random.normal((16, 32)) if full_output else None
    target = nn.silu(x @ g.T) * (x @ u.T)
    if full_output:
        target = target @ down.T
    a, b, history = joint_seed_fit(
        x, gp, up, g, u, gate_spec=spec, up_spec=spec, target=target, down=down
    )
    assert history[-1] <= history[0]
    pred = nn.silu(x @ unpack(*a, **spec).T) * (x @ unpack(*b, **spec).T)
    if full_output:
        pred = pred @ down.T
    assert mx.mean((pred - target) ** 2).item() == pytest.approx(history[-1], rel=1e-5)


@pytest.mark.parametrize("width", [8, 16, 32])
@pytest.mark.parametrize("partial", [False, True])
def test_packet_width_reference(width, partial):
    mx.random.seed(2)
    spec = recipe("morph32-c" if partial else "morph32-3s", 16, "up_proj")
    q, s = fit(mx.random.normal((64, 1024)) * 0.02, **spec)
    x = mx.random.normal((1, 1024))
    y = matvec_fast(x, q, s, **spec, values=width)
    expected = np.array(x) @ reference(q, s, **spec).T
    np.testing.assert_allclose(np.array(y), expected, rtol=2e-4, atol=2e-5)
    with pytest.raises(ValueError):
        matvec_fast(x, q, s, **spec, values=64)


@pytest.mark.parametrize("partial", [False, True])
def test_refinement_tiny_and_outlier_tiles(partial):
    mx.random.seed(33)
    w = mx.random.normal((4, 64)) * 1e-8
    w = w.at[1, 3].add(100.0)
    spec = recipe("morph32-c" if partial else "morph32-3s", 16, "up_proj")
    initial = fit(w, **spec)
    q, s = refine(w, initial, spec=spec, restarts=2)
    before = mx.sum((w - unpack(*initial, **spec)).reshape(4, 2, 32) ** 2, axis=-1)
    after = mx.sum((w - unpack(q, s, **spec)).reshape(4, 2, 32) ** 2, axis=-1)
    assert mx.all(mx.isfinite(s)).item()
    assert mx.all(after <= before + 1e-7).item()


@pytest.mark.parametrize("batch", [3, 129, 257])
def test_register_larger_tile(batch):
    mx.random.seed(25)
    spec = recipe("morph32-3s", 16, "up_proj")
    q, s = fit(mx.random.normal((64, 1024)) * 0.02, **spec)
    x = mx.random.normal((batch, 1024)).astype(mx.bfloat16)
    a = matmul(x, q, s, **spec, variant="registers", block_m=256)
    b = matmul(x, q, s, **spec, variant="original", block_m=128)
    np.testing.assert_array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)))


@pytest.mark.parametrize("partial_g,partial_u", [(False, False), (True, True), (False, True)])
def test_fused_gate_up_and_mlp(partial_g, partial_u, monkeypatch):
    from mlx_lm.models.qwen3_next import Qwen3NextMLP

    from morph32.fused import FusedMorphMLP
    from morph32.fused import matvec as dual
    from morph32.runtime import MorphLinear

    mx.random.seed(2)
    modules = []
    for partial in (partial_g, partial_u):
        spec = recipe("morph32-c" if partial else "morph32-3s", 16, "up_proj")
        q, s = fit(mx.random.normal((64, 1024)) * 0.02, **spec)
        modules.append(MorphLinear(q, s, spec))
    x = mx.random.normal((1, 1024)).astype(mx.bfloat16)
    a, b = dual(x, *modules)
    for y, module in zip((a, b), modules, strict=True):
        np.testing.assert_array_equal(
            np.array(y.astype(mx.float32)), np.array(module(x).astype(mx.float32))
        )
    inner = Qwen3NextMLP(1024, 64)
    inner.gate_proj, inner.up_proj = modules
    fused = FusedMorphMLP(inner)
    monkeypatch.setenv("MORPH32_FUSE_GATE_UP", "1")
    for batch in (1, 3):
        xx = mx.random.normal((batch, 1024)).astype(mx.bfloat16)
        np.testing.assert_array_equal(np.array(fused(xx)), np.array(inner(xx)))


def test_metadata_native_override(monkeypatch):
    from morph32.conversion import encode_metadata
    from morph32.metadata import CorrelatedLinear

    mx.random.seed(7)
    dense = (mx.random.normal((64, 1024)) * 0.04).astype(mx.bfloat16)
    weight, scales, biases = mx.quantize(dense, group_size=64, bits=4)
    arrays, spec = encode_metadata(scales, biases)
    assert spec["kind"] == "correlated_bf16_metadata"
    module = CorrelatedLinear(weight, **arrays, bits=spec["bits"], decode="packet")
    x = mx.random.normal((1, 1024)).astype(mx.bfloat16)
    monkeypatch.setenv("MORPH32_METADATA_VARIANT", "native")
    expected = mx.quantized_matmul(x, weight, scales, biases, transpose=True, group_size=64, bits=4)
    np.testing.assert_array_equal(
        np.array(module(x).astype(mx.float32)), np.array(expected.astype(mx.float32))
    )


@pytest.mark.parametrize("partial", [False, True])
def test_independent_planes_with_strong_search(partial):
    from morph32.optimization import fit_optimized

    mx.random.seed(19)
    spec = dict(recipe("morph32-c" if partial else "morph32-3s", 16, "up_proj"), planes=3)
    w = mx.random.normal((64, 1024)) * 0.02
    q, s = fit_optimized(w, spec=spec, sweeps=2, restarts=3, seed=2026)
    dense = reference(q, s, **spec)
    original = unpack(*fit(w, **spec), **spec)
    before = mx.sum((w - original).reshape(64, -1, 32) ** 2, axis=-1)
    after = mx.sum((w - mx.array(dense)).reshape(64, -1, 32) ** 2, axis=-1)
    assert mx.all(after <= before + 1e-7).item()
    np.testing.assert_array_equal(np.array(unpack(q, s, **spec)), dense)
    for batch in (1, 33):
        x = mx.random.normal((batch, 1024)).astype(mx.bfloat16)
        y = matvec_fast(x, q, s, **spec, values=32) if batch == 1 else matmul(x, q, s, **spec)
        weights = mx.array(dense) if batch == 1 else mx.array(dense).astype(x.dtype)
        expected = x.astype(mx.float32) @ weights.astype(mx.float32).T
        np.testing.assert_allclose(
            np.array(y.astype(mx.float32)), np.array(expected), atol=0.02, rtol=0.02
        )

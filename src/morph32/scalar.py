"""Experimental affine 3-bit control: 192 code bits + two FP16 values / 64 weights.

Uses the same BF16 targets and activation importance as MORPH. Direct Metal decode
never allocates a complete dense projection. The simple fused kernel is not an
optimized external quantizer and must be labelled as such in speed comparisons.
"""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def fit(weight, importance, sweeps=2, starts=3):
    n, k = weight.shape
    if k % 64:
        raise ValueError("Affine control requires groups of 64")
    w = weight.astype(mx.float32).reshape(n, k // 64, 64)
    a = importance.reshape(1, k // 64, 64)
    lo, hi = mx.min(w, axis=-1), mx.max(w, axis=-1)
    best_loss = mx.full(lo.shape, float("inf"))
    best_q = mx.zeros(w.shape, dtype=mx.uint32)
    best_s, best_b = mx.ones(lo.shape), mx.zeros(lo.shape)
    for start in range(starts):
        span = (hi - lo) * (0.7 + 0.15 * start)
        s = mx.maximum(span / 7, 2**-24).astype(mx.float16).astype(mx.float32)
        b = ((hi + lo - span) / 2).astype(mx.float16).astype(mx.float32)
        for _ in range(sweeps):
            q = mx.clip(mx.round((w - b[..., None]) / s[..., None]), 0, 7)
            total = mx.sum(a, axis=-1)
            aq = mx.sum(a * q, axis=-1)
            aw = mx.sum(a * w, axis=-1)
            aqq = mx.sum(a * q * q, axis=-1)
            aqw = mx.sum(a * q * w, axis=-1)
            denominator = total * aqq - aq * aq
            candidate = (total * aqw - aq * aw) / mx.maximum(denominator, 1e-20)
            s = mx.where(denominator > 1e-20, mx.maximum(candidate, 2**-24), s)
            s = s.astype(mx.float16).astype(mx.float32)
            b = ((aw - s * aq) / mx.maximum(total, 1e-20)).astype(mx.float16).astype(mx.float32)
        q = mx.clip(mx.round((w - b[..., None]) / s[..., None]), 0, 7).astype(mx.uint32)
        loss = mx.sum(a * (w - (q.astype(mx.float32) * s[..., None] + b[..., None])) ** 2, axis=-1)
        better = loss < best_loss
        best_q = mx.where(better[..., None], q, best_q)
        best_s, best_b = mx.where(better, s, best_s), mx.where(better, b, best_b)
        best_loss = mx.minimum(best_loss, loss)
    # Six words per group; each 32-weight half has three independent bitplanes.
    q = best_q.reshape(n, k // 32, 32)
    shifts = mx.arange(32, dtype=mx.uint32)
    words = mx.stack(
        [mx.sum(((q >> bit) & 1) << shifts, axis=-1).astype(mx.uint32) for bit in range(3)], axis=-1
    )
    return words, best_s.astype(mx.float16), best_b.astype(mx.float16)


def reference(words, scales, biases):
    q = np.asarray(words)
    shift = np.arange(32, dtype=np.uint32)
    codes = sum(((q[..., i, None] >> shift) & 1) << i for i in range(3))
    codes = codes.reshape(q.shape[0], -1, 64).astype(np.float32)
    return (codes * np.asarray(scales)[..., None] + np.asarray(biases)[..., None]).reshape(
        q.shape[0], -1
    )


@lru_cache(None)
def kernel():
    return mx.fast.metal_kernel(
        name="affine3_direct",
        input_names=["x", "words", "scales", "biases"],
        output_names=["y"],
        source="""
        uint lane=thread_position_in_threadgroup.x%32;
        uint out=thread_position_in_grid.x/32;
        if(out>=M*N)return;
        uint row=out%N,batch=out/N;float total=0;
        for(uint tile=0;tile<K/32;tile++){
            uint address=(row*(K/32)+tile)*3;
            uint code=((words[address]>>lane)&1u)|(((words[address+1]>>lane)&1u)<<1)
                     |(((words[address+2]>>lane)&1u)<<2);
            uint group=row*(K/64)+tile/2;
            float weight=float(scales[group])*float(code)+float(biases[group]);
            total+=float(x[batch*K+tile*32+lane])*weight;
        }
        total=simd_sum(total);if(lane==0)y[out]=T(total);
        """,
    )


class Affine3Linear(nn.Module):
    def __init__(self, words, scales, biases, bias=None):
        super().__init__()
        if (
            words.dtype != mx.uint32
            or words.ndim != 3
            or words.shape[-1] != 3
            or words.shape[1] % 2
            or scales.shape != (words.shape[0], words.shape[1] // 2)
            or scales.dtype != mx.float16
            or biases.dtype != mx.float16
            or biases.shape != scales.shape
        ):
            raise ValueError("Invalid affine3 tensors")
        self.words, self.scales, self.biases = words, scales, biases
        if bias is not None:
            self.bias = bias

    def __call__(self, x):
        n, k = self.words.shape[0], self.words.shape[1] * 32
        if x.shape[-1] != k:
            raise ValueError("Affine3 input shape mismatch")
        m = x.size // k
        y = kernel()(
            inputs=[x, self.words, self.scales, self.biases],
            template=[("M", m), ("N", n), ("K", k), ("T", x.dtype)],
            grid=(m * n * 32, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(*x.shape[:-1], n)],
            output_dtypes=[x.dtype],
        )[0]
        return y + self.bias if "bias" in self else y

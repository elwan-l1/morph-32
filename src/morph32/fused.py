"""Opt-in shared-activation gate/up matvec for the supported Qwen SwiGLU MLP."""

import json
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.qwen3_next import Qwen3NextMLP

from .codec import header, load_words


@lru_cache(None)
def kernel(gate_json, up_json, values):
    g = json.loads(gate_json)
    u = json.loads(up_json)
    gh = header(**g).replace("repair_seed(", "gate_repair_seed(").replace("value(", "gate_value(")
    uh = header(**u).replace("repair_seed(", "up_repair_seed(").replace("value(", "up_value(")
    gl = (
        load_words(g["seeds"], g.get("partial", False), "tile")
        .replace("words[", "gwords[")
        .replace("p[", "gp[")
    )
    ul = (
        load_words(u["seeds"], u.get("partial", False), "tile")
        .replace("words[", "uwords[")
        .replace("p[", "up[")
    )
    return mx.fast.metal_kernel(
        name="morph_gate_up",
        input_names=["x", "gwords", "gscales", "uwords", "uscales"],
        output_names=["gy", "uy"],
        header=gh + uh,
        source=f"""
        uint lane=thread_position_in_threadgroup.x%32,row=thread_position_in_grid.x/32;
        if(row>=N)return;
        float gt=0,ut=0;
        for(uint k=lane*{values};k<K;k+=32*{values}){{
            uint tile=row*(K/32)+k/32,gp[{g["planes"]}],up[{u["planes"]}];
            {{ {gl} }} {{ {ul} }} gate_repair_seed(gp);up_repair_seed(up);
            float gs=0,us=0;
            for(uint j=0;j<{values};j++){{
                float a=float(x[k+j]);
                gs+=a*gate_value(gp,(k+j)%32);us+=a*up_value(up,(k+j)%32);
            }}
            gt+=float(gscales[tile])*gs;ut+=float(uscales[tile])*us;
        }}
        gt=simd_sum(gt);ut=simd_sum(ut);
        if(lane==0){{gy[row]=T(gt);uy[row]=T(ut);}}
    """,
    )


def matvec(x, gate, up, *, values=32):
    if (
        gate.words.shape[:2] != up.words.shape[:2]
        or x.size != x.shape[-1]
        or x.shape[-1] != gate.words.shape[1] * 32
        or values not in (1, 2, 4, 8, 16, 32)
        or x.shape[-1] % (32 * values)
    ):
        raise ValueError("Incompatible fused gate/up dimensions or packet width")
    n = gate.words.shape[0]
    k = x.shape[-1]
    out = kernel(
        json.dumps(gate._spec, sort_keys=True), json.dumps(up._spec, sort_keys=True), values
    )(
        inputs=[x, gate.words, gate.scales, up.words, up.scales],
        template=[("N", n), ("K", k), ("T", x.dtype)],
        grid=(n * 32, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(*x.shape[:-1], n)] * 2,
        output_dtypes=[x.dtype] * 2,
    )
    return tuple(
        y + module.bias if "bias" in module else y
        for y, module in zip(out, (gate, up), strict=True)
    )


class FusedMorphMLP(nn.Module):
    def __init__(self, inner):
        super().__init__()
        if type(inner) is not Qwen3NextMLP:
            raise ValueError("Fusion is validated only for the Qwen3Next SwiGLU MLP")
        self.gate_proj = inner.gate_proj
        self.up_proj = inner.up_proj
        self.down_proj = inner.down_proj

    def __call__(self, x):
        import os

        if x.size == x.shape[-1] and os.environ.get("MORPH32_FUSE_GATE_UP", "0") == "1":
            g, u = matvec(
                x,
                self.gate_proj,
                self.up_proj,
                values=int(os.environ.get("MORPH32_DECODE_VALUES", "32")),
            )
        else:
            g, u = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(swiglu(g, u))

"""Seed refinement for conversion defaults and additional opt-in fitting experiments."""

from functools import lru_cache

import mlx.core as mx
import numpy as np

from .codec import fit, header, load_words, unpack


def second_moments(x):
    """Uncentered per-input-tile second moments, shared across output rows."""
    x = x.reshape(-1, x.shape[-1]).astype(mx.float32)
    if x.shape[0] == 0 or x.shape[1] % 32:
        raise ValueError("Require nonempty activations with K divisible by 32")
    tiles = x.reshape(x.shape[0], -1, 32).transpose(1, 0, 2)
    return mx.matmul(tiles.transpose(0, 2, 1), tiles, stream=mx.cpu) / x.shape[0]


@lru_cache(None)
def refinement_kernel(
    seeds,
    planes,
    decay,
    partial,
    coefficients,
    sweeps,
    restarts,
    seed,
    covariance,
):
    load = load_words(seeds, partial, "tile")
    # One SIMD group evaluates one tile. H row resides in each lane's private storage.
    hload = (
        "float h[32]; for(uint j=0;j<32;j++)h[j]=moments[((tile%(K/32))*32+lane)*32+j];"
        if covariance
        else ""
    )

    def product(value):
        return (
            f"float hv=0;for(uint j=0;j<32;j++)hv+=h[j]*simd_shuffle({value},ushort(j));"
            if covariance
            else f"float hv=imp*{value};"
        )

    return mx.fast.metal_kernel(
        name="morph_refine",
        input_names=["weight", "importance", "moments", "words", "scales"],
        output_names=["out_words", "out_scales"],
        header=header(seeds, planes, decay, partial, coefficients),
        source=f"""
        uint lane=thread_position_in_threadgroup.x%32,tile=thread_position_in_grid.x/32;
        if(tile>=TILES)return;
        float target=float(weight[tile*32+lane]),imp=importance[(tile*32+lane)%K];
        {hload}
        uint p[{planes}];{load}repair_seed(p);
        float scale=float(scales[tile]),err=target-scale*value(p,lane);
        {product("err")}
        float best_loss=simd_sum(err*hv),best_scale=scale;
        uint best[{seeds}];for(uint s=0;s<{seeds};s++)best[s]=p[s];
        for(uint start=0;start<={restarts};start++){{
            for(uint s=0;s<{seeds};s++)p[s]=best[s];scale=best_scale;
            if(start>0){{
                uint rng=uint({seed})^((tile+1)*747796405u)^((start+1)*2891336453u);
                for(uint change=0;change<1+start%4;change++){{
                    rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
                    uint s=rng%{seeds},b=(rng>>8)%((s==2&&{"true" if partial else "false"})?16:32);
                    p[s]^=1u<<b;
                }}
            }}repair_seed(p);
            for(uint sweep=0;sweep<{sweeps};sweep++){{
                bool changed=false;
                for(uint s=0;s<{seeds};s++)for(uint b=0;b<((s==2&&{"true" if partial else "false"})?16:32);b++){{
                    uint trial[{planes}];for(uint i=0;i<{seeds};i++)trial[i]=p[i];
                    trial[s]^=1u<<b;repair_seed(trial);
                    float before=target-scale*value(p,lane),after=target-scale*value(trial,lane);
                    float delta=after-before,sumerr=after+before;
                    {product("delta")}
                    if(simd_sum(sumerr*hv)<0){{
                        for(uint i=0;i<{planes};i++)p[i]=trial[i];changed=true;
                    }}
                }}
                float v=value(p,lane);{product("v")}
                float new_scale=float(half(clamp(simd_sum(target*hv)/max(simd_sum(v*hv),1e-20f),0.0f,65504.0f)));
                float e0=target-scale*v,e1=target-new_scale*v,delta=e1-e0,sumerr=e1+e0;
                {{ {product("delta")}
                   if(simd_sum(sumerr*hv)<0){{changed=true;scale=new_scale;}}
                }}
                float err=target-scale*v;
                {{ {product("err")}
                   float loss=simd_sum(err*hv);
                   if(loss<best_loss){{best_loss=loss;best_scale=scale;
                       for(uint i=0;i<{seeds};i++)best[i]=p[i];}}
                }}
                if(!changed)break;
            }}
        }}
        if(lane==0){{for(uint s=0;s<{seeds};s++)out_words[tile*{seeds}+s]=best[s];out_scales[tile]=half(best_scale);}}
        """,
    )


def refine(weight, initial, *, spec, importance=None, moments=None, sweeps=2, restarts=0, seed=0):
    """Improve an existing packed candidate; retain it when no stored-scale gain exists."""
    n, k = weight.shape
    if (
        any(type(v) is not int for v in (sweeps, restarts, seed))
        or sweeps < 1
        or restarts < 0
        or not 0 <= seed < 2**32
    ):
        raise ValueError("Invalid refinement budget or random seed")
    spec = dict(spec)
    if importance is None:
        importance = mx.ones(k)
    if importance.shape != (k,) or not mx.all(mx.isfinite(importance) & (importance >= 0)).item():
        raise ValueError("Invalid channel importance")
    if not mx.all(mx.isfinite(weight)).item():
        raise ValueError("Nonfinite weights")
    covariance = moments is not None
    if covariance:
        if moments.shape != (k // 32, 32, 32) or not mx.all(mx.isfinite(moments)).item():
            raise ValueError("Invalid tile second moments")
        if not mx.allclose(moments, moments.transpose(0, 2, 1), atol=1e-6).item():
            raise ValueError("Second moments must be symmetric")
    else:
        moments = mx.zeros(1)
    seeds = spec.get("seeds", 3)
    partial = spec.get("partial", False)
    q, scale = initial
    if q.shape != (n, k // 32, 5 if partial else seeds) or scale.shape != (n, k // 32):
        raise ValueError("Initial candidate shape mismatch")
    call = refinement_kernel(
        seeds,
        spec.get("planes", 3),
        spec.get("decay", 0.5),
        partial,
        tuple(spec["coefficients"]) if "coefficients" in spec else None,
        sweeps,
        restarts,
        seed,
        covariance,
    )
    q, scale = call(
        inputs=[
            mx.contiguous(weight),
            importance.astype(mx.float32),
            moments.astype(mx.float32),
            q,
            scale,
        ],
        template=[("TILES", n * k // 32), ("K", k)],
        grid=(n * k, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n, k // 32, seeds), (n, k // 32)],
        output_dtypes=[mx.uint32, mx.float16],
    )
    if partial:
        q = mx.stack(
            [
                q[..., 0] & 65535,
                q[..., 0] >> 16,
                q[..., 1] & 65535,
                q[..., 1] >> 16,
                q[..., 2] & 65535,
            ],
            axis=-1,
        ).astype(mx.uint16)
    return q, scale


def fit_optimized(weight, *, spec, importance=None, moments=None, sweeps=2, restarts=0, seed=0):
    initial = fit(weight, **spec, importance=importance)
    return refine(
        weight,
        initial,
        spec=spec,
        importance=importance,
        moments=moments,
        sweeps=sweeps,
        restarts=restarts,
        seed=seed,
    )


def allocate(candidates, budget_bytes):
    """Greedy measured-cost allocation, starting from lowest-loss candidates.

    Each projection maps to candidates with bytes/loss. This is a proposal, not an
    exact knapsack solution or an assumption that full-model losses are additive.
    """
    if not candidates or budget_bytes < 0:
        raise ValueError("Require candidates and a nonnegative payload budget")
    for rows in candidates.values():
        if not rows or any(
            r["bytes"] < 0 or not np.isfinite(r["loss"]) or r["loss"] < 0 for r in rows
        ):
            raise ValueError("Invalid allocation candidates")
    chosen = {
        name: min(rows, key=lambda r: (r["loss"], r["bytes"])) for name, rows in candidates.items()
    }
    while sum(r["bytes"] for r in chosen.values()) > budget_bytes:
        moves = []
        for name, rows in candidates.items():
            old = chosen[name]
            for row in rows:
                saved = old["bytes"] - row["bytes"]
                if saved > 0:
                    moves.append(((row["loss"] - old["loss"]) / saved, name, row))
        if not moves:
            raise ValueError("Budget below minimum available representation")
        _, name, row = min(moves, key=lambda m: (m[0], m[1]))
        chosen[name] = row
    return chosen


def joint_scale_fit(x, gate, up, *, gate_spec, up_spec, target, steps=8, rate=0.02, down=None):
    """Joint nonlinear gate/up fitting over exported FP16 tile scales.

    Discrete seed words remain fixed. This bounded first experiment optimizes an
    actual supported subset of the representation, without surrogate weights.
    Optional down is a fixed matrix defining the full-MLP target boundary.
    """
    import mlx.nn as nn

    if steps < 1 or rate <= 0:
        raise ValueError("Invalid joint fitting budget")
    gq, gs = gate
    uq, us = up
    gv = unpack(gq, mx.ones_like(gs), **gate_spec).reshape(*gs.shape, 32)
    uv = unpack(uq, mx.ones_like(us), **up_spec).reshape(*us.shape, 32)
    x = x.astype(mx.float32)
    target = target.astype(mx.float32)

    def loss(a, b):
        g = x @ (gv * a[..., None]).reshape(gs.shape[0], -1).T
        u = x @ (uv * b[..., None]).reshape(us.shape[0], -1).T
        out = nn.silu(g) * u
        if down is not None:
            out = out @ down.astype(mx.float32).T
        return mx.mean((out - target) ** 2)

    grad = mx.value_and_grad(loss, argnums=(0, 1))
    a = gs.astype(mx.float32)
    b = us.astype(mx.float32)
    history = [loss(a, b).item()]
    for _ in range(steps):
        _, (ga, gb) = grad(a, b)
        # Normalize a relative-scale step, then backtrack on the rounded candidate.
        da = ga * mx.maximum(a, 1e-8) ** 2
        db = gb * mx.maximum(b, 1e-8) ** 2
        norm = mx.maximum(
            mx.max(mx.abs(da) / mx.maximum(a, 1e-8)), mx.max(mx.abs(db) / mx.maximum(b, 1e-8))
        )
        improved = False
        for fraction in (1.0, 0.5, 0.25, 0.125):
            aa = (
                mx.clip(a - rate * fraction * da / mx.maximum(norm, 1e-20), 0, 65504)
                .astype(mx.float16)
                .astype(mx.float32)
            )
            bb = (
                mx.clip(b - rate * fraction * db / mx.maximum(norm, 1e-20), 0, 65504)
                .astype(mx.float16)
                .astype(mx.float32)
            )
            score = loss(aa, bb).item()
            if np.isfinite(score) and score < history[-1]:
                a, b = aa, bb
                history.append(score)
                improved = True
                break
        if not improved:
            break
    return (gq, a.astype(mx.float16)), (uq, b.astype(mx.float16)), history


def joint_seed_fit(
    x, gate, up, gate_weight, up_weight, *, gate_spec, up_spec, target, seed=2026, down=None
):
    """Select discrete seed proposals using the actual joint nonlinear objective.

    Proposals come from bounded weight refinement. Gate-only, up-only and joint
    replacements compete with the original. The product objective permits exact
    per-neuron choices; a fixed down matrix requires whole-block choices instead.
    """
    import mlx.nn as nn

    x = x.astype(mx.float32)
    importance = mx.mean(x * x, axis=0)
    importance /= mx.maximum(mx.mean(importance), 1e-20)
    proposed_g = refine(
        gate_weight, gate, spec=gate_spec, importance=importance, sweeps=2, restarts=1, seed=seed
    )
    proposed_u = refine(
        up_weight, up, spec=up_spec, importance=importance, sweeps=2, restarts=1, seed=seed + 1
    )

    def error(g, u):
        out = nn.silu(x @ unpack(*g, **gate_spec).T) * (x @ unpack(*u, **up_spec).T)
        if down is not None:
            return mx.mean((out @ down.T - target) ** 2)
        return mx.mean((out - target) ** 2, axis=0)

    best_g, best_u = gate, up
    best = error(best_g, best_u)
    history = [mx.mean(best).item()]
    for g, u in ((proposed_g, up), (gate, proposed_u), (proposed_g, proposed_u)):
        trial = error(g, u)
        if down is not None:
            if trial.item() < best.item():
                best_g, best_u, best = g, u, trial
        else:
            mask = trial < best
            best_g = (
                mx.where(mask[:, None, None], g[0], best_g[0]),
                mx.where(mask[:, None], g[1], best_g[1]),
            )
            best_u = (
                mx.where(mask[:, None, None], u[0], best_u[0]),
                mx.where(mask[:, None], u[1], best_u[1]),
            )
            best = mx.minimum(best, trial)
        mx.eval(best_g, best_u, best)
        history.append(mx.mean(best).item())
    return best_g, best_u, history

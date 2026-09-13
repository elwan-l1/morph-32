"""Three packed sign planes with a direct Metal decoder and greedy encoder."""

from functools import lru_cache

import mlx.core as mx
import numpy as np


def header(seeds, planes, decay, partial=False, coefficients=None):
    if seeds != 3 or planes != 3:
        raise ValueError("MORPH requires exactly three stored/reconstruction planes")
    if coefficients is not None:
        if len(coefficients) != planes or not all(np.isfinite(c) and c > 0 for c in coefficients):
            raise ValueError("Require one finite positive coefficient per plane")
    import math

    fast = ""
    if coefficients is None and decay == 0.5:
        divisor = 2 ** (planes - 1)
        integer_coefficients = [int(divisor * decay**i) for i in range(planes)]
        expression = " | ".join(
            f"(((p[{i}]>>lane)&1u)<<{int(math.log2(c))})"
            for i, c in enumerate(integer_coefficients)
        )
        fast = f"return (float({expression})*2.0f-{sum(integer_coefficients)}.0f)/{divisor}.0f;"
    if coefficients is not None:
        # General path; never apply the bit-assembly shortcut to arbitrary coefficients.
        fast = (
            "return "
            + "+".join(
                f"{float(c)}f*(float((p[{i}]>>lane)&1u)*2.0f-1.0f)"
                for i, c in enumerate(coefficients)
            )
            + ";"
        )
    if coefficients is not None:
        exponents = [-math.log2(c) for c in coefficients]
        if all(e == int(e) and 0 <= e <= 20 for e in exponents) and len(set(exponents)) == len(
            exponents
        ):
            divisor = 2 ** int(max(exponents))
            integer_coefficients = [int(divisor * c) for c in coefficients]
            expression = " | ".join(
                f"(((p[{i}]>>lane)&1u)<<{int(math.log2(c))})"
                for i, c in enumerate(integer_coefficients)
            )
            fast = f"return (float({expression})*2.0f-{sum(integer_coefficients)}.0f)/{divisor}.0f;"
    if coefficients is not None:
        # Small dyadic rationals can also use one integer sum and one conversion.
        # Addition is essential when coefficient bits overlap (e.g. 9/16, 9/32).
        denominator = max(float(c).as_integer_ratio()[1] for c in coefficients)
        if denominator <= 4096:
            integers = [int(c * denominator) for c in coefficients]
            if sum(integers) < 2**23 and not all(c > 0 and c & (c - 1) == 0 for c in integers):
                expression = " + ".join(
                    f"(((p[{i}]>>lane)&1u)*{c}u)" for i, c in enumerate(integers)
                )
                fast = f"return (float({expression})*2.0f-{sum(integers)}.0f)/{denominator}.0f;"
    repair = (
        "p[2]=(p[2]&65535u)|(((p[2]^p[0]^rotate(p[1],uint(5)))&65535u)<<16);" if partial else ""
    )
    return f"""
    inline void repair_seed(thread uint* p){{
        {repair}
    }}
    inline float value(thread uint* p,uint lane){{
        {fast}
        float result=0,amplitude=1;
        for(uint i=0;i<{planes};i++){{result+=amplitude*(float((p[i]>>lane)&1)*2-1);amplitude*={decay}f;}}
        return result;
    }}
    """


@lru_cache(None)
def encoder(
    seeds=3,
    planes=3,
    decay=0.5,
    sweeps=2,
    starts=3,
    partial=False,
    coefficients=None,
):
    init = ""
    if coefficients is not None:
        init = (
            "const float init_coeff[] = {"
            + ",".join(f"{float(c)}f" for c in coefficients[:seeds])
            + "};"
        )
    return mx.fast.metal_kernel(
        name=f"morph_fit_{seeds}_{planes}_{sweeps}_{starts}",
        input_names=["weight", "importance"],
        output_names=["words", "scales"],
        header=header(seeds, planes, decay, partial, coefficients),
        source=f"""
        uint lane=thread_position_in_threadgroup.x%32;
        uint tile=thread_position_in_grid.x/32;
        if(tile>=TILES)return;
        float target=float(weight[tile*32+lane]);
        float imp=importance[(tile*32+lane)%K];
        float maximum=simd_max(abs(target)),best_loss=INFINITY,best_scale=0;
        {init}
        uint best[{seeds}],p[{planes}];
        for(uint start=0;start<{starts};start++){{
            float factor=0.7f+0.15f*float(start);
            float scale=max(maximum*factor/{sum(coefficients[:seeds]) if coefficients is not None else (1 - decay**seeds) / (1 - decay)}f,1e-12f);
            float residual=target/scale,amplitude=1;
            for(uint s=0;s<{seeds};s++){{
                {"amplitude=init_coeff[s];" if coefficients is not None else ""}
                uint bit=residual>=0?1:0;
                p[s]=simd_sum(bit<<lane);
                residual-=amplitude*(float(bit)*2-1);amplitude*={decay}f;
            }}
            repair_seed(p);
            for(uint sweep=0;sweep<{sweeps};sweep++){{
                for(uint s=0;s<{seeds};s++){{
                    // Test each seed bit in sequence; lanes score affected weights together.
                    for(uint b=0;b<((s==2 && {"true" if partial else "false"})?16:32);b++){{
                        float before=value(p,lane);
                        uint trial[{planes}];for(uint i=0;i<{seeds};i++)trial[i]=p[i];
                        trial[s]^=1u<<b;repair_seed(trial);
                        float after=value(trial,lane);
                        float difference=imp*((target-scale*after)*(target-scale*after)-(target-scale*before)*(target-scale*before));
                        if(simd_sum(difference)<0)for(uint i=0;i<{planes};i++)p[i]=trial[i];
                    }}
                }}
                float v=value(p,lane);
                scale=max(simd_sum(imp*target*v)/max(simd_sum(imp*v*v),1e-20f),1e-12f);
                scale=float(half(scale));
            }}
            float error=target-scale*value(p,lane),loss=simd_sum(imp*error*error);
            if(loss<best_loss){{best_loss=loss;best_scale=scale;for(uint i=0;i<{seeds};i++)best[i]=p[i];}}
        }}
        if(lane==0){{for(uint i=0;i<{seeds};i++)words[tile*{seeds}+i]=best[i];scales[tile]=half(best_scale);}}
        """,
    )


def fit(
    weight,
    *,
    seeds=3,
    planes=3,
    decay=0.5,
    sweeps=2,
    starts=3,
    importance=None,
    partial=False,
    coefficients=None,
):
    n, k = weight.shape
    if k % 32:
        raise ValueError("MORPH requires K divisible by 32")
    if importance is None:
        importance = mx.ones(k, dtype=mx.float32)
    q, s = encoder(
        seeds,
        planes,
        decay,
        sweeps,
        starts,
        partial,
        None if coefficients is None else tuple(coefficients),
    )(
        inputs=[mx.contiguous(weight), importance],
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
    return q, s


@lru_cache(None)
def kernel(seeds, planes, decay, decode_only, partial=False, coefficients=None):
    load = load_words(seeds, partial, "row*(K/32)+tile")
    source = f"""
    uint lane=thread_position_in_threadgroup.x%32,row=thread_position_in_grid.x/32;
    if(row>=N)return;
    float total=0;
    for(uint tile=0;tile<K/32;tile++){{
        uint p[{planes}];{load}
        repair_seed(p);float w=float(scales[row*(K/32)+tile])*value(p,lane);
        """
    if decode_only:
        source += "dense[row*K+tile*32+lane]=T(w);"
    else:
        source += "total+=w*float(x[tile*32+lane]);"
    source += "}\n"
    if not decode_only:
        source += "total=simd_sum(total);if(lane==0)y[row]=T(total);"
    return mx.fast.metal_kernel(
        name=f"morph_{'unpack' if decode_only else 'matvec'}_{seeds}_{planes}",
        input_names=["words", "scales"] if decode_only else ["x", "words", "scales"],
        output_names=["dense"] if decode_only else ["y"],
        header=header(seeds, planes, decay, partial, coefficients),
        source=source,
    )


def unpack(
    q,
    s,
    *,
    seeds=3,
    planes=3,
    decay=0.5,
    dtype=mx.float32,
    partial=False,
    coefficients=None,
):
    n, tiles, _ = q.shape
    k = tiles * 32
    return kernel(
        seeds,
        planes,
        decay,
        True,
        partial,
        None if coefficients is None else tuple(coefficients),
    )(
        inputs=[q, s],
        template=[("N", n), ("K", k), ("T", dtype)],
        grid=(n * 32, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n, k)],
        output_dtypes=[dtype],
    )[0]


def matvec(
    x,
    q,
    s,
    *,
    seeds=3,
    planes=3,
    decay=0.5,
    partial=False,
    coefficients=None,
):
    n, tiles, _ = q.shape
    k = tiles * 32
    if x.size != k:
        raise ValueError("Direct matvec requires one token")
    return kernel(
        seeds,
        planes,
        decay,
        False,
        partial,
        None if coefficients is None else tuple(coefficients),
    )(
        inputs=[x, q, s],
        template=[("N", n), ("K", k), ("T", x.dtype)],
        grid=(n * 32, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[x.dtype],
    )[0]


def reference(
    q,
    s,
    *,
    seeds=3,
    planes=3,
    decay=0.5,
    partial=False,
    coefficients=None,
):
    header(seeds, planes, decay, partial, coefficients)
    q = np.array(q)
    if partial:
        q = q.astype(np.uint32)
        q = np.stack(
            [q[..., 0] | (q[..., 1] << 16), q[..., 2] | (q[..., 3] << 16), q[..., 4]], axis=-1
        )
        q[..., 2] |= (
            (q[..., 2] ^ q[..., 0] ^ ((q[..., 1] << 5) | (q[..., 1] >> 27))) & 65535
        ) << 16
    p = [q[..., i] for i in range(seeds)]
    values = sum(
        (
            (((word[..., None] >> np.arange(32, dtype=np.uint32)) & 1).astype(np.float32) * 2 - 1)
            * (coefficients[i] if coefficients is not None else decay**i)
        )
        for i, word in enumerate(p)
    )
    return (values * np.array(s)[..., None]).reshape(q.shape[0], -1)


@lru_cache(None)
def packet_kernel(
    seeds,
    planes,
    decay,
    values,
    partial=False,
    optimized=False,
    coefficients=None,
):
    load = load_words(seeds, partial, "tile")
    dot = f"for(uint j=0;j<{values};j++)sum+=float(x[k+j])*value(p,(k+j)%32);"
    if optimized:
        # Experimental: shorten the serial accumulation dependency; do not change default.
        if values % 2:
            raise ValueError("Split accumulator requires an even packet size")
        dot = f"""float odd=0;
            for(uint j=0;j<{values};j+=2){{
                sum+=float(x[k+j])*value(p,(k+j)%32);
                odd+=float(x[k+j+1])*value(p,(k+j+1)%32);
            }}sum+=odd;"""
    return mx.fast.metal_kernel(
        name="morph_packet",
        input_names=["x", "words", "scales"],
        output_names=["y"],
        header=header(seeds, planes, decay, partial, coefficients),
        source=f"""
        uint lane=thread_position_in_threadgroup.x%32,row=thread_position_in_grid.x/32;
        if(row>=N)return;float total=0;
        for(uint k=lane*{values};k<K;k+=32*{values}){{
            uint tile=row*(K/32)+k/32,p[{planes}];
            {load}repair_seed(p);
            float sum=0;
            {dot}
            total+=float(scales[tile])*sum;
        }}
        total=simd_sum(total);if(lane==0)y[row]=T(total);
        """,
    )


def matvec_fast(
    x,
    q,
    s,
    *,
    seeds=3,
    planes=3,
    decay=0.5,
    values=16,
    partial=False,
    optimized=False,
    coefficients=None,
):
    n, tiles, _ = q.shape
    k = tiles * 32
    if values not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Packet width must divide one 32-weight tile")
    if x.size != k or k % (32 * values):
        raise ValueError("Packet shape mismatch")
    return packet_kernel(
        seeds,
        planes,
        decay,
        values,
        partial,
        optimized,
        None if coefficients is None else tuple(coefficients),
    )(
        inputs=[x, q, s],
        template=[("N", n), ("K", k), ("T", x.dtype)],
        grid=(n * 32, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[x.dtype],
    )[0]


def load_words(seeds, partial, index):
    if partial:
        return f"uint address=({index})*5;p[0]=uint(words[address])|(uint(words[address+1])<<16);p[1]=uint(words[address+2])|(uint(words[address+3])<<16);p[2]=uint(words[address+4]);"
    return f"for(uint i=0;i<{seeds};i++)p[i]=words[({index})*{seeds}+i];"

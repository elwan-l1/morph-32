"""Fused bitplane reconstruction into M5 NAX threadgroup tiles; no expanded matrix."""

import os
import re
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from .codec import header, load_words


@lru_cache(None)
def nax_header():
    root = Path(mx.__file__).parent / "include"
    seen = set()

    def expand(relative):
        if relative in seen:
            return ""
        seen.add(relative)
        return re.sub(
            r'^#include "([^"]+)"',
            lambda m: expand(m.group(1)),
            (root / relative).read_text(),
            flags=re.M,
        )

    return expand("mlx/backend/metal/kernels/steel/gemm/nax.h")


@lru_cache(None)
def kernel(seeds, planes, decay, partial=False, coefficients=None):
    load = load_words(seeds, partial, "tile")
    return mx.fast.metal_kernel(
        name="morph_nax",
        input_names=["x", "words", "scales"],
        output_names=["y"],
        header=nax_header() + header(seeds, planes, decay, partial, coefficients),
        source=f"""
        using namespace mlx::steel;
        uint lid=thread_position_in_threadgroup.x,sg=lid/32,row0=threadgroup_position_in_grid.x*64;
        uint batch0=threadgroup_position_in_grid.y*BM,tm=(sg/2)*(BM/2),tn=(sg%2)*32;
        threadgroup T weights[64*72];NAXTile<float,BM/32,2> accum;accum.clear();
        for(uint start=0;start<K;start+=64){{
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint local_row=lid/2,local_k=(lid%2)*32,row=row0+local_row,k=start+local_k;
            uint tile=row*(K/32)+k/32,p[{planes}];
            {load}repair_seed(p);
            float scale=float(scales[tile]);
            for(uint j=0;j<32;j++)weights[local_row*72+local_k+j]=T(scale*value(p,j));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint kk=0;kk<64;kk+=32){{
                NAXTile<T,BM/32,2> a;NAXTile<T,2,2> b;
                if(batch0+tm+BM/2<=M)a.load(x+(batch0+tm)*K+start+kk,K);
                else a.load_safe(x+(batch0+tm)*K+start+kk,K,short2(32,max(0,int(M)-int(batch0+tm))));
                b.template load<T,72,1>(weights+tn*72+kk);
                tile_matmad_nax(accum,a,metal::bool_constant<false>{{}},b,metal::bool_constant<true>{{}});
            }}
        }}
        if(batch0+tm+BM/2<=M)accum.store(y+(batch0+tm)*N+row0+tn,N);
        else accum.store_safe(y+(batch0+tm)*N+row0+tn,N,short2(32,max(0,int(M)-int(batch0+tm))));
        """,
    )


@lru_cache(None)
def register_kernel(seeds, planes, decay, partial=False, coefficients=None):
    load = load_words(seeds, partial, "tile")
    return mx.fast.metal_kernel(
        name="morph_nax_registers",
        input_names=["x", "words", "scales"],
        output_names=["y"],
        header=nax_header() + header(seeds, planes, decay, partial, coefficients),
        source=f"""
        using namespace mlx::steel;
        uint lid=thread_position_in_threadgroup.x,sg=lid/32,row0=threadgroup_position_in_grid.x*64;
        uint batch0=threadgroup_position_in_grid.y*BM,tm=(sg/2)*(BM/2),tn=(sg%2)*32;
        NAXTile<float,BM/32,2> accum;accum.clear();
        short2 sc=BaseNAXFrag::get_coord();
        for(uint start=0;start<K;start+=32){{
            NAXTile<T,BM/32,2> a;NAXTile<T,2,2> b;
            if(batch0+tm+BM/2<=M)a.load(x+(batch0+tm)*K+start,K);
            else a.load_safe(x+(batch0+tm)*K+start,K,short2(32,max(0,int(M)-int(batch0+tm))));
            for(short r=0;r<2;r++)for(short i=0;i<2;i++){{
                uint row=row0+tn+r*16+sc.y+i*8,tile=row*(K/32)+start/32,p[{planes}];
                {load}repair_seed(p);float scale=float(scales[tile]);
                for(short c=0;c<2;c++)for(short j=0;j<4;j++)
                    b.frag_at(r,c)[i*4+j]=T(scale*value(p,c*16+sc.x+j));
            }}
            tile_matmad_nax(accum,a,metal::bool_constant<false>{{}},b,metal::bool_constant<true>{{}});
        }}
        if(batch0+tm+BM/2<=M)accum.store(y+(batch0+tm)*N+row0+tn,N);
        else accum.store_safe(y+(batch0+tm)*N+row0+tn,N,short2(32,max(0,int(M)-int(batch0+tm))));
        """,
    )


def matmul(
    x,
    q,
    s,
    *,
    seeds=3,
    planes=3,
    decay=0.5,
    block_m=None,
    partial=False,
    coefficients=None,
    variant=None,
):
    if block_m is None:
        block_m = int(os.environ.get("MORPH32_PREFILL_BLOCK_M", "128"))
    if block_m not in (64, 128, 256):
        raise ValueError("Prefill block size must be 64, 128, or 256")
    k = x.shape[-1]
    n = q.shape[0]
    m = x.size // k
    if k % 64 or n % 64:
        raise ValueError("NAX requires N/K multiples of 64")
    variant = variant or os.environ.get("MORPH32_PREFILL_VARIANT", "original")
    if variant not in ("original", "registers"):
        raise ValueError("Unknown MORPH32_PREFILL_VARIANT")
    selected = register_kernel if variant == "registers" else kernel
    return selected(
        seeds,
        planes,
        decay,
        partial,
        None if coefficients is None else tuple(coefficients),
    )(
        inputs=[x, q, s],
        template=[("T", x.dtype), ("K", k), ("N", n), ("M", m), ("BM", block_m)],
        grid=(n // 64 * 128, (m + block_m - 1) // block_m, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[x.dtype],
    )[0]

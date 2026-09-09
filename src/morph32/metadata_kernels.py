"""Lossless BF16 metadata reconstruction and native MLX products."""

from functools import lru_cache

import mlx.core as mx


@lru_cache(None)
def metadata_kernel():
    return mx.fast.metal_kernel(
        name="morph_metadata_direct",
        input_names=["x", "q", "metadata", "bases"],
        output_names=["y"],
        source="""
        uint lane=thread_position_in_grid.x;
        uint row=thread_position_in_grid.y;
        uint batch=thread_position_in_grid.z;
        float sum=0.0f;
        for(uint k=lane*8;k<K;k+=256){
            uint group=row*(K/64)+k/64;
            uint bit=group*(2*B+2);
            uint idx=bit/32, shift=bit%32;
            ulong pair=(ulong(metadata[idx+1])<<32)|ulong(metadata[idx]);
            uint v=uint(pair>>shift);
            uint sb=(v&((1u<<B)-1))+bases[0];
            uint bb=((v>>B)&((1u<<B)-1))+bases[1];
            sb|=((v>>(2*B))&1)<<15;
            bb|=((v>>(2*B+1))&1)<<15;
            float s=as_type<float>(sb<<16), b=as_type<float>(bb<<16);
            uint word=q[row*(K/8)+k/8];
            for(uint j=0;j<8;j++){
                float w=float(T(float((word>>(4*j))&15)*s+b));
                sum+=float(x[batch*K+k+j])*w;
            }
        }
        sum=simd_sum(sum);
        if(lane==0)y[batch*N+row]=T(sum);
        """,
    )


def metadata_matmul(x, q, metadata, bases, bits):
    k = x.shape[-1]
    n = q.shape[0]
    m = x.size // k
    return metadata_kernel()(
        inputs=[x, q, metadata, bases],
        template=[("T", x.dtype), ("K", k), ("N", n), ("B", bits)],
        grid=(32, n, m),
        threadgroup=(32, 4, 1),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[x.dtype],
    )[0]


@lru_cache(None)
def metadata_unpack_kernel():
    return mx.fast.metal_kernel(
        name="morph_unpack_metadata",
        input_names=["metadata", "bases"],
        output_names=["scales", "biases"],
        source="""
        uint group=thread_position_in_grid.x;
        if(group>=COUNT)return;
        uint bit=group*(2*B+2),idx=bit/32,shift=bit%32;
        ulong pair=(ulong(metadata[idx+1])<<32)|ulong(metadata[idx]);
        uint field=uint(pair>>shift);
        uint sb=(field&((1u<<B)-1))+bases[0];
        uint bb=((field>>B)&((1u<<B)-1))+bases[1];
        sb|=((field>>(2*B))&1)<<15;
        bb|=((field>>(2*B+1))&1)<<15;
        scales[group]=bfloat16_t(as_type<float>(sb<<16));
        biases[group]=bfloat16_t(as_type<float>(bb<<16));
        """,
    )


def unpack_metadata(metadata, bases, bits, shape):
    count = shape[0] * shape[1]
    return metadata_unpack_kernel()(
        inputs=[metadata, bases],
        template=[("B", bits), ("COUNT", count)],
        grid=(count, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape, shape],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )


def metadata_native_matmul(x, q, metadata, bases, bits):
    s, b = unpack_metadata(metadata, bases, bits, (q.shape[0], x.shape[-1] // 64))
    return mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=4)


@lru_cache(None)
def correlated_unpack_kernel():
    return mx.fast.metal_kernel(
        name="morph_correlated_metadata",
        input_names=["fields", "bases", "index", "residual"],
        output_names=["scales", "biases"],
        source="""
        uint group=thread_position_in_grid.x;
        if(group>=COUNT)return;
        uint bit=group*(B+1),idx=bit/32,shift=bit%32;
        ulong pair=(ulong(fields[idx+1])<<32)|ulong(fields[idx]);
        uint field=uint(pair>>shift);
        uint mag=(field&((1u<<B)-1))+bases[0];
        uint sign=(field>>B)&1;
        uint chunk=group/32,lane=group%32;
        uint mask=index[chunk*2];
        int delta=0;
        if((mask>>lane)&1){
            uint rank=popcount(mask&((1u<<lane)-1));
            delta=int(residual[index[chunk*2+1]+rank])+as_type<int>(bases[1]);
        }
        uint sb=mag|(sign<<15);
        uint bb=uint(int(mag)+384+delta)|((sign^1)<<15);
        scales[group]=bfloat16_t(as_type<float>(sb<<16));
        biases[group]=bfloat16_t(as_type<float>(bb<<16));
        """,
    )


def unpack_correlated(fields, bases, index, residual, bits, shape):
    count = shape[0] * shape[1]
    return correlated_unpack_kernel()(
        inputs=[fields, bases, index, residual],
        template=[("B", bits), ("COUNT", count)],
        grid=(count, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape, shape],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )


def correlated_native_matmul(x, q, fields, bases, index, residual, bits):
    s, b = unpack_correlated(fields, bases, index, residual, bits, (q.shape[0], x.shape[-1] // 64))
    return mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=4)

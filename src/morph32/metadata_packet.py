"""Single-token native int4 products with compressed BF16 metadata."""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx


@lru_cache(maxsize=1)
def kernel():
    return mx.fast.metal_kernel(
        name="morph_metadata_packet",
        input_names=["x", "q", "fields", "bases", "index", "residual"],
        output_names=["y"],
        source=(Path(__file__).parent / "metal/metadata_packet.metal").read_text(),
    )


def matmul(x, q, fields, bases, index, residual, *, bits):
    k, n, m = x.shape[-1], q.shape[0], x.size // x.shape[-1]
    if k % 512:
        raise ValueError("Metadata packet requires complete 512-weight SIMD blocks")
    return kernel()(
        inputs=[x, q, fields, bases, index, residual],
        template=[
            ("T", x.dtype),
            ("K", k),
            ("N", n),
            ("B", bits),
            ("VALUES", 16),
            ("ROWS", 4),
            ("G", 64),
            ("WALSH", False),
            ("NATIVE_SUM", True),
            ("ROW_GAIN", False),
            ("SYNDROME", False),
        ],
        grid=(32, (n + 3) // 4, m),
        threadgroup=(32, 2, 1),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[x.dtype],
    )[0]

"""Exact bit packing for BF16 quantization metadata."""

import mlx.core as mx
import numpy as np


def pack_fields(values, bits):
    values = np.asarray(values, dtype=np.uint32).ravel()
    if not 1 <= bits <= 32 or (bits < 32 and np.any(values >= (1 << bits))):
        raise ValueError("Field exceeds declared bit width")
    positions = np.arange(values.size, dtype=np.uint64) * bits
    words = np.zeros((values.size * bits + 31) // 32 + 1, dtype=np.uint32)
    index = (positions // 32).astype(np.int64)
    shift = (positions % 32).astype(np.uint32)
    np.bitwise_or.at(words, index, values << shift)
    overflow = shift + bits > 32
    np.bitwise_or.at(words, index[overflow] + 1, values[overflow] >> (32 - shift[overflow]))
    return mx.array(words)


def unpack_fields(words, bits, count):
    words = np.asarray(words, dtype=np.uint32)
    p = np.arange(count, dtype=np.uint64) * bits
    i = (p // 32).astype(np.int64)
    s = p % 32
    pair = words[i].astype(np.uint64) | (words[i + 1].astype(np.uint64) << 32)
    return ((pair >> s) & ((1 << bits) - 1)).astype(np.uint32)


def pack_metadata(scales, biases):
    if scales.dtype != mx.bfloat16 or biases.dtype != mx.bfloat16:
        raise ValueError("BF16 metadata required")
    s = np.array(scales.view(mx.uint16)).astype(np.uint32).ravel()
    b = np.array(biases.view(mx.uint16)).astype(np.uint32).ravel()
    sm = s & 32767
    bm = b & 32767
    base_s = int(sm.min())
    base_b = int(bm.min())
    bits = max(int(sm.max() - base_s).bit_length(), int(bm.max() - base_b).bit_length(), 1)
    if 2 * bits + 2 > 32:
        raise ValueError("Metadata is not compressible in a 32-bit field")
    value = (
        (sm - base_s)
        | ((bm - base_b) << bits)
        | ((s >> 15) << (2 * bits))
        | ((b >> 15) << (2 * bits + 1))
    )
    packed = pack_fields(value, 2 * bits + 2)
    bases = mx.array([base_s, base_b], dtype=mx.uint32)
    decoded = unpack_fields(np.array(packed), 2 * bits + 2, s.size)
    sr = (decoded & ((1 << bits) - 1)) + base_s | (((decoded >> (2 * bits)) & 1) << 15)
    br = ((decoded >> bits) & ((1 << bits) - 1)) + base_b | (
        ((decoded >> (2 * bits + 1)) & 1) << 15
    )
    assert np.array_equal(s, sr) and np.array_equal(b, br)
    return packed, bases, bits


def pack_correlated_metadata(scales, biases):
    """Exact BF16 scale fields plus sparse bias-minus-8*scale bit residuals."""
    if scales.dtype != mx.bfloat16 or biases.dtype != mx.bfloat16:
        raise ValueError("BF16 required")
    s = np.array(scales.view(mx.uint16)).astype(np.uint32).ravel()
    b = np.array(biases.view(mx.uint16)).astype(np.uint32).ravel()
    if not np.all((s >> 15) != (b >> 15)):
        raise ValueError("Scale and bias signs must be opposite")
    sm = s & 32767
    bm = b & 32767
    minimum = int(sm.min())
    bits = max(1, int(sm.max() - minimum).bit_length())
    fields = (sm - minimum) | ((s >> 15) << bits)
    delta = bm.astype(np.int32) - sm.astype(np.int32) - 384
    dmin = int(delta.min())
    dmax = int(delta.max())
    if dmax - dmin > 255:
        raise ValueError("Residual range exceeds byte")
    present = delta != 0
    padded = np.pad(present, (0, (-len(present)) % 32)).reshape(-1, 32)
    masks = np.sum(
        padded.astype(np.uint32) * (np.uint32(1) << np.arange(32, dtype=np.uint32)),
        axis=1,
        dtype=np.uint32,
    )
    counts = padded.sum(axis=1, dtype=np.uint32)
    offsets = np.cumsum(counts, dtype=np.uint32) - counts
    index = mx.array(np.stack([masks, offsets], axis=1))
    residual = mx.array((delta[present] - dmin).astype(np.uint8))
    bases = mx.array([minimum, dmin & 0xFFFFFFFF], dtype=mx.uint32)
    return pack_fields(fields, bits + 1), bases, index, residual, bits

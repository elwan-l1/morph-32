"""Exact metadata reconstruction for retained native four-bit projections."""

import mlx.nn as nn

from .metadata_kernels import correlated_native_matmul, metadata_matmul, metadata_native_matmul


class MetadataLinear(nn.Module):
    def __init__(self, weight, metadata, bases, bits, bias=None, direct=False):
        super().__init__()
        self.weight = weight
        self.metadata = metadata
        self.bases = bases
        self._metadata_bits = bits
        self._direct = direct
        if bias is not None:
            self.bias = bias

    def __call__(self, x):
        operation = (
            metadata_matmul if self._direct and x.size == x.shape[-1] else metadata_native_matmul
        )
        y = operation(x, self.weight, self.metadata, self.bases, self._metadata_bits)
        return y + self.bias if "bias" in self else y


class CorrelatedLinear(nn.Module):
    def __init__(self, weight, fields, bases, index, residual, bits, bias=None, decode="native"):
        super().__init__()
        self.weight = weight
        self.fields = fields
        self.bases = bases
        self.index = index
        self.residual = residual
        self._metadata_bits = bits
        self._decode = decode
        if bias is not None:
            self.bias = bias

    def __call__(self, x):
        if (
            self._decode == "packet"
            and x.size == x.shape[-1]
            and x.shape[-1] % 512 == 0
            and self.weight.shape[0] % 8 == 0
        ):
            from .metadata_packet import matmul as packet_matmul

            y = packet_matmul(
                x,
                self.weight,
                self.fields,
                self.bases,
                self.index,
                self.residual,
                bits=self._metadata_bits,
            )
        else:
            y = correlated_native_matmul(
                x,
                self.weight,
                self.fields,
                self.bases,
                self.index,
                self.residual,
                self._metadata_bits,
            )
        return y + self.bias if "bias" in self else y

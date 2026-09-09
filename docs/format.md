# MORPH-32 format, version 1

## Representation

A projection has shape `[N, K]` and `K / 32` tiles per output row. Tile position follows the input dimension; there is no weight permutation. Seed bit `j` describes position `j` within the tile, least-significant bit first. Integer operations wrap at 32 bits.

morph32-3s stores `words: uint32[N, K/32, 3]` and `scales: float16[N, K/32]`. For words `p[0:3]`, generate `p[i] = p[i-2] XOR rotl(p[i-1],5) XOR rotl(p[i-1],13) XOR rotl(p[i-1],21)`, for `i=3,4`. Set `sign(p,j)=2*((p>>j)&1)-1` and sum the five signs with coefficients `(1, 1/2, 1/4, 1/16, 1/32)`. Multiply by the tile's FP16 scale.

morph32-c tiles store `words: uint16[N, K/32, 5]`, ordered `[A.low16, A.high16, B.low16, B.high16, C.low16]`. Reconstruct:

```text
A = words[0] | (words[1] << 16)
B = words[2] | (words[3] << 16)
L = words[4]
C = L | (((L XOR A XOR rotl(B, 5)) & 65535) << 16)
```

The same five-plane recurrence and coefficients then apply. The partial layout consumes 96 bits including its scale, versus 112 bits for morph32-3s. morph32-c uses partial tiles only for `gate_proj` and `up_proj` in zero-based layers 8–55. All `down_proj` matrices and the remaining gate/up matrices use morph32-3s. These fixed recipes are defined in `src/morph32/profiles.py`.

## Fitting

For target BF16 weights `w[j]`, minimize `sum_j importance[j] * (w[j] - scale*v[j])**2`. Importance is the mean squared native activation for each input channel, divided by its mean across channels. Gate and up projections share the same captured input. Only the first 128-token calibration chunk is used.

Initialize seed planes greedily from the residual at three clipping factors: 0.70, 0.85 and 1.00. For each start, perform two sweeps over stored seed bits. Accept a bit flip when its tile-wide weighted error decreases. After each sweep, refit the scale by weighted least squares and round to FP16. Keep the best start. Partial tiles search only the low 16 bits of the third seed. The generated planes are always recomputed after a flip.

The recurrence introduces coupling: generated planes are not free extra precision. Its benefit must be judged by actual activation and model quality, not the number of generated magnitude levels. This release does not claim a globally optimal encoder or an established novelty priority.

## Metal execution

Generation uses `matvec_fast(..., values=32)`: each of 32 SIMD lanes reconstructs a whole tile in registers, accumulates its dot product, and advances by 1,024 input weights. `simd_sum` combines partial dot products. The supported model dimensions are multiples of 1,024.

Prefill uses 128 threads per group and M5 NAX matrix operations. Four SIMD groups cooperate on a 64×64 reconstructed weight tile, stored with a padded stride of 72 in threadgroup memory. Reconstructed weights are rounded to the activation dtype before multiplication. The implementation supports partial token batches; `N` and `K` must be multiples of 64. No full reconstructed projection is materialized by either inference path.

The reference `unpack` function does materialize a matrix and exists for conversion checks and small tests. Calling it manually on a full projection changes memory behavior; the runtime does not use it.

## Retained tensors

All 192 language MLP projections are re-quantized from BF16. Other tensors retain the native checkpoint's values. Supported language projections additionally compress BF16 scale/bias metadata losslessly, using bit-packed BF16 magnitude offsets and correlated scale/bias bit patterns with sparse residuals. Unsupported metadata remains native. The encoder verifies exact reconstruction and accepts the metadata encoding only when it uses fewer bytes.

For the correlated representation, store each scale magnitude bit pattern relative to a tensor-wide minimum, plus its sign. The bias has the opposite sign. Its magnitude bits are predicted as `scale_magnitude_bits + 384` (the BF16 exponent shift corresponding to a factor of eight). Store nonzero prediction residuals as bytes, with one 32-bit presence mask and prefix offset per 32 groups. The fallback stores both magnitude offsets and signs directly. These are exact operations on BF16 bit patterns, not approximate floating-point predictions.

At inference, supported one-token retained projections use a fused metadata reconstruction/product kernel. Other shapes reconstruct their small scale/bias arrays and call native MLX quantized matrix multiplication. Their int4 codes remain packed. See `packing.py`, `metadata_kernels.py` and `metal/metadata_packet.metal` for the exact layouts and arithmetic.

## Container and accounting

`.morph` is a little-endian safetensors container. Its metadata key `morph_manifest` contains JSON with `format="morph32"`, `version=1`, `standalone=true`, descriptors, embedded asset names, logical parameter count and a SHA-256 for every tensor payload. Descriptors select the MORPH or retained-metadata module for each projection. Config, tokenizer and other original top-level non-weight files are stored as byte tensors under `__assets__.`.

Loading validates the format, hashes, inventory, architecture shapes and strict model tensor assignment. It extracts assets to a temporary cache directory, constructs the model and tokenizer, then removes those temporary files. No native or BF16 source directory is required.

`morph32 inspect` independently counts logical parameters from tensor shapes and descriptors. Physical bpw is `8 * file_bytes / logical_parameters`. It includes unconverted tensors, metadata, embedded assets and the entire container header. This differs from the 3.0/3.5 matrix rates. The resident parameter count describes model arrays after loading; allocator peak and process physical footprint are separate measurements.

Historical file sizes refer to the archived checkpoints. Newly converted files may have different header sizes because provenance fields changed; exact physical bytes are always reported for the file actually produced. A matching matrix layout alone is not evidence of bit-identical model output.

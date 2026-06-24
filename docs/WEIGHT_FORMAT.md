# .vxcpm Binary Weight Format — Specification v1

> **Version**: 1
> **Extension**: `.vxcpm`
> **Endianness**: Little-endian (all multi-byte values)

---

## 1. File Layout

```
┌─────────────────────────────────────────┐
│             File Header (64 bytes)       │
├─────────────────────────────────────────┤
│         Tensor Metadata Array            │
│  (num_tensors × 32 bytes per entry)     │
├─────────────────────────────────────────┤
│              Tensor Data                 │
│  (raw float32/bfloat16 arrays,          │
│   concatenated in metadata order)       │
└─────────────────────────────────────────┘
```

---

## 2. File Header (64 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | `magic` | Magic number: `0x56435850` = `"VXCP"` |
| 4 | 4 | `version` | Format version (currently `1`) |
| 8 | 4 | `num_tensors` | Number of tensors in file |
| 12 | 4 | `reserved[0]` | Reserved (must be 0) |
| 16 | 4 | `reserved[1]` | Reserved (must be 0) |
| 20 | 4 | `reserved[2]` | Reserved (must be 0) |
| 24 | 4 | `reserved[3]` | Reserved (must be 0) |
| 28 | 4 | `reserved[4]` | Reserved (must be 0) |
| 32 | 4 | `reserved[5]` | Reserved (must be 0) |
| 36 | 4 | `reserved[6]` | Reserved (must be 0) |
| 40 | 4 | `reserved[7]` | Reserved (must be 0) |
| 44 | 4 | `reserved[8]` | Reserved (must be 0) |
| 48 | 4 | `reserved[9]` | Reserved (must be 0) |
| 52 | 4 | `reserved[10]` | Reserved (must be 0) |
| 56 | 4 | `checksum` | CRC32C of bytes 0–55 (header self-checksum = 0 for calculation) |
| 60 | 4 | `data_checksum` | CRC32C of metadata + tensor data |

---

## 3. Tensor Metadata Entry (32 bytes each)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | `name_offset` | Byte offset into string table for weight name |
| 4 | 4 | `name_length` | Length of weight name (bytes, not including null) |
| 8 | 4 | `dtype` | Data type: 0=float32, 1=bfloat16, 2=float16, 3=q4_0, 4=q4_1 |
| 12 | 4 | `ndim` | Number of dimensions (1–4) |
| 16 | 16 | `shape` | Shape array: `[dim0, dim1, dim2, dim3]`, unused dims = 1 |
| 32 | — | *end* | |

The string table immediately follows the metadata array. Each string is null-terminated.
String table layout: `[name_1\0][name_2\0]...[name_N\0]`

### String Table Offset Calculation

```
string_table_offset = 64 + num_tensors * 32
```

Name offsets in metadata entries are relative to the start of the string table.

---

## 4. Weight Name Convention

Weight names match PyTorch `named_parameters()` keys from the original model,
with `model.` prefix removed and `audio_vae.` kept as-is (loaded from separate file).

### 4.1 Top-Level Weights

| Weight Name | Shape | Description |
|-------------|-------|-------------|
| `base_lm.embed_tokens.weight` | (73448, 2048) | Token embedding, BF16/FP32 |
| `base_lm.norm.weight` | (2048,) | Final RMS norm |
| `residual_lm.norm.weight` | (2048,) | Residual LM final RMS norm |
| `enc_to_lm_proj.weight` | (2048, 1024) | Encoder→LM projection |
| `enc_to_lm_proj.bias` | (2048,) | Bias |
| `lm_to_dit_proj.weight` | (1024, 2048) | LM→DiT projection |
| `lm_to_dit_proj.bias` | (1024,) | Bias |
| `res_to_dit_proj.weight` | (1024, 2048) | Residual→DiT projection |
| `res_to_dit_proj.bias` | (1024,) | Bias |
| `fusion_concat_proj.weight` | (2048, 4096) | Fusion concatenation projection |
| `fusion_concat_proj.bias` | (2048,) | Bias |
| `fsq_layer.in_proj.weight` | (64, 2048) | Scalar quantization input |
| `fsq_layer.in_proj.bias` | (64,) | Bias |
| `fsq_layer.out_proj.weight` | (2048, 64) | Scalar quantization output |
| `fsq_layer.out_proj.bias` | (2048,) | Bias |
| `stop_proj.weight` | (2048, 2048) | Stop predictor projection |
| `stop_proj.bias` | (2048,) | Bias |
| `stop_head.weight` | (2, 2048) | Stop classifier head (no bias) |
| `feat_encoder.special_token` | (1, 1, 1, 1024) | Local encoder special token |
| `feat_encoder.in_proj.weight` | (1024, 64) | Local encoder input projection |
| `feat_encoder.in_proj.bias` | (1024,) | Bias |
| `feat_encoder.encoder.norm.weight` | (1024,) | Encoder final RMS norm |

### 4.2 TSLM Layers (base_lm) — 28 layers (0–27)

Per layer `i` (0..27):

| Weight Name | Shape | Description |
|-------------|-------|-------------|
| `base_lm.layers.{i}.input_layernorm.weight` | (2048,) | Pre-attention RMS norm |
| `base_lm.layers.{i}.self_attn.q_proj.weight` | (2048, 2048) | Q projection (16 heads × 128) |
| `base_lm.layers.{i}.self_attn.k_proj.weight` | (256, 2048) | K projection (2 KV heads × 128) |
| `base_lm.layers.{i}.self_attn.v_proj.weight` | (256, 2048) | V projection (2 KV heads × 128) |
| `base_lm.layers.{i}.self_attn.o_proj.weight` | (2048, 2048) | Output projection |
| `base_lm.layers.{i}.mlp.gate_proj.weight` | (6144, 2048) | SwiGLU gate |
| `base_lm.layers.{i}.mlp.up_proj.weight` | (6144, 2048) | SwiGLU up |
| `base_lm.layers.{i}.mlp.down_proj.weight` | (2048, 6144) | SwiGLU down |
| `base_lm.layers.{i}.post_attention_layernorm.weight` | (2048,) | Post-attention RMS norm |

**Count per layer**: 9 tensors × 28 = 252 tensors

### 4.3 RALM Layers (residual_lm) — 8 layers (0–7)

Per layer `i` (0..7):

| Weight Name | Shape | Description |
|-------------|-------|-------------|
| `residual_lm.layers.{i}.input_layernorm.weight` | (2048,) | Pre-attention RMS norm |
| `residual_lm.layers.{i}.self_attn.q_proj.weight` | (2048, 2048) | Q projection |
| `residual_lm.layers.{i}.self_attn.k_proj.weight` | (256, 2048) | K projection |
| `residual_lm.layers.{i}.self_attn.v_proj.weight` | (256, 2048) | V projection |
| `residual_lm.layers.{i}.self_attn.o_proj.weight` | (2048, 2048) | Output projection |
| `residual_lm.layers.{i}.mlp.gate_proj.weight` | (6144, 2048) | SwiGLU gate |
| `residual_lm.layers.{i}.mlp.up_proj.weight` | (6144, 2048) | SwiGLU up |
| `residual_lm.layers.{i}.mlp.down_proj.weight` | (2048, 6144) | SwiGLU down |
| `residual_lm.layers.{i}.post_attention_layernorm.weight` | (2048,) | Post-attention RMS norm |

**Count per layer**: 9 tensors × 8 = 72 tensors

### 4.4 LocEnc Encoder Layers (feat_encoder.encoder) — 12 layers (0–11)

Per layer `i` (0..11):

| Weight Name | Shape | Description |
|-------------|-------|-------------|
| `feat_encoder.encoder.layers.{i}.input_layernorm.weight` | (1024,) | RMS norm |
| `feat_encoder.encoder.layers.{i}.self_attn.q_proj.weight` | (2048, 1024) | Q (16 heads × 128) |
| `feat_encoder.encoder.layers.{i}.self_attn.k_proj.weight` | (256, 1024) | K (2 KV heads) |
| `feat_encoder.encoder.layers.{i}.self_attn.v_proj.weight` | (256, 1024) | V (2 KV heads) |
| `feat_encoder.encoder.layers.{i}.self_attn.o_proj.weight` | (1024, 2048) | Output projection |
| `feat_encoder.encoder.layers.{i}.mlp.gate_proj.weight` | (4096, 1024) | SwiGLU gate |
| `feat_encoder.encoder.layers.{i}.mlp.up_proj.weight` | (4096, 1024) | SwiGLU up |
| `feat_encoder.encoder.layers.{i}.mlp.down_proj.weight` | (1024, 4096) | SwiGLU down |
| `feat_encoder.encoder.layers.{i}.post_attention_layernorm.weight` | (1024,) | RMS norm |

**Count per layer**: 9 tensors × 12 = 108 tensors

### 4.5 LocDiT Estimator Layers (feat_decoder.estimator.decoder) — 12 layers (0–11)

Same structure as LocEnc layers:

| Weight Name | Shape |
|-------------|-------|
| `feat_decoder.estimator.decoder.layers.{i}.input_layernorm.weight` | (1024,) |
| `feat_decoder.estimator.decoder.layers.{i}.self_attn.q_proj.weight` | (2048, 1024) |
| `feat_decoder.estimator.decoder.layers.{i}.self_attn.k_proj.weight` | (256, 1024) |
| `feat_decoder.estimator.decoder.layers.{i}.self_attn.v_proj.weight` | (256, 1024) |
| `feat_decoder.estimator.decoder.layers.{i}.self_attn.o_proj.weight` | (1024, 2048) |
| `feat_decoder.estimator.decoder.layers.{i}.mlp.gate_proj.weight` | (4096, 1024) |
| `feat_decoder.estimator.decoder.layers.{i}.mlp.up_proj.weight` | (4096, 1024) |
| `feat_decoder.estimator.decoder.layers.{i}.mlp.down_proj.weight` | (1024, 4096) |
| `feat_decoder.estimator.decoder.layers.{i}.post_attention_layernorm.weight` | (1024,) |

Plus DiT-specific:

| Weight Name | Shape | Description |
|-------------|-------|-------------|
| `feat_decoder.estimator.in_proj.weight` | (1024, 64) | DiT input projection |
| `feat_decoder.estimator.in_proj.bias` | (1024,) | |
| `feat_decoder.estimator.cond_proj.weight` | (1024, 64) | Condition projection |
| `feat_decoder.estimator.cond_proj.bias` | (1024,) | |
| `feat_decoder.estimator.out_proj.weight` | (64, 1024) | DiT output projection |
| `feat_decoder.estimator.out_proj.bias` | (64,) | |
| `feat_decoder.estimator.time_mlp.linear_1.weight` | (1024, 1024) | Time embedding MLP |
| `feat_decoder.estimator.time_mlp.linear_1.bias` | (1024,) | |
| `feat_decoder.estimator.time_mlp.linear_2.weight` | (1024, 1024) | |
| `feat_decoder.estimator.time_mlp.linear_2.bias` | (1024,) | |
| `feat_decoder.estimator.delta_time_mlp.linear_1.weight` | (1024, 1024) | Delta time MLP |
| `feat_decoder.estimator.delta_time_mlp.linear_1.bias` | (1024,) | |
| `feat_decoder.estimator.delta_time_mlp.linear_2.weight` | (1024, 1024) | |
| `feat_decoder.estimator.delta_time_mlp.linear_2.bias` | (1024,) | |
| `feat_decoder.estimator.decoder.norm.weight` | (1024,) | Decoder final RMS norm |

**Total DiT weights**: 12 × 9 + 16 = 124 tensors

### 4.6 AudioVAE V2 Weights (loaded from separate `audiovae.vxcpm`)

| Weight Name | Description |
|-------------|-------------|
| `audio_vae.encoder.block.0.weight` | First conv: (128, 1, 7) + bias |
| `audio_vae.encoder.block.N.weight` | Downsample blocks |
| `audio_vae.encoder.fc_mu.weight` | (64, d_model, 3) + bias |
| `audio_vae.encoder.fc_logvar.weight` | (64, d_model, 3) + bias |
| `audio_vae.decoder.model.*` | Transpose conv + residual blocks |
| `audio_vae.decoder.sr_cond_model.*` | Sample rate condition layers |

AudioVAE uses `weight_norm` wrappers, so actual weight names use `weight_v` and `weight_g` suffixes
(e.g. `audio_vae.encoder.block.0.weight_v`, `audio_vae.encoder.block.0.weight_g`).

---

## 5. Data Type Definitions

| dtype code | name | bytes per element | description |
|------------|------|-------------------|-------------|
| 0 | FP32 | 4 | IEEE 754 float32 |
| 1 | BF16 | 2 | Brain float 16 |
| 2 | FP16 | 2 | IEEE 754 float16 |
| 3 | Q4_0 | 0.5 | 4-bit quantization (block size 32) |
| 4 | Q4_1 | 0.5 | 4-bit quantization (block size 32, with min) |

### Q4_0 Block Format

Each block of 32 values:
- `fp16 d` (2 bytes): scale factor
- `uint8 q[16]` (16 bytes): 4-bit quantized values (packed, nibbles)

### Q4_1 Block Format

Each block of 32 values:
- `fp16 d` (2 bytes): scale factor
- `fp16 m` (2 bytes): min value
- `uint8 q[16]` (16 bytes): 4-bit quantized values

---

## 6. Tensor Order

Tensors must appear in metadata in the following group order for mmap-friendly loading:

1. `base_lm.embed_tokens.weight`
2. `base_lm.norm.weight`
3. `base_lm.layers.{0..27}.*` (all 9 per layer, in order)
4. `residual_lm.norm.weight`
5. `residual_lm.layers.{0..7}.*`
6. `enc_to_lm_proj.*`
7. `lm_to_dit_proj.*`
8. `res_to_dit_proj.*`
9. `fusion_concat_proj.*`
10. `fsq_layer.*`
11. `stop_proj.*` / `stop_head.*`
12. `feat_encoder.*`
13. `feat_encoder.encoder.layers.{0..11}.*`
14. `feat_decoder.estimator.*`
15. `feat_decoder.estimator.decoder.layers.{0..11}.*`

---

## 7. AudioVAE Separate File

The AudioVAE V2 weights are stored in a separate file `audiovae.vxcpm`
(same format, 64-byte header + metadata + data).

This separation allows:
- Independent updates to AudioVAE without changing the main model
- Smaller download for users who already have VAE weights
- Cleaner memory mapping

---

## 8. Total Weight Count

| Group | Tensors |
|-------|---------|
| base_lm (embed + norm + 28 layers × 9) | 1 + 1 + 252 = 254 |
| residual_lm (norm + 8 layers × 9) | 1 + 72 = 73 |
| enc_to_lm_proj | 2 |
| lm_to_dit_proj | 2 |
| res_to_dit_proj | 2 |
| fusion_concat_proj | 2 |
| fsq_layer | 4 |
| stop predictor | 3 |
| feat_encoder (special_token, in_proj, norm + 12 layers × 9) | 3 + 108 = 111 |
| feat_decoder.estimator (top-level 16 + 12 layers × 9) | 16 + 108 = 124 |
| **Total (main model)** | **~575** |
| audio_vae | ~100+ |

---

## 9. Memory Layout Notes

- All tensors are stored in **row-major** (C-order) contiguous layout.
- 1D tensors (norm weights, bias) are stored as flat arrays.
- The string table is **not** padded.
- All offsets in the header and metadata are relative to the start of the file.
- CRC32C uses the Castagnoli polynomial (0x1EDC6F41).

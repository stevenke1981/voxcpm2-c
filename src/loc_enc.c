// loc_enc.c — LocEnc forward pass
// VoxCPM2-C Project
// License: Apache-2.0
//
// Local Encoder (LocEnc): 12-layer transformer encoder.
//   input:  audio_feats [batch, time, patch, feat_dim]  (feat_dim=64)
//   output: features [batch, time, enc_hidden]          (enc_hidden=1024)
//
// Architecture:
//   1. in_proj: Linear(64 -> 1024) maps patches to hidden dim
//   2. special_token: learnable [CLS] token prepended per time step
//   3. encoder: MiniCPMModel(12 layers, hidden=1024, 16 heads, GQA=2)
//   4. Output: CLS token (first token) from each group = [batch, time, 1024]

#include "model.h"
#include "nn.h"
#include <string.h>

VoxCPMError loc_enc_forward(const LocEnc* enc, const Tensor* audio_feats, Tensor* out) {
    if (!enc || !audio_feats || !out) return VOXCPM_ERR_INTERNAL;

    // ─────────────────────────────────────────────────────────────
    // Input validation
    //   audio_feats: [batch, time, patch, feat_dim]
    //   out:         [batch, time, enc_hidden]
    // ─────────────────────────────────────────────────────────────
    if (audio_feats->ndim != 4) return VOXCPM_ERR_SHAPE_MISMATCH;
    int B = audio_feats->shape[0];
    int T = audio_feats->shape[1];
    int P = audio_feats->shape[2];
    int F = audio_feats->shape[3];
    int D = enc->d_model;

    if (out->ndim != 3) return VOXCPM_ERR_SHAPE_MISMATCH;
    if (out->shape[0] != B || out->shape[1] != T || out->shape[2] != D)
        return VOXCPM_ERR_SHAPE_MISMATCH;

    if (enc->n_layers < 1 || !enc->layers || !enc->layers[0].attn)
        return VOXCPM_ERR_INTERNAL;

    // ─────────────────────────────────────────────────────────────
    // Derived dimensions
    // ─────────────────────────────────────────────────────────────
    int n_kv_heads = enc->layers[0].attn->n_kv_heads;
    int head_dim   = enc->layers[0].attn->head_dim;
    int seq_total  = T * (P + 1);       /* total tokens after prepending CLS */
    int group_size = P + 1;             /* CLS + patches per time step */
    int flat_count = B * T * P;         /* total patches */

    // ─────────────────────────────────────────────────────────────
    // Temporary tensors (all freed on cleanup path)
    // ─────────────────────────────────────────────────────────────
    Tensor* x        = NULL;   /* [B, seq_total, D] — running hidden state */
    Tensor* x_flat   = NULL;   /* [B*T*P, F] — flat view of audio_feats (non-owning) */
    Tensor* proj     = NULL;   /* [B*T*P, D] — projected input (after bias) */
    Tensor* cache_k  = NULL;   /* [B, n_kv_heads, seq_total, head_dim] */
    Tensor* cache_v  = NULL;   /* [B, n_kv_heads, seq_total, head_dim] */

    VoxCPMError err = VOXCPM_SUCCESS;

    // ═════════════════════════════════════════════════════════════
    // Step 1: Input projection
    //   audio_feats [B, T, P, F] → flatten [B*T*P, F]
    //   → matmul_nt with in_proj_weight [D, F] → [B*T*P, D]
    //   → add bias [D] → [B*T*P, D]
    // ═════════════════════════════════════════════════════════════

    // Flat view of input (shares data, no copy)
    x_flat = tensor_create_from_buffer(2, (int[]){ flat_count, F }, audio_feats->data);
    if (!x_flat) { err = VOXCPM_ERR_OOM; goto cleanup; }

    // Projection result: [B*T*P, D]
    proj = tensor_create(2, (int[]){ flat_count, D });
    if (!proj) { err = VOXCPM_ERR_OOM; goto cleanup; }

    // proj = audio_feats_flat @ in_proj_weight^T
    //   A[M,K] = [flat_count, F], B[N,K] = [D, F]
    //   tensor_matmul_nt: A @ B^T = [flat_count, D]
    err = tensor_matmul_nt(x_flat, enc->in_proj_weight, proj);
    if (err) goto cleanup;

    // Add bias: proj[i, :] += bias[:]
    // in_proj_bias: [D]
    if (enc->in_proj_bias) {
        for (int i = 0; i < flat_count; i++) {
            float* row = proj->data + (size_t)i * D;
            for (int j = 0; j < D; j++) {
                row[j] += enc->in_proj_bias->data[j];
            }
        }
    }

    // ═════════════════════════════════════════════════════════════
    // Step 2: Build encoder input
    //   For each (b, t): [special_token(1, D)] + [proj_patches(P, D)]
    //   → x shape: [B, T*(P+1), D]
    // ═════════════════════════════════════════════════════════════

    x = tensor_create(3, (int[]){ B, seq_total, D });
    if (!x) { err = VOXCPM_ERR_OOM; goto cleanup; }

    const float* special_data = enc->special_token->data;  /* [1, 1, 1, D] */

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            /* Offset into x for this time-step group */
            size_t x_grp_off = ((size_t)b * seq_total + (size_t)t * group_size) * D;

            /* Copy special token (CLS) — first position in the group */
            memcpy(x->data + x_grp_off, special_data, (size_t)D * sizeof(float));

            /* Copy projected patches — positions 1..P in the group */
            size_t proj_off = ((size_t)b * T + (size_t)t) * (size_t)P * D;
            memcpy(x->data + x_grp_off + D, proj->data + proj_off,
                   (size_t)P * D * sizeof(float));
        }
    }

    // ═════════════════════════════════════════════════════════════
    // Step 3+4: Transformer encoder layers
    //   Bidirectional self-attention (no causal mask)
    //   No RoPE for LocEnc
    //   In-place: x is updated by each layer
    // ═════════════════════════════════════════════════════════════

    // K/V cache for each layer (reused — overwritten each layer since
    // each layer computes independent self-attention)
    cache_k = tensor_create(4, (int[]){ B, n_kv_heads, seq_total, head_dim });
    cache_v = tensor_create(4, (int[]){ B, n_kv_heads, seq_total, head_dim });
    if (!cache_k || !cache_v) { err = VOXCPM_ERR_OOM; goto cleanup; }

    for (int i = 0; i < enc->n_layers; i++) {
        err = transformer_block_forward(
            &enc->layers[i],     /* layer */
            x,                   /* input (and output, in-place) */
            cache_k, cache_v,    /* K/V cache (single-use per layer) */
            NULL,                /* mask = NULL (bidirectional attention) */
            NULL,                /* freqs_cis = NULL (no RoPE) */
            0,                   /* position offset */
            x                    /* output (same as input = in-place) */
        );
        if (err) {
            LOG_ERROR("loc_enc_forward: transformer_block_forward layer %d failed", i);
            goto cleanup;
        }
    }

    // ═════════════════════════════════════════════════════════════
    // Step 5: Extract CLS tokens
    //   CLS is the first token (position 0) of each (P+1)-sized group
    //   x -> out: [B, T, D]
    // ═════════════════════════════════════════════════════════════

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            size_t src_off = ((size_t)b * seq_total + (size_t)t * group_size) * D;
            size_t dst_off = ((size_t)b * T + (size_t)t) * D;
            memcpy(out->data + dst_off, x->data + src_off,
                   (size_t)D * sizeof(float));
        }
    }

    // ═════════════════════════════════════════════════════════════
    // Step 6: Apply output RMS norm
    // ═════════════════════════════════════════════════════════════

    err = rms_norm_forward(enc->output_norm, out, out);
    if (err) {
        LOG_ERROR("loc_enc_forward: output norm failed");
        goto cleanup;
    }

cleanup:
    tensor_free(x_flat);
    tensor_free(proj);
    tensor_free(x);
    tensor_free(cache_k);
    tensor_free(cache_v);
    return err;
}

/* ═════════════════════════════════════════════════════════════════
 * loc_enc_to_cuda — Upload LocEnc weights and sub-modules to GPU
 * ═════════════════════════════════════════════════════════════════ */
#ifdef VOXCPM_CUDA
VoxCPMError loc_enc_to_cuda(LocEnc* enc) {
    if (!enc) return VOXCPM_ERR_INTERNAL;

    VoxCPMError err;

    err = tensor_to_cuda(enc->in_proj_weight);
    if (err) return err;

    err = tensor_to_cuda(enc->in_proj_bias);
    if (err) return err;

    err = tensor_to_cuda(enc->special_token);
    if (err) return err;

    for (int i = 0; i < enc->n_layers; i++) {
        err = transformer_block_to_cuda(&enc->layers[i]);
        if (err) return err;
    }

    err = rms_norm_to_cuda(enc->output_norm);
    if (err) return err;

    return VOXCPM_SUCCESS;
}
#else
VoxCPMError loc_enc_to_cuda(LocEnc* enc) {
    (void)enc;
    return VOXCPM_ERR_CUDA_NOT_FOUND;
}
#endif /* VOXCPM_CUDA */

// loc_dit.c — LocDiT forward pass and DDIM sampling
// VoxCPM2-C Project
// License: Apache-2.0
//
// Local Diffusion Transformer (LocDiT): 12-layer diffusion denoising transformer.
//
// Architecture:
//   1. Input projection:   x [B,F,T]  → permute → linear → [B,T,dit_hidden]
//   2. Condition projection: cond [B,F,T'] → permute → linear → [B,T',dit_hidden]
//   3. Sinusoidal time embedding + time_mlp (SiLU MLP)
//   4. Sinusoidal delta embedding + delta_mlp (SiLU MLP)
//   5. Build sequence = [mu_prefix(1); x_proj(T); cond_proj(T')] + global_cond(time+delta)
//   6. 12 transformer decoder layers (bidirectional, no RoPE)
//   7. Extract x portion, output RMS norm
//   8. Output projection + permute → [B,feat_dim,time]

#include "model.h"
#include "nn.h"

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <string.h>

/* ═════════════════════════════════════════════════════════════════
 * Helpers
 * ═════════════════════════════════════════════════════════════════ */

/* Sinusoidal timestep embedding.
 *   out[b, 2*i]   = sin(t[b] * inv_freq[i])
 *   out[b, 2*i+1] = cos(t[b] * inv_freq[i])
 *   inv_freq[i] = 1.0f / powf(10000.0f, 2*i / d_model)
 */
static void sinusoidal_timestep_embedding(
    const float* t_vals, int B, int d_model, float* out)
{
    int half = d_model / 2;
    for (int b = 0; b < B; b++) {
        float t = t_vals[b];
        float* row = out + (size_t)b * (size_t)d_model;
        for (int i = 0; i < half; i++) {
            float inv_freq = 1.0f / powf(10000.0f, 2.0f * (float)i / (float)d_model);
            float val = t * inv_freq;
            row[2 * i]       = sinf(val);
            row[2 * i + 1]   = cosf(val);
        }
    }
}

/* Add 1D bias to 2D tensor in-place: out[i, :] += bias[:] */
static VoxCPMError add_bias_2d(Tensor* out, const Tensor* bias) {
    if (!out || !bias || out->ndim != 2 || bias->ndim != 1)
        return VOXCPM_ERR_INTERNAL;
    if (out->shape[1] != bias->shape[0])
        return VOXCPM_ERR_SHAPE_MISMATCH;
    int rows = out->shape[0];
    int cols = out->shape[1];
    for (int i = 0; i < rows; i++) {
        float* row = out->data + (size_t)i * (size_t)cols;
        for (int j = 0; j < cols; j++) {
            row[j] += bias->data[j];
        }
    }
    return VOXCPM_SUCCESS;
}

/* Add 2D conditioning [B, D] to all positions of 3D tensor [B, T, D] in-place:
 *   seq[b, t, :] += cond[b, :]  for all t
 */
static VoxCPMError add_cond_to_all(Tensor* seq, const Tensor* cond) {
    if (!seq || !cond || seq->ndim != 3 || cond->ndim != 2)
        return VOXCPM_ERR_INTERNAL;
    if (seq->shape[0] != cond->shape[0] || seq->shape[2] != cond->shape[1])
        return VOXCPM_ERR_SHAPE_MISMATCH;
    int B = seq->shape[0];
    int T = seq->shape[1];
    int D = seq->shape[2];
    for (int b = 0; b < B; b++) {
        const float* c_row = cond->data + (size_t)b * (size_t)D;
        float* s_base      = seq->data  + (size_t)b * (size_t)T * (size_t)D;
        for (int t = 0; t < T; t++) {
            float* s_row = s_base + (size_t)t * (size_t)D;
            for (int d = 0; d < D; d++) {
                s_row[d] += c_row[d];
            }
        }
    }
    return VOXCPM_SUCCESS;
}

/* ═════════════════════════════════════════════════════════════════
 * loc_dit_forward
 * ═════════════════════════════════════════════════════════════════ */
VoxCPMError loc_dit_forward(
    const LocDiT* dit,
    const Tensor* x,          /* [batch, feat_dim, time] — noisy latent */
    const Tensor* mu,         /* [batch, dit_hidden] — TSLM conditioning */
    const Tensor* cond,       /* [batch, feat_dim, time'] — prefix condition */
    const Tensor* t,          /* [batch] — timestep */
    const Tensor* dt,         /* [batch] — delta time */
    Tensor* out               /* [batch, feat_dim, time] — denoised */
) {
    if (!dit || !x || !mu || !cond || !t || !dt || !out)
        return VOXCPM_ERR_INTERNAL;

    /* ─────────────────────────────────────────────────────────────
     * Input validation
     * ───────────────────────────────────────────────────────────── */
    if (x->ndim != 3) return VOXCPM_ERR_SHAPE_MISMATCH;
    int B = x->shape[0];
    int F = x->shape[1];       /* feat_dim */
    int T = x->shape[2];       /* time (sequence length of x) */

    if (cond->ndim != 3 || cond->shape[0] != B || cond->shape[1] != F)
        return VOXCPM_ERR_SHAPE_MISMATCH;
    int T_cond = cond->shape[2];

    int D = dit->d_model;      /* dit_hidden (1024) */
    if (mu->ndim != 2 || mu->shape[0] != B || mu->shape[1] != D)
        return VOXCPM_ERR_SHAPE_MISMATCH;
    if (t->ndim != 1 || t->shape[0] != B)
        return VOXCPM_ERR_SHAPE_MISMATCH;
    if (dt->ndim != 1 || dt->shape[0] != B)
        return VOXCPM_ERR_SHAPE_MISMATCH;
    if (out->ndim  != 3 || out->shape[0]  != B ||
        out->shape[1] != F || out->shape[2] != T)
        return VOXCPM_ERR_SHAPE_MISMATCH;

    if (dit->n_layers < 1 || !dit->layers || !dit->layers[0].attn)
        return VOXCPM_ERR_INTERNAL;

    /* Derived dimensions */
    int n_kv_heads = dit->layers[0].attn->n_kv_heads;
    int head_dim   = dit->layers[0].attn->head_dim;
    int seq_total  = 1 + T + T_cond;  /* mu_prefix + x_proj + cond_proj */

    /* ─────────────────────────────────────────────────────────────
     * Temporary tensors
     * ───────────────────────────────────────────────────────────── */
    Tensor* x_perm      = NULL;  /* [B, T, F] */
    Tensor* cond_perm   = NULL;  /* [B, T', F] */
    Tensor* x_flat      = NULL;  /* [B*T, F]   view */
    Tensor* cond_flat   = NULL;  /* [B*T', F]  view */
    Tensor* x_proj      = NULL;  /* [B*T, D] */
    Tensor* cond_proj   = NULL;  /* [B*T', D] */
    Tensor* sin_t       = NULL;  /* [B, D] */
    Tensor* sin_dt      = NULL;  /* [B, D] */
    Tensor* mlp_temp    = NULL;  /* [B, D] reusable */
    Tensor* time_feat   = NULL;  /* [B, D] */
    Tensor* delta_feat  = NULL;  /* [B, D] */
    Tensor* seq         = NULL;  /* [B, seq_total, D] */
    Tensor* cache_k     = NULL;  /* [B, n_kv_heads, seq_total, head_dim] */
    Tensor* cache_v     = NULL;  /* [B, n_kv_heads, seq_total, head_dim] */
    Tensor* x_slice     = NULL;  /* [B, T, D] */
    Tensor* x_slice_flat = NULL; /* [B*T, D]  view */
    Tensor* out_perm    = NULL;  /* [B, T, F] */
    Tensor* out_flat    = NULL;  /* [B*T, F]  view */

    VoxCPMError err = VOXCPM_SUCCESS;

    /* ═════════════════════════════════════════════════════════════
     * Step 1: Input projection
     *   x [B, F, T] → permute(0,2,1) → [B, T, F]
     *   → flatten [B*T, F] → matmul_nt(in_proj_weight[D,F]) → [B*T, D]
     *   → add bias [D]
     * ═════════════════════════════════════════════════════════════ */
    x_perm = tensor_create(3, (int[]){ B, T, F });
    if (!x_perm) { err = VOXCPM_ERR_OOM; goto cleanup; }
    err = tensor_permute(x, x_perm, (int[]){ 0, 2, 1 });
    if (err) goto cleanup;

    x_flat = tensor_create_from_buffer(2, (int[]){ B * T, F }, x_perm->data);
    if (!x_flat) { err = VOXCPM_ERR_OOM; goto cleanup; }

    x_proj = tensor_create(2, (int[]){ B * T, D });
    if (!x_proj) { err = VOXCPM_ERR_OOM; goto cleanup; }

    err = tensor_matmul_nt(x_flat, dit->in_proj_weight, x_proj);
    if (err) goto cleanup;

    if (dit->in_proj_bias) {
        err = add_bias_2d(x_proj, dit->in_proj_bias);
        if (err) goto cleanup;
    }

    /* ═════════════════════════════════════════════════════════════
     * Step 2: Condition projection (if T_cond > 0)
     *   cond [B, F, T'] → permute(0,2,1) → [B, T', F]
     *   → flatten → matmul_nt(cond_proj_weight) → [B*T', D]
     *   → add bias [D]
     * ═════════════════════════════════════════════════════════════ */
    if (T_cond > 0) {
        cond_perm = tensor_create(3, (int[]){ B, T_cond, F });
        if (!cond_perm) { err = VOXCPM_ERR_OOM; goto cleanup; }
        err = tensor_permute(cond, cond_perm, (int[]){ 0, 2, 1 });
        if (err) goto cleanup;

        cond_flat = tensor_create_from_buffer(2, (int[]){ B * T_cond, F },
                                               cond_perm->data);
        if (!cond_flat) { err = VOXCPM_ERR_OOM; goto cleanup; }

        cond_proj = tensor_create(2, (int[]){ B * T_cond, D });
        if (!cond_proj) { err = VOXCPM_ERR_OOM; goto cleanup; }

        err = tensor_matmul_nt(cond_flat, dit->cond_proj_weight, cond_proj);
        if (err) goto cleanup;

        if (dit->cond_proj_bias) {
            err = add_bias_2d(cond_proj, dit->cond_proj_bias);
            if (err) goto cleanup;
        }
    }

    /* ═════════════════════════════════════════════════════════════
     * Step 3: Sinusoidal timestep embedding + time_mlp
     *   t [B] → sin_emb [B, D]
     *   → matmul_nt(W1[D,D]) → +b1 → SiLU
     *   → matmul_nt(W2[D,D]) → +b2 → time_feat [B, D]
     * ═════════════════════════════════════════════════════════════ */
    sin_t = tensor_create(2, (int[]){ B, D });
    if (!sin_t) { err = VOXCPM_ERR_OOM; goto cleanup; }
    sinusoidal_timestep_embedding(t->data, B, D, sin_t->data);

    mlp_temp = tensor_create(2, (int[]){ B, D });
    if (!mlp_temp) { err = VOXCPM_ERR_OOM; goto cleanup; }

    /* hidden = SiLU(sin_emb @ W1^T + b1) */
    err = tensor_matmul_nt(sin_t, dit->time_mlp_1_weight, mlp_temp);
    if (err) goto cleanup;
    if (dit->time_mlp_1_bias) {
        err = add_bias_2d(mlp_temp, dit->time_mlp_1_bias);
        if (err) goto cleanup;
    }
    err = tensor_silu(mlp_temp);
    if (err) goto cleanup;

    /* time_feat = hidden @ W2^T + b2 */
    time_feat = tensor_create(2, (int[]){ B, D });
    if (!time_feat) { err = VOXCPM_ERR_OOM; goto cleanup; }
    err = tensor_matmul_nt(mlp_temp, dit->time_mlp_2_weight, time_feat);
    if (err) goto cleanup;
    if (dit->time_mlp_2_bias) {
        err = add_bias_2d(time_feat, dit->time_mlp_2_bias);
        if (err) goto cleanup;
    }

    /* ═════════════════════════════════════════════════════════════
     * Step 4: Sinusoidal delta embedding + delta_mlp
     *   dt [B] → sin_emb [B, D]
     *   → matmul_nt(W1[D,D]) → +b1 → SiLU
     *   → matmul_nt(W2[D,D]) → +b2 → delta_feat [B, D]
     * ═════════════════════════════════════════════════════════════ */
    sin_dt = tensor_create(2, (int[]){ B, D });
    if (!sin_dt) { err = VOXCPM_ERR_OOM; goto cleanup; }
    sinusoidal_timestep_embedding(dt->data, B, D, sin_dt->data);

    /* hidden = SiLU(sin_dt @ W1^T + b1)  (reuse mlp_temp) */
    err = tensor_matmul_nt(sin_dt, dit->delta_mlp_1_weight, mlp_temp);
    if (err) goto cleanup;
    if (dit->delta_mlp_1_bias) {
        err = add_bias_2d(mlp_temp, dit->delta_mlp_1_bias);
        if (err) goto cleanup;
    }
    err = tensor_silu(mlp_temp);
    if (err) goto cleanup;

    /* delta_feat = hidden @ W2^T + b2 */
    delta_feat = tensor_create(2, (int[]){ B, D });
    if (!delta_feat) { err = VOXCPM_ERR_OOM; goto cleanup; }
    err = tensor_matmul_nt(mlp_temp, dit->delta_mlp_2_weight, delta_feat);
    if (err) goto cleanup;
    if (dit->delta_mlp_2_bias) {
        err = add_bias_2d(delta_feat, dit->delta_mlp_2_bias);
        if (err) goto cleanup;
    }

    /* ═════════════════════════════════════════════════════════════
     * Step 5: Build full sequence
     *
     *   global_cond = time_feat + delta_feat  [B, D]
     *     (added to every token position)
     *
     *   Sequence layout:
     *     pos 0:      mu [B, 1, D]         — conditioning prefix
     *     pos 1..T:   x_proj [B, T, D]     — input projection
     *     pos T+1..:  cond_proj [B, T', D]  — condition projection
     *
     *   Then add global_cond to all tokens.
     * ═════════════════════════════════════════════════════════════ */

    /* global_cond = time_feat + delta_feat (in-place) */
    err = tensor_add(time_feat, delta_feat, time_feat);
    if (err) goto cleanup;

    seq = tensor_create(3, (int[]){ B, seq_total, D });
    if (!seq) { err = VOXCPM_ERR_OOM; goto cleanup; }

    /* pos 0: mu (conditioning prefix) */
    for (int b = 0; b < B; b++) {
        memcpy(seq->data + (size_t)b * (size_t)seq_total * (size_t)D,
               mu->data + (size_t)b * (size_t)D,
               (size_t)D * sizeof(float));
    }

    /* pos 1..T: x_proj */
    for (int b = 0; b < B; b++) {
        float* dst = seq->data + (size_t)b * (size_t)seq_total * (size_t)D + (size_t)D;
        float* src = x_proj->data + (size_t)b * (size_t)T * (size_t)D;
        memcpy(dst, src, (size_t)T * (size_t)D * sizeof(float));
    }

    /* pos 1+T..1+T+T': cond_proj (if any) */
    if (T_cond > 0 && cond_proj) {
        for (int b = 0; b < B; b++) {
            float* dst = seq->data  + (size_t)b * (size_t)seq_total * (size_t)D
                        + (size_t)(1 + T) * (size_t)D;
            float* src = cond_proj->data + (size_t)b * (size_t)T_cond * (size_t)D;
            memcpy(dst, src, (size_t)T_cond * (size_t)D * sizeof(float));
        }
    }

    /* Add global_cond to every token position */
    err = add_cond_to_all(seq, time_feat);
    if (err) goto cleanup;

    /* ═════════════════════════════════════════════════════════════
     * Step 6: Transformer decoder layers
     *   Bidirectional self-attention (no causal mask, no RoPE)
     *   In-place: seq is updated by each layer
     * ═════════════════════════════════════════════════════════════ */
    cache_k = tensor_create(4, (int[]){ B, n_kv_heads, seq_total, head_dim });
    cache_v = tensor_create(4, (int[]){ B, n_kv_heads, seq_total, head_dim });
    if (!cache_k || !cache_v) { err = VOXCPM_ERR_OOM; goto cleanup; }

    for (int i = 0; i < dit->n_layers; i++) {
        err = transformer_block_forward(
            &dit->layers[i],
            seq,
            cache_k, cache_v,
            NULL,           /* mask = NULL (bidirectional) */
            NULL,           /* freqs_cis = NULL (no RoPE) */
            0,              /* position offset */
            seq             /* output = input (in-place) */
        );
        if (err) {
            LOG_ERROR("loc_dit_forward: transformer_block_forward layer %d failed", i);
            goto cleanup;
        }
    }

    /* ═════════════════════════════════════════════════════════════
     * Step 7: Extract x portion + output RMS norm
     *   Extract seq[:, 1:1+T, :] → x_slice [B, T, D]
     *   Apply output_norm
     * ═════════════════════════════════════════════════════════════ */
    x_slice = tensor_create(3, (int[]){ B, T, D });
    if (!x_slice) { err = VOXCPM_ERR_OOM; goto cleanup; }

    for (int b = 0; b < B; b++) {
        float* src = seq->data + (size_t)b * (size_t)seq_total * (size_t)D + (size_t)D;
        float* dst = x_slice->data + (size_t)b * (size_t)T * (size_t)D;
        memcpy(dst, src, (size_t)T * (size_t)D * sizeof(float));
    }

    err = rms_norm_forward(dit->output_norm, x_slice, x_slice);
    if (err) {
        LOG_ERROR("loc_dit_forward: output norm failed");
        goto cleanup;
    }

    /* ═════════════════════════════════════════════════════════════
     * Step 8: Output projection
     *   x_slice [B, T, D] → flatten [B*T, D]
     *   → matmul_nt(out_proj_weight[F, D]) → [B*T, F]
     *   → add bias [F]
     *   → reshape [B, T, F] → permute(0,2,1) → [B, F, T] = out
     * ═════════════════════════════════════════════════════════════ */
    x_slice_flat = tensor_create_from_buffer(2, (int[]){ B * T, D }, x_slice->data);
    if (!x_slice_flat) { err = VOXCPM_ERR_OOM; goto cleanup; }

    out_perm = tensor_create(3, (int[]){ B, T, F });
    if (!out_perm) { err = VOXCPM_ERR_OOM; goto cleanup; }

    out_flat = tensor_create_from_buffer(2, (int[]){ B * T, F }, out_perm->data);
    if (!out_flat) { err = VOXCPM_ERR_OOM; goto cleanup; }

    err = tensor_matmul_nt(x_slice_flat, dit->out_proj_weight, out_flat);
    if (err) { LOG_ERROR("loc_dit: out_proj matmul failed"); goto cleanup; }

    if (dit->out_proj_bias) {
        err = add_bias_2d(out_flat, dit->out_proj_bias);
        if (err) { LOG_ERROR("loc_dit: out_proj bias failed"); goto cleanup; }
    }

    /* Permute [B, T, F] → [B, F, T] */
    err = tensor_permute(out_perm, out, (int[]){ 0, 2, 1 });
    if (err) { LOG_ERROR("loc_dit: permute out failed"); goto cleanup; }

    /* ────────────────────────────────────────────────────────────
     * Cleanup
     * ──────────────────────────────────────────────────────────── */
cleanup:
    /* Free views first (before their owning buffers) */
    tensor_free(x_flat);
    tensor_free(cond_flat);
    tensor_free(x_slice_flat);
    tensor_free(out_flat);
    /* Free owning tensors */
    tensor_free(x_perm);
    tensor_free(cond_perm);
    tensor_free(x_proj);
    tensor_free(cond_proj);
    tensor_free(sin_t);
    tensor_free(sin_dt);
    tensor_free(mlp_temp);
    tensor_free(time_feat);
    tensor_free(delta_feat);
    tensor_free(seq);
    tensor_free(cache_k);
    tensor_free(cache_v);
    tensor_free(x_slice);
    tensor_free(out_perm);
    return err;
}

/* ═════════════════════════════════════════════════════════════════
 * loc_dit_sample — DDIM diffusion sampling with optional CFG
 *
 *   Generates a clean latent patch from random noise using
 *   DDIM (deterministic) denoising with cosine noise schedule.
 *   Supports classifier-free guidance (CFG).
 * ═════════════════════════════════════════════════════════════════ */
VoxCPMError loc_dit_sample(
    const LocDiT* dit,
    const Tensor* mu,                /* [batch, dit_hidden] */
    const Tensor* cond,              /* [batch, feat_dim, time'] */
    int n_timesteps,
    float cfg_value,
    Tensor* out                      /* [batch, feat_dim, patch_size] */
) {
    if (!dit || !mu || !cond || !out) return VOXCPM_ERR_INTERNAL;
    if (mu->ndim != 2) return VOXCPM_ERR_SHAPE_MISMATCH;
    int B = mu->shape[0];
    int F = dit->feat_dim;

    if (out->ndim != 3) return VOXCPM_ERR_SHAPE_MISMATCH;
    if (out->shape[0] != B || out->shape[1] != F)
        return VOXCPM_ERR_SHAPE_MISMATCH;
    int P = out->shape[2];  /* patch_size */

    /* Verify mu shape matches dit_hidden */
    if (mu->shape[1] != dit->d_model)
        return VOXCPM_ERR_SHAPE_MISMATCH;

    if (cond->ndim != 3 || cond->shape[0] != B || cond->shape[1] != F)
        return VOXCPM_ERR_SHAPE_MISMATCH;
    int T_cond = cond->shape[2];

    if (n_timesteps < 1) return VOXCPM_ERR_INTERNAL;
    if (cfg_value < 0.0f) cfg_value = 0.0f;

    /* ─────────────────────────────────────────────────────────────
     * Allocate state
     * ───────────────────────────────────────────────────────────── */

    /* Initial noise: x_T  [B, F, P] */
    uint64_t seed = 42;
    Tensor* x_t = tensor_create(3, (int[]){ B, F, P });
    if (!x_t) return VOXCPM_ERR_OOM;
    tensor_rand_normal(x_t, &seed);

    /* Unconditional condition for CFG (zero-filled) */
    Tensor* uncond = NULL;
    if (cfg_value > 1.0f && T_cond > 0) {
        uncond = tensor_create(3, (int[]){ B, F, T_cond });
        if (!uncond) { tensor_free(x_t); return VOXCPM_ERR_OOM; }
        tensor_zero(uncond);
    }

    /* Reusable temporaries */
    Tensor* pred      = tensor_create(3, (int[]){ B, F, P });
    Tensor* pred_un   = tensor_create(3, (int[]){ B, F, P });
    Tensor* t_tensor  = tensor_create(1, (int[]){ B });
    Tensor* dt_tensor = tensor_create(1, (int[]){ B });
    Tensor* x_next    = tensor_create(3, (int[]){ B, F, P });

    if (!pred || !pred_un || !t_tensor || !dt_tensor || !x_next) {
        tensor_free(x_t);
        tensor_free(uncond);
        tensor_free(pred);
        tensor_free(pred_un);
        tensor_free(t_tensor);
        tensor_free(dt_tensor);
        tensor_free(x_next);
        return VOXCPM_ERR_OOM;
    }

    VoxCPMError err = VOXCPM_SUCCESS;

    /* DDIM noise schedule: cosine schedule over T=1000 steps
     *   alpha_bar(t)  = cos(pi/2 * t/T)^2
     *   sqrt(alpha)   = cos(pi/2 * t/T)
     *   sqrt(1-alpha) = sin(pi/2 * t/T)
     */
    const float T_total = 1000.0f;
    const float step    = T_total / (float)n_timesteps;

    for (int i = 0; i < n_timesteps; i++) {
        /* Current and next timestep values (decreasing) */
        float t_val       = T_total - 1.0f - (float)i * step;
        float t_next_val  = t_val - step;
        if (t_val      < 0.0f) t_val      = 0.0f;
        if (t_next_val < 0.0f) t_next_val = 0.0f;

        /* Fill timestep tensors (dt = positive step size) */
        for (int b = 0; b < B; b++) {
            t_tensor->data[b]  = t_val;
            dt_tensor->data[b] = step;
        }

        /* ─── Conditional prediction (x0_pred) ─── */
        err = loc_dit_forward(dit, x_t, mu, cond, t_tensor, dt_tensor, pred);
        if (err) {
            LOG_ERROR("loc_dit_sample: forward failed at step %d/%d", i, n_timesteps);
            goto cleanup_sample;
        }

        /* ─── CFG: unconditional prediction + combine ─── */
        if (cfg_value > 1.0f && uncond) {
            err = loc_dit_forward(dit, x_t, mu, uncond, t_tensor, dt_tensor, pred_un);
            if (err) {
                LOG_ERROR("loc_dit_sample: uncond forward failed at step %d/%d",
                          i, n_timesteps);
                goto cleanup_sample;
            }

            /* CFG combine: pred = uncond + cfg * (cond - uncond) */
            for (size_t j = 0; j < pred->size; j++) {
                float cond_p = pred->data[j];
                float uncond_p = pred_un->data[j];
                pred->data[j] = uncond_p + cfg_value * (cond_p - uncond_p);
            }
        }

        /* ─── DDIM reverse step ───
         *
         * Cosine schedule at timestep t:
         *   sqrt(alpha_bar(t))  = a(t) = cos(pi/2 * t/T)
         *   sqrt(1-alpha_bar(t)) = b(t) = sin(pi/2 * t/T)
         *
         * Model predicts x0 (denoised).
         * Implied noise: eps = (x_t - a(t) * x0_pred) / b(t)
         *
         * DDIM update (x0-pred formulation):
         *   x_{t-1} = a(t-1) * x0_pred + b(t-1) * eps
         *           = a_next * x0p + b_next * (x_t - a_cur * x0p) / b_cur
         */
        {
            float a_cur = cosf(0.5f * (float)M_PI * t_val / T_total);
            float a_next = cosf(0.5f * (float)M_PI * t_next_val / T_total);
            float b_cur = sinf(0.5f * (float)M_PI * t_val / T_total);
            float b_next = sinf(0.5f * (float)M_PI * t_next_val / T_total);

            for (size_t j = 0; j < x_t->size; j++) {
                float x0p = pred->data[j];
                float xt  = x_t->data[j];

                /* Derived noise prediction (avoid division by zero) */
                float eps_pred;
                if (b_cur > 1e-8f) {
                    eps_pred = (xt - a_cur * x0p) / b_cur;
                } else {
                    eps_pred = 0.0f;
                }

                /* x_{t-1} = a_next * x0p + b_next * eps_pred */
                x_next->data[j] = a_next * x0p + b_next * eps_pred;
            }
        }

        /* Swap x_t and x_next for next iteration */
        {
            Tensor* tmp = x_t;
            x_t   = x_next;
            x_next = tmp;
        }
    }

    /* Copy final denoised result to output */
    err = tensor_copy(out, x_t);

cleanup_sample:
    tensor_free(x_t);
    tensor_free(uncond);
    tensor_free(pred);
    tensor_free(pred_un);
    tensor_free(t_tensor);
    tensor_free(dt_tensor);
    tensor_free(x_next);
    return err;
}

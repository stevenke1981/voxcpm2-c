// nn.c — Neural network layer implementations (stubs for Phase 0)
// VoxCPM2-C Project
// License: Apache-2.0
//
// Full implementations will be added in Phase 1-2.
// Phase 0: stub functions that return VOXCPM_ERR_UNSUPPORTED.

#include "nn.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════
 * RMS Norm
 * ═══════════════════════════════════════════════════════════════ */

RmsNorm* rms_norm_create(int d_model, float eps) {
    RmsNorm* norm = (RmsNorm*)calloc(1, sizeof(RmsNorm));
    if (!norm) return NULL;
    norm->weight = tensor_create(1, (int[]){d_model});
    norm->eps = eps;
    if (!norm->weight) { free(norm); return NULL; }
    tensor_fill(norm->weight, 1.0f); // Init to ones
    return norm;
}

void rms_norm_free(RmsNorm* norm) {
    if (!norm) return;
    tensor_free(norm->weight);
    free(norm);
}

VoxCPMError rms_norm_forward(const RmsNorm* norm, const Tensor* x, Tensor* out) {
    if (!norm || !x || !out) return VOXCPM_ERR_INTERNAL;
    return tensor_rms_norm(x, norm->weight, norm->eps, out);
}

/* ═══════════════════════════════════════════════════════════════
 * Layer Norm
 * ═══════════════════════════════════════════════════════════════ */

LayerNorm* layer_norm_create(int d_model, float eps, bool use_bias) {
    LayerNorm* norm = (LayerNorm*)calloc(1, sizeof(LayerNorm));
    if (!norm) return NULL;

    norm->weight = tensor_create(1, (int[]){d_model});
    if (!norm->weight) { free(norm); return NULL; }
    tensor_fill(norm->weight, 1.0f);

    if (use_bias) {
        norm->bias = tensor_create(1, (int[]){d_model});
        if (!norm->bias) { tensor_free(norm->weight); free(norm); return NULL; }
        tensor_zero(norm->bias);
    } else {
        norm->bias = NULL;
    }

    norm->eps = eps;
    return norm;
}

void layer_norm_free(LayerNorm* norm) {
    if (!norm) return;
    tensor_free(norm->weight);
    tensor_free(norm->bias);
    free(norm);
}

VoxCPMError layer_norm_forward(const LayerNorm* norm, const Tensor* x, Tensor* out) {
    if (!norm || !x || !out) return VOXCPM_ERR_INTERNAL;
    return tensor_layer_norm(x, norm->weight, norm->bias, norm->eps, out);
}

/* ═══════════════════════════════════════════════════════════════
 * SwiGLU FFN
 * ═══════════════════════════════════════════════════════════════ */

SwiGLU* swiglu_create(int d_model, int d_ff) {
    SwiGLU* ff = (SwiGLU*)calloc(1, sizeof(SwiGLU));
    if (!ff) return NULL;

    ff->d_model = d_model;
    ff->d_ff = d_ff;
    ff->w1 = NULL; // Will be created during weight loading
    ff->w2 = NULL;
    ff->w3 = NULL;
    return ff;
}

void swiglu_free(SwiGLU* ff) {
    if (!ff) return;
    tensor_free(ff->w1);
    tensor_free(ff->w2);
    tensor_free(ff->w3);
    free(ff);
}

VoxCPMError swiglu_forward(const SwiGLU* ff, const Tensor* x, Tensor* out) {
    LOG_INFO("TRACE swiglu ENTER");
    if (!ff || !x || !out) return VOXCPM_ERR_INTERNAL;
    if (!ff->w1 || !ff->w2 || !ff->w3) return VOXCPM_ERR_INTERNAL;

    int ndim = x->ndim;
    if (ndim < 2) { LOG_ERROR("swiglu S01: ndim=%d < 2", ndim); return VOXCPM_ERR_SHAPE_MISMATCH; }
    if (x->shape[ndim - 1] != ff->d_model) {
        LOG_ERROR("swiglu S02: x[last=%d]=%d != d_model=%d", ndim-1, x->shape[ndim-1], ff->d_model);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (!tensor_shape_eq(x, out)) {
        LOG_ERROR("swiglu S03: x!=out shape");
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    // Compute flat batch size: product of all dims except the last (d_model)
    int B = 1;
    for (int i = 0; i < ndim - 1; i++) {
        B *= x->shape[i];
    }

    int d_ff   = ff->d_ff;
    int d_model = ff->d_model;

    // Create 2D flat views of input and output (non-owning, shares data)
    int shape_2d_m[2] = {B, d_model};
    int shape_2d_f[2] = {B, d_ff};

    Tensor* x_flat = tensor_create_from_buffer(2, shape_2d_m, x->data);
    Tensor* out_flat = tensor_create_from_buffer(2, shape_2d_m, out->data);
    if (!x_flat || !out_flat) {
        tensor_free(x_flat);
        tensor_free(out_flat);
        return VOXCPM_ERR_OOM;
    }

    // Allocate temporary tensors for intermediate results
    Tensor* gate   = tensor_create(2, shape_2d_f);
    Tensor* up     = tensor_create(2, shape_2d_f);
    Tensor* hidden = tensor_create(2, shape_2d_f);

    VoxCPMError err = VOXCPM_SUCCESS;

    if (!gate || !up || !hidden) {
        err = VOXCPM_ERR_OOM;
        goto cleanup;
    }

    // FFN(x) = (silu(x @ W1^T) ⊙ (x @ W3^T)) @ W2^T
    // Weights stored in PyTorch format [out_dim, in_dim], so use matmul_nt: out = x @ W^T
    //
    // Step 1: gate = x @ W1^T   [B, d_model] @ [d_ff, d_model]^T = [B, d_ff]
    err = tensor_matmul_nt(x_flat, ff->w1, gate);
    if (err) goto cleanup;

    // Step 2: up   = x @ W3^T   [B, d_model] @ [d_ff, d_model]^T = [B, d_ff]
    err = tensor_matmul_nt(x_flat, ff->w3, up);
    if (err) goto cleanup;

    // Step 3: gate = silu(gate)  (in-place activation)
    err = tensor_silu(gate);
    if (err) goto cleanup;

    // Step 4: hidden = gate * up  (element-wise multiply)
    err = tensor_mul(gate, up, hidden);
    if (err) goto cleanup;

    // Step 5: out = hidden @ W2^T   [B, d_ff] @ [d_model, d_ff]^T = [B, d_model]
    err = tensor_matmul_nt(hidden, ff->w2, out_flat);
    if (err) goto cleanup;

cleanup:
    tensor_free(x_flat);
    tensor_free(out_flat);
    tensor_free(gate);
    tensor_free(up);
    tensor_free(hidden);
    return err;
}

/* ═══════════════════════════════════════════════════════════════
 * Attention
 * ═══════════════════════════════════════════════════════════════ */

Attention* attention_create(int d_model, int n_heads, int n_kv_heads) {
    Attention* attn = (Attention*)calloc(1, sizeof(Attention));
    if (!attn) return NULL;

    attn->d_model = d_model;
    attn->n_heads = n_heads;
    attn->n_kv_heads = n_kv_heads;
    attn->head_dim = d_model / n_heads;
    attn->wq = NULL;
    attn->wk = NULL;
    attn->wv = NULL;
    attn->wo = NULL;
    return attn;
}

void attention_free(Attention* attn) {
    if (!attn) return;
    tensor_free(attn->wq);
    tensor_free(attn->wk);
    tensor_free(attn->wv);
    tensor_free(attn->wo);
    free(attn);
}

VoxCPMError attention_forward(
    const Attention* attn, const Tensor* x,
    Tensor* cache_k, Tensor* cache_v, const Tensor* mask,
    const Tensor* freqs_cis, int pos, Tensor* out)
{
    if (pos == 0) LOG_INFO("TRACE attn_forward ENTER pos=%d", pos);
    if (!attn || !x || !cache_k || !cache_v || !out)
        return VOXCPM_ERR_INTERNAL;

    // --- Input shape validation ---
    if (x->ndim != 3) {
        LOG_ERROR("attn SHAPE A01: x->ndim=%d != 3", x->ndim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    int batch   = x->shape[0];
    int seq     = x->shape[1];
    int d_model = x->shape[2];

    if (d_model != attn->d_model) {
        LOG_ERROR("attn SHAPE A02: d_model=%d != attn->d_model=%d", d_model, attn->d_model);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    // NOTE: qk_dim (head_dim * n_heads) may differ from d_model for some sub-models
    //       (e.g. DiT uses 2048 Q projection with d_model=1024).
    //       The output projection wo projects from qk_dim back to d_model.
    //       We rely on weight shape validation below instead of this strict equality.
    if (attn->n_heads % attn->n_kv_heads != 0) {
        LOG_ERROR("attn SHAPE A04: n_heads=%d %% n_kv_heads=%d != 0", attn->n_heads, attn->n_kv_heads);
        return VOXCPM_ERR_INTERNAL;
    }
    if (cache_k->ndim != 4 || cache_v->ndim != 4) {
        LOG_ERROR("attn SHAPE A05: cache_k->ndim=%d cache_v->ndim=%d", cache_k->ndim, cache_v->ndim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (cache_k->shape[0] != batch || cache_v->shape[0] != batch) {
        LOG_ERROR("attn SHAPE A06: cache_k[0]=%d cache_v[0]=%d batch=%d",
                  cache_k->shape[0], cache_v->shape[0], batch);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (cache_k->shape[1] != attn->n_kv_heads) {
        LOG_ERROR("attn SHAPE A07: cache_k[1]=%d != n_kv_heads=%d", cache_k->shape[1], attn->n_kv_heads);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (cache_v->shape[1] != attn->n_kv_heads) {
        LOG_ERROR("attn SHAPE A08: cache_v[1]=%d != n_kv_heads=%d", cache_v->shape[1], attn->n_kv_heads);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (cache_k->shape[3] != attn->head_dim) {
        LOG_ERROR("attn SHAPE A09: cache_k[3]=%d != head_dim=%d", cache_k->shape[3], attn->head_dim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (cache_v->shape[3] != attn->head_dim) {
        LOG_ERROR("attn SHAPE A10: cache_v[3]=%d != head_dim=%d", cache_v->shape[3], attn->head_dim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (out->ndim != 3) {
        LOG_ERROR("attn SHAPE A11: out->ndim=%d != 3", out->ndim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (out->shape[0] != batch || out->shape[1] != seq || out->shape[2] != d_model) {
        LOG_ERROR("attn SHAPE A12: out [%d,%d,%d] != x [%d,%d,%d]",
                  out->shape[0], out->shape[1], out->shape[2],
                  batch, seq, d_model);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    int n_heads      = attn->n_heads;
    int n_kv_heads   = attn->n_kv_heads;
    int head_dim     = attn->head_dim;
    int group_size   = n_heads / n_kv_heads;
    int max_seq      = cache_k->shape[2];    // maximum cache length
    int cache_len    = pos + seq;            // total cached positions after write
    int qk_dim       = n_heads   * head_dim; // = d_model
    int kv_dim       = n_kv_heads * head_dim;

    if (cache_len > max_seq) {
        LOG_ERROR("attn SHAPE A13: cache_len=%d > max_seq=%d", cache_len, max_seq);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    // Validate weight shapes match expected dimensions
    if (attn->wq->shape[0] != qk_dim || attn->wq->shape[1] != d_model) {
        LOG_ERROR("attn WEIGHT WQ: wq=[%d,%d] expected=[%d,%d]",
                  attn->wq->shape[0], attn->wq->shape[1], qk_dim, d_model);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (attn->wk->shape[0] != kv_dim || attn->wk->shape[1] != d_model) {
        LOG_ERROR("attn WEIGHT WK: wk=[%d,%d] expected=[%d,%d]",
                  attn->wk->shape[0], attn->wk->shape[1], kv_dim, d_model);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (attn->wv->shape[0] != kv_dim || attn->wv->shape[1] != d_model) {
        LOG_ERROR("attn WEIGHT WV: wv=[%d,%d] expected=[%d,%d]",
                  attn->wv->shape[0], attn->wv->shape[1], kv_dim, d_model);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (attn->wo->shape[0] != d_model || attn->wo->shape[1] != qk_dim) {
        LOG_ERROR("attn WEIGHT WO: wo=[%d,%d] expected=[%d,%d]",
                  attn->wo->shape[0], attn->wo->shape[1], d_model, qk_dim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    // ---- Temporary tensor pointers (all freed in cleanup) ----
    Tensor* x_flat    = NULL;   // 2D view of x
    Tensor* out_flat  = NULL;   // 2D view of out
    Tensor* q_proj    = NULL;   // [batch*seq, d_model]       → after reshape [batch, seq, n_heads, head_dim]
    Tensor* k_proj    = NULL;   // [batch*seq, kv_dim]        → after reshape [batch, seq, n_kv_heads, head_dim]
    Tensor* v_proj    = NULL;   // [batch*seq, kv_dim]        → after reshape [batch, seq, n_kv_heads, head_dim]
    Tensor* q_rope    = NULL;   // [batch, n_heads, seq, head_dim]
    Tensor* k_rope    = NULL;   // [batch, n_kv_heads, seq, head_dim]
    Tensor* v_rope    = NULL;   // [batch, n_kv_heads, seq, head_dim]
    Tensor* scores    = NULL;   // [batch, n_heads, seq, cache_len]
    Tensor* attn_out  = NULL;   // [batch, n_heads, seq, head_dim]
    Tensor* attn_perm = NULL;   // [batch, seq, n_heads, head_dim] (permuted view for output proj)

    VoxCPMError err = VOXCPM_SUCCESS;

    // ─────────────────────────────────────────────────────────────
    // Step 1: Create 2D flat views of x and out (non-owning)
    // ─────────────────────────────────────────────────────────────
    {
        int shape_2d[2] = { batch * seq, d_model };
        x_flat   = tensor_create_from_buffer(2, shape_2d, x->data);
        out_flat = tensor_create_from_buffer(2, shape_2d, out->data);
        if (!x_flat || !out_flat) { err = VOXCPM_ERR_OOM; goto cleanup; }
    }

    // ─────────────────────────────────────────────────────────────
    // Step 2: QKV projections  (x @ Wq, x @ Wk, x @ Wv)
    // ─────────────────────────────────────────────────────────────
    q_proj = tensor_create(2, (int[]){ batch * seq, qk_dim });
    k_proj = tensor_create(2, (int[]){ batch * seq, kv_dim });
    v_proj = tensor_create(2, (int[]){ batch * seq, kv_dim });
    if (!q_proj || !k_proj || !v_proj) { err = VOXCPM_ERR_OOM; goto cleanup; }

    // Weights are stored in PyTorch format [out_dim, in_dim], so use matmul_nt: out = x @ W^T
    err = tensor_matmul_nt(x_flat, attn->wq, q_proj);  if (err) goto cleanup;
    err = tensor_matmul_nt(x_flat, attn->wk, k_proj);  if (err) goto cleanup;
    err = tensor_matmul_nt(x_flat, attn->wv, v_proj);  if (err) goto cleanup;

    // ─────────────────────────────────────────────────────────────
    // Step 3: Reshape projections to 4D head layout
    //    q_proj: [batch*seq, d_model]  → [batch, seq, n_heads, head_dim]
    //    k_proj: [batch*seq, kv_dim]   → [batch, seq, n_kv_heads, head_dim]
    //    v_proj: [batch*seq, kv_dim]   → [batch, seq, n_kv_heads, head_dim]
    // ─────────────────────────────────────────────────────────────
    err = tensor_reshape(q_proj, 4, (int[]){ batch, seq, n_heads,   head_dim });  if (err) goto cleanup;
    err = tensor_reshape(k_proj, 4, (int[]){ batch, seq, n_kv_heads, head_dim });  if (err) goto cleanup;
    err = tensor_reshape(v_proj, 4, (int[]){ batch, seq, n_kv_heads, head_dim });  if (err) goto cleanup;

    // ─────────────────────────────────────────────────────────────
    // Step 4: Permute to [batch, n_heads, seq, head_dim] for RoPE
    //    from [batch, seq, n_heads, head_dim] → axes {0, 2, 1, 3}
    // ─────────────────────────────────────────────────────────────
    q_rope = tensor_create(4, (int[]){ batch, n_heads,   seq, head_dim });
    k_rope = tensor_create(4, (int[]){ batch, n_kv_heads, seq, head_dim });
    v_rope = tensor_create(4, (int[]){ batch, n_kv_heads, seq, head_dim });
    if (!q_rope || !k_rope || !v_rope) { err = VOXCPM_ERR_OOM; goto cleanup; }

    {
        int axes[4] = { 0, 2, 1, 3 };
        err = tensor_permute(q_proj, q_rope, axes);  if (err) goto cleanup;
        err = tensor_permute(k_proj, k_rope, axes);  if (err) goto cleanup;
        err = tensor_permute(v_proj, v_rope, axes);  if (err) goto cleanup;
    }

    // ─────────────────────────────────────────────────────────────
    // Step 5: Apply Rotary Position Embedding (in-place on q_rope, k_rope)
    // ─────────────────────────────────────────────────────────────
    if (freqs_cis) {
        err = tensor_rotary_emb(q_rope, k_rope, freqs_cis, pos);
        if (err) goto cleanup;
    }

    // ─────────────────────────────────────────────────────────────
    // Step 6: Write K, V into KV cache at position pos
    //    cache_k/v: [batch, n_kv_heads, max_seq, head_dim]
    //    k_rope:    [batch, n_kv_heads, seq, head_dim]
    // ─────────────────────────────────────────────────────────────
    {
        size_t head_bytes = (size_t)head_dim * sizeof(float);
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < n_kv_heads; h++) {
                for (int s = 0; s < seq; s++) {
                    size_t dst_off = ((size_t)b * n_kv_heads + (size_t)h) * (size_t)max_seq * (size_t)head_dim
                                   + ((size_t)pos + (size_t)s) * (size_t)head_dim;
                    size_t src_off = ((size_t)b * n_kv_heads + (size_t)h) * (size_t)seq * (size_t)head_dim
                                   + (size_t)s * (size_t)head_dim;

                    memcpy(cache_k->data + dst_off, k_rope->data + src_off, head_bytes);
                    memcpy(cache_v->data + dst_off, v_rope->data + src_off, head_bytes);
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Step 7: Compute attention scores  Q @ K^T  with GQA
    //    scores[b, h, s, t] = sum_d( Q[b,h,s,d] * K_cache[b, kv_h, t, d] ) / sqrt(head_dim)
    //    kv_h = h / group_size
    // ─────────────────────────────────────────────────────────────
    scores = tensor_create(4, (int[]){ batch, n_heads, seq, cache_len });
    if (!scores) { err = VOXCPM_ERR_OOM; goto cleanup; }

    {
        float inv_scale = 1.0f / sqrtf((float)head_dim);
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < n_heads; h++) {
                int kv_h = h / group_size;

                // Precompute base offsets for q_rope and cache_k
                size_t q_base = ((size_t)b * n_heads + (size_t)h) * (size_t)seq * (size_t)head_dim;
                size_t k_base = ((size_t)b * n_kv_heads + (size_t)kv_h) * (size_t)max_seq * (size_t)head_dim;

                for (int s = 0; s < seq; s++) {
                    size_t q_s_off = q_base + (size_t)s * (size_t)head_dim;

                    // Prefetch q_s row (head_dim floats)
                    const float* q_row = q_rope->data + q_s_off;

                    for (int t = 0; t < cache_len; t++) {
                        size_t k_t_off = k_base + (size_t)t * (size_t)head_dim;
                        const float* k_row = cache_k->data + k_t_off;

                        // Inner product
                        float sum = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            sum += q_row[d] * k_row[d];
                        }
                        sum *= inv_scale;

                        // Apply causal mask (if provided and within bounds)
                        if (mask != NULL) {
                            if ((size_t)s < (size_t)mask->shape[0] &&
                                (size_t)t < (size_t)mask->shape[1]) {
                                sum += mask->data[(size_t)s * (size_t)mask->shape[1] + (size_t)t];
                            }
                        }

                        // Store score
                        size_t score_off = ((size_t)b * n_heads + (size_t)h) * (size_t)seq * (size_t)cache_len
                                         + (size_t)s * (size_t)cache_len + (size_t)t;
                        scores->data[score_off] = sum;
                    }
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Step 8: Softmax over the last dimension (cache_len)
    // ─────────────────────────────────────────────────────────────
    err = tensor_softmax(scores, -1);
    if (err) goto cleanup;

    // ─────────────────────────────────────────────────────────────
    // Step 9: Weighted sum  attn_out[b,h,s,d] = sum_t( score[b,h,s,t] * V[b,kv_h,t,d] )
    // ─────────────────────────────────────────────────────────────
    attn_out = tensor_create(4, (int[]){ batch, n_heads, seq, head_dim });
    if (!attn_out) { err = VOXCPM_ERR_OOM; goto cleanup; }

    {
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < n_heads; h++) {
                int kv_h = h / group_size;

                for (int s = 0; s < seq; s++) {
                    for (int d = 0; d < head_dim; d++) {
                        float sum = 0.0f;
                        for (int t = 0; t < cache_len; t++) {
                            size_t score_off = ((size_t)b * n_heads + (size_t)h) * (size_t)seq * (size_t)cache_len
                                             + (size_t)s * (size_t)cache_len + (size_t)t;
                            size_t v_off = ((size_t)b * n_kv_heads + (size_t)kv_h) * (size_t)max_seq * (size_t)head_dim
                                         + (size_t)t * (size_t)head_dim + (size_t)d;
                            sum += scores->data[score_off] * cache_v->data[v_off];
                        }
                        size_t a_off = ((size_t)b * n_heads + (size_t)h) * (size_t)seq * (size_t)head_dim
                                     + (size_t)s * (size_t)head_dim + (size_t)d;
                        attn_out->data[a_off] = sum;
                    }
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Step 10: Output projection  out = attn_out @ Wo
    //    attn_out: [batch, n_heads, seq, head_dim]
    //    → permute to [batch, seq, n_heads, head_dim]
    //    → reshape to [batch*seq, n_heads*head_dim]
    //    → matmul with Wo: [n_heads*head_dim, d_model]
    //    → result: [batch*seq, d_model] → write into out_flat
    // ─────────────────────────────────────────────────────────────
    attn_perm = tensor_create(4, (int[]){ batch, seq, n_heads, head_dim });
    if (!attn_perm) { err = VOXCPM_ERR_OOM; goto cleanup; }

    err = tensor_permute(attn_out, attn_perm, (int[]){ 0, 2, 1, 3 });
    if (err) goto cleanup;

    // Flatten to 2D for matmul: [batch*seq, n_heads*head_dim]
    err = tensor_reshape(attn_perm, 2, (int[]){ batch * seq, n_heads * head_dim });
    if (err) goto cleanup;

    // out_flat is already [batch*seq, d_model] — write result
    err = tensor_matmul_nt(attn_perm, attn->wo, out_flat);
    if (err) goto cleanup;

    // ─────────────────────────────────────────────────────────────
    // Cleanup
    // ─────────────────────────────────────────────────────────────
cleanup:
    tensor_free(x_flat);
    tensor_free(out_flat);
    tensor_free(q_proj);
    tensor_free(k_proj);
    tensor_free(v_proj);
    tensor_free(q_rope);
    tensor_free(k_rope);
    tensor_free(v_rope);
    tensor_free(scores);
    tensor_free(attn_out);
    tensor_free(attn_perm);
    return err;
}

VoxCPMError cross_attention_forward(
    const Attention* attn, const Tensor* x, const Tensor* enc, Tensor* out)
{
    if (!attn || !x || !enc || !out) return VOXCPM_ERR_INTERNAL;
    return VOXCPM_ERR_UNSUPPORTED;
}

/* ═══════════════════════════════════════════════════════════════
 * Transformer Block
 * ═══════════════════════════════════════════════════════════════ */

TransformerBlock* transformer_block_create(
    int d_model, int n_heads, int n_kv_heads, int d_ff, float norm_eps)
{
    TransformerBlock* block = (TransformerBlock*)calloc(1, sizeof(TransformerBlock));
    if (!block) return NULL;

    block->rms_attn = rms_norm_create(d_model, norm_eps);
    block->attn = attention_create(d_model, n_heads, n_kv_heads);
    block->rms_ffn = rms_norm_create(d_model, norm_eps);
    block->ffn = swiglu_create(d_model, d_ff);

    if (!block->rms_attn || !block->attn || !block->rms_ffn || !block->ffn) {
        transformer_block_free(block);
        return NULL;
    }

    return block;
}

void transformer_block_free(TransformerBlock* block) {
    if (!block) return;
    rms_norm_free(block->rms_attn);
    attention_free(block->attn);
    rms_norm_free(block->rms_ffn);
    swiglu_free(block->ffn);
    free(block);
}

VoxCPMError transformer_block_forward(
    const TransformerBlock* block, const Tensor* x,
    Tensor* cache_k, Tensor* cache_v, const Tensor* mask,
    const Tensor* freqs_cis, int pos, Tensor* out)
{
    LOG_INFO("TRACE tbf ENTER");
    // ─────────────────────────────────────────────────────────────
    // Input validation
    // ─────────────────────────────────────────────────────────────
    if (!block || !x || !out)    { LOG_ERROR("tbf: null input"); return VOXCPM_ERR_INTERNAL; }
    if (x->ndim   != 3)          { LOG_ERROR("tbf: x ndim=%d", x->ndim); return VOXCPM_ERR_SHAPE_MISMATCH; }
    if (out->ndim != 3)          { LOG_ERROR("tbf: out ndim=%d", out->ndim); return VOXCPM_ERR_SHAPE_MISMATCH; }
    if (x->shape[0] != out->shape[0] ||
        x->shape[1] != out->shape[1] ||
        x->shape[2] != out->shape[2]) {
        LOG_ERROR("tbf: shape mismatch x=[%d,%d,%d] out=[%d,%d,%d]",
                  x->shape[0], x->shape[1], x->shape[2],
                  out->shape[0], out->shape[1], out->shape[2]);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    // Defensive check: all sub-modules must be present
    if (!block->rms_attn || !block->attn || !block->rms_ffn || !block->ffn) {
        LOG_ERROR("tbf: null submodule");
        return VOXCPM_ERR_INTERNAL;
    }

    int batch   = x->shape[0];
    int seq     = x->shape[1];
    int d_model = x->shape[2];

    // ─────────────────────────────────────────────────────────────
    // Temporary tensors
    //    normed:   reused for both pre-attn and pre-ffn norm output
    //    temp_out: reused for attention and swiglu output
    // ─────────────────────────────────────────────────────────────
    Tensor* normed   = NULL;
    Tensor* temp_out = NULL;
    VoxCPMError err  = VOXCPM_SUCCESS;

    normed   = tensor_create(3, (int[]){ batch, seq, d_model });
    temp_out = tensor_create(3, (int[]){ batch, seq, d_model });
    if (!normed || !temp_out) {
        err = VOXCPM_ERR_OOM;
        LOG_ERROR("transformer_block_forward: OOM for temp tensors "
                  "(%d x %d x %d)", batch, seq, d_model);
        goto cleanup;
    }

    // ─────────────────────────────────────────────────────────────
    // Step 1-2: Pre-attention RMS norm
    //   normed = RMSNorm(x)
    // ─────────────────────────────────────────────────────────────
    err = rms_norm_forward(block->rms_attn, x, normed);
    if (err) {
        LOG_ERROR("transformer_block_forward: rms_norm_forward (attn) failed");
        goto cleanup;
    }
    {float s=0;int n=0;for(int i=0;i<100;i++){float v=normed->data[i];s+=fabsf(v);if(isnan(v))n++;} LOG_INFO("tbf N01 rms_attn: sum=%.2f NaN=%d",s,n);}

    // ─────────────────────────────────────────────────────────────
    // Step 3: Multi-head attention
    //   temp_out = Attention(normed, cache_k, cache_v, mask, freqs_cis, pos)
    // ─────────────────────────────────────────────────────────────
    err = attention_forward(block->attn, normed, cache_k, cache_v,
                            mask, freqs_cis, pos, temp_out);
    if (err) {
        LOG_ERROR("transformer_block_forward: attention_forward failed");
        goto cleanup;
    }
    {float s=0;int n=0;for(int i=0;i<100;i++){float v=temp_out->data[i];s+=fabsf(v);if(isnan(v))n++;} LOG_INFO("tbf N02 attn_out: sum=%.2f NaN=%d",s,n);}

    // ─────────────────────────────────────────────────────────────
    // Step 4: Residual connection (pre-attention)
    //   out = x + temp_out
    //
    //   Safe for in-place (out == x): tensor_add reads a[i] before
    //   writing out[i]; each element is read once before overwrite.
    // ─────────────────────────────────────────────────────────────
    err = tensor_add(x, temp_out, out);
    if (err) {
        LOG_ERROR("transformer_block_forward: tensor_add (attn residual) failed");
        goto cleanup;
    }
    {float s=0;int n=0;for(int i=0;i<100;i++){float v=out->data[i];s+=fabsf(v);if(isnan(v))n++;} LOG_INFO("tbf N03 attn_res: sum=%.2f NaN=%d",s,n);}

    // ─────────────────────────────────────────────────────────────
    // Step 5-6: Pre-FFN RMS norm
    //   normed = RMSNorm(out)
    // ─────────────────────────────────────────────────────────────
    err = rms_norm_forward(block->rms_ffn, out, normed);
    if (err) {
        LOG_ERROR("transformer_block_forward: rms_norm_forward (ffn) failed");
        goto cleanup;
    }
    {float s=0;int n=0;for(int i=0;i<100;i++){float v=normed->data[i];s+=fabsf(v);if(isnan(v))n++;} LOG_INFO("tbf N04 rms_ffn: sum=%.2f NaN=%d",s,n);}

    // ─────────────────────────────────────────────────────────────
    // Step 7: SwiGLU FFN
    //   temp_out = SwiGLU(normed)
    // ─────────────────────────────────────────────────────────────
    err = swiglu_forward(block->ffn, normed, temp_out);
    if (err) {
        LOG_ERROR("transformer_block_forward: swiglu_forward failed");
        goto cleanup;
    }
    {float s=0;int n=0;for(int i=0;i<100;i++){float v=temp_out->data[i];s+=fabsf(v);if(isnan(v))n++;} LOG_INFO("tbf N05 swiglu: sum=%.2f NaN=%d",s,n);}

    // ─────────────────────────────────────────────────────────────
    // Step 8: Residual connection (pre-FFN)
    //   out = out + temp_out
    // ─────────────────────────────────────────────────────────────
    err = tensor_add(out, temp_out, out);
    if (err) {
        LOG_ERROR("transformer_block_forward: tensor_add (ffn residual) failed");
        goto cleanup;
    }
    {float s=0;int n=0;for(int i=0;i<100;i++){float v=out->data[i];s+=fabsf(v);if(isnan(v))n++;} LOG_INFO("tbf N06 ffn_res: sum=%.2f NaN=%d",s,n);}

cleanup:
    tensor_free(normed);
    tensor_free(temp_out);
    return err;
}

/* ═══════════════════════════════════════════════════════════════
 * DiT Block
 * ═══════════════════════════════════════════════════════════════ */

DiTBlock* dit_block_create(
    int d_model, int n_heads, int n_kv_heads, int d_ff, float norm_eps)
{
    DiTBlock* block = (DiTBlock*)calloc(1, sizeof(DiTBlock));
    if (!block) return NULL;

    block->rms_attn = rms_norm_create(d_model, norm_eps);
    block->attn = attention_create(d_model, n_heads, n_kv_heads);
    block->rms_cross = rms_norm_create(d_model, norm_eps);
    block->cross_attn = attention_create(d_model, n_heads, n_kv_heads);
    block->rms_ffn = rms_norm_create(d_model, norm_eps);
    block->ffn = swiglu_create(d_model, d_ff);

    return block;
}

void dit_block_free(DiTBlock* block) {
    if (!block) return;
    rms_norm_free(block->rms_attn);
    attention_free(block->attn);
    rms_norm_free(block->rms_cross);
    attention_free(block->cross_attn);
    rms_norm_free(block->rms_ffn);
    swiglu_free(block->ffn);
    free(block);
}

VoxCPMError dit_block_forward(
    const DiTBlock* block, const Tensor* x, const Tensor* cond,
    const Tensor* adaLN_params, Tensor* out)
{
    if (!block || !x || !out) return VOXCPM_ERR_INTERNAL;
    (void)cond; (void)adaLN_params;
    return VOXCPM_ERR_UNSUPPORTED;
}

/* ═══════════════════════════════════════════════════════════════
 * Conv1D
 * ═══════════════════════════════════════════════════════════════ */

Conv1D* conv1d_create(int in_channels, int out_channels,
                       int kernel_size, int stride, int padding)
{
    Conv1D* conv = (Conv1D*)calloc(1, sizeof(Conv1D));
    if (!conv) return NULL;

    conv->in_channels = in_channels;
    conv->out_channels = out_channels;
    conv->kernel_size = kernel_size;
    conv->stride = stride;
    conv->padding = padding;
    conv->groups = 1;
    conv->weight = NULL;
    conv->bias = NULL;
    return conv;
}

void conv1d_free(Conv1D* conv) {
    if (!conv) return;
    tensor_free(conv->weight);
    tensor_free(conv->bias);
    free(conv);
}

VoxCPMError conv1d_forward(const Conv1D* conv, const Tensor* x, Tensor* out) {
    if (!conv || !x || !out) { LOG_ERROR("conv1d C01: NULL"); return VOXCPM_ERR_INTERNAL; }
    if (!conv->weight) { LOG_ERROR("conv1d C02: no weight"); return VOXCPM_ERR_INTERNAL; }
    if (x->ndim != 3) { LOG_ERROR("conv1d C03: x->ndim=%d", x->ndim); return VOXCPM_ERR_SHAPE_MISMATCH; }
    if (out->ndim != 3) { LOG_ERROR("conv1d C04: out->ndim=%d", out->ndim); return VOXCPM_ERR_SHAPE_MISMATCH; }

    int batch        = x->shape[0];
    int in_channels  = x->shape[1];
    int in_time      = x->shape[2];
    int out_channels = conv->out_channels;
    int kernel_size  = conv->kernel_size;
    int stride       = conv->stride;
    int padding      = conv->padding;
    int groups       = conv->groups;

    if (in_channels != conv->in_channels) {
        LOG_ERROR("conv1d C05: in_channels=%d != conv->in_channels=%d", in_channels, conv->in_channels);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    int out_time = (in_time + 2 * padding - kernel_size) / stride + 1;
    if (out_time <= 0) {
        LOG_ERROR("conv1d C06: out_time=%d (in_time=%d pad=%d kernel=%d stride=%d)",
                  out_time, in_time, padding, kernel_size, stride);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (out->shape[0] != batch || out->shape[1] != out_channels || out->shape[2] != out_time) {
        LOG_ERROR("conv1d C07: out [%d,%d,%d] != expected [%d,%d,%d]",
                  out->shape[0], out->shape[1], out->shape[2],
                  batch, out_channels, out_time);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    /* For grouped/depthwise conv:
     *   weight shape: [out_channels, in_channels/groups, kernel_size]
     *   Each output channel o in group g connects to input channels
     *     [g * in_per_group, (g+1) * in_per_group) where g = o / out_per_group */
    const float* w = conv->weight->data;
    const float* b = conv->bias ? conv->bias->data : NULL;
    int in_per_group = (groups > 0) ? (in_channels / groups) : in_channels;
    int out_per_group = (groups > 0) ? (out_channels / groups) : out_channels;

    for (int n = 0; n < batch; n++) {
        for (int o = 0; o < out_channels; o++) {
            /* Determine which group this output channel belongs to */
            int group_id = (out_per_group > 0) ? (o / out_per_group) : 0;
            int c_start = group_id * in_per_group;
            int c_end   = c_start + in_per_group;

            for (int t = 0; t < out_time; t++) {
                float sum = b ? b[o] : 0.0f;
                for (int c = c_start; c < c_end; c++) {
                    for (int k = 0; k < kernel_size; k++) {
                        int in_t = t * stride + k - padding;
                        if (in_t < 0 || in_t >= in_time) continue;
                        /* weight index: group-local input channel */
                        int c_local = c - c_start;
                        size_t w_idx = ((size_t)o * in_per_group + (size_t)c_local) * kernel_size + (size_t)k;
                        size_t x_idx = ((size_t)n * in_channels + (size_t)c) * in_time + (size_t)in_t;
                        sum += w[w_idx] * x->data[x_idx];
                    }
                }
                size_t out_idx = ((size_t)n * out_channels + (size_t)o) * out_time + (size_t)t;
                out->data[out_idx] = sum;
            }
        }
    }

    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * Conv1D Transpose
 * ═══════════════════════════════════════════════════════════════ */

Conv1DTranspose* conv1d_transpose_create(
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding)
{
    Conv1DTranspose* conv = (Conv1DTranspose*)calloc(1, sizeof(Conv1DTranspose));
    if (!conv) return NULL;

    conv->in_channels = in_channels;
    conv->out_channels = out_channels;
    conv->kernel_size = kernel_size;
    conv->stride = stride;
    conv->padding = padding;
    conv->weight = NULL;
    conv->bias = NULL;
    return conv;
}

void conv1d_transpose_free(Conv1DTranspose* conv) {
    if (!conv) return;
    tensor_free(conv->weight);
    tensor_free(conv->bias);
    free(conv);
}

VoxCPMError conv1d_transpose_forward(const Conv1DTranspose* conv, const Tensor* x, Tensor* out) {
    if (!conv || !x || !out) { LOG_ERROR("conv1d_t T01: NULL"); return VOXCPM_ERR_INTERNAL; }
    if (!conv->weight) { LOG_ERROR("conv1d_t T02: no weight"); return VOXCPM_ERR_INTERNAL; }
    if (x->ndim != 3) { LOG_ERROR("conv1d_t T03: x->ndim=%d", x->ndim); return VOXCPM_ERR_SHAPE_MISMATCH; }
    if (out->ndim != 3) { LOG_ERROR("conv1d_t T04: out->ndim=%d", out->ndim); return VOXCPM_ERR_SHAPE_MISMATCH; }

    int batch        = x->shape[0];
    int in_channels  = x->shape[1];
    int in_time      = x->shape[2];
    int out_channels = conv->out_channels;
    int kernel_size  = conv->kernel_size;
    int stride       = conv->stride;
    int padding      = conv->padding;

    if (in_channels != conv->in_channels) {
        LOG_ERROR("conv1d_t T05: in_channels=%d != conv->in_channels=%d", in_channels, conv->in_channels);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    /* out_time = (in_time - 1) * stride - 2 * padding + kernel_size */
    int out_time = (in_time - 1) * stride - 2 * padding + kernel_size;
    if (out_time <= 0) {
        LOG_ERROR("conv1d_t T06: out_time=%d (in_time=%d stride=%d pad=%d kernel=%d)",
                  out_time, in_time, stride, padding, kernel_size);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (out->shape[0] != batch || out->shape[1] != out_channels || out->shape[2] != out_time) {
        LOG_ERROR("conv1d_t T07: out [%d,%d,%d] != expected [%d,%d,%d]",
                  out->shape[0], out->shape[1], out->shape[2],
                  batch, out_channels, out_time);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    /* Zero output */
    memset(out->data, 0, out->size * sizeof(float));

    const float* w = conv->weight->data;   /* [in_channels, out_channels, kernel_size] */
    const float* b = conv->bias ? conv->bias->data : NULL;

    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < in_channels; c++) {
            for (int t_in = 0; t_in < in_time; t_in++) {
                float x_val = x->data[((size_t)n * in_channels + (size_t)c) * in_time + (size_t)t_in];
                for (int k = 0; k < kernel_size; k++) {
                    int out_t = t_in * stride + k - padding;
                    if (out_t < 0 || out_t >= out_time) continue;
                    for (int o = 0; o < out_channels; o++) {
                        size_t w_idx = ((size_t)c * out_channels + (size_t)o) * kernel_size + (size_t)k;
                        size_t out_idx = ((size_t)n * out_channels + (size_t)o) * out_time + (size_t)out_t;
                        out->data[out_idx] += w[w_idx] * x_val;
                    }
                }
            }
        }

        /* Add bias */
        if (b) {
            for (int o = 0; o < out_channels; o++) {
                for (int t = 0; t < out_time; t++) {
                    size_t idx = ((size_t)n * out_channels + (size_t)o) * out_time + (size_t)t;
                    out->data[idx] += b[o];
                }
            }
        }
    }

    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * Snake activation
 * ═══════════════════════════════════════════════════════════════ */

VoxCPMError tensor_snake(Tensor* t, float beta) {
    if (!t || !t->data) return VOXCPM_ERR_INTERNAL;
    // snake(x) = x + (1/beta) * sin^2(beta * x)
    for (size_t i = 0; i < t->size; i++) {
        float x = t->data[i];
        float sin_val = sinf(beta * x);
        t->data[i] = x + (1.0f / beta) * sin_val * sin_val;
    }
    return VOXCPM_SUCCESS;
}

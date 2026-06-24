// ralm.c — RALM forward pass implementation
// VoxCPM2-C Project
// License: Apache-2.0
//
// Phase 2: ralm_forward (prefill) and ralm_forward_step (decode).
//
// Architecture:
//   RALM (Reference-Aware Language Model) is an 8-layer residual autoregressive
//   transformer decoder, identical to TSLM but with no RoPE positional encoding
//   and no token embedding (x is already embedded on input).
//
//   ralm_forward:      Full sequence prefill (or continuation). Processes
//                      [batch, seq, d_model] through all layers, applies
//                      output RMS norm, and updates KV cache.
//
//   ralm_forward_step: Single-token decode. Uses existing KV cache to
//                      compute one output token. No causal mask needed
//                      (seq=1 attention is trivially causal).

#include "model.h"
#include "platform.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 *  Internal: Create a 4D non-owning view into a layer's KV cache slice.
 *
 *  The RALM cache_k / cache_v is 5D:
 *    [n_layers, batch, n_kv_heads, max_seq_len, head_dim]
 *
 *  For layer i, the 4D view points to:
 *    cache->data + i * batch * n_kv_heads * max_seq_len * head_dim
 *    with shape [batch, n_kv_heads, max_seq_len, head_dim]
 * ═══════════════════════════════════════════════════════════════ */
static Tensor* layer_cache_slice(Tensor* cache_5d, int layer_idx,
                                  int batch, int n_kv_heads,
                                  int max_seq_len, int head_dim)
{
    if (!cache_5d) return NULL;

    size_t layer_size = (size_t)batch * (size_t)n_kv_heads *
                        (size_t)max_seq_len * (size_t)head_dim;
    float* data = cache_5d->data + (size_t)layer_idx * layer_size;

    int shape_4d[4] = { batch, n_kv_heads, max_seq_len, head_dim };
    return tensor_create_from_buffer(4, shape_4d, data);
}

/* ═══════════════════════════════════════════════════════════════
 *  Internal: Create a causal attention mask for prefill/continuation.
 *
 *  Shape: [seq, cache_len + seq]
 *  Value: 0.0f for visible positions, -INFINITY for masked positions.
 *
 *  A query at position s (0-indexed within the current batch) can attend
 *  to all cached tokens t ∈ [0, cache_len) and to tokens
 *  t ∈ [cache_len, cache_len + s] within the current sequence.
 *  Future tokens (t > cache_len + s) are masked.
 * ═══════════════════════════════════════════════════════════════ */
static Tensor* create_causal_mask(int seq, int cache_len) {
    if (seq <= 0) return NULL;

    int total_len = seq + cache_len;
    int shape[2] = { seq, total_len };
    Tensor* mask = tensor_create(2, shape);
    if (!mask) return NULL;

    for (int s = 0; s < seq; s++) {
        int visible_until = cache_len + s;
        for (int t = 0; t < total_len; t++) {
            mask->data[(size_t)s * (size_t)total_len + (size_t)t] =
                (t <= visible_until) ? 0.0f : -INFINITY;
        }
    }
    return mask;
}

/* ═══════════════════════════════════════════════════════════════
 *  ralm_setup_cache — Allocate KV cache for RALM.
 *
 *  Allocates 5D tensors for K and V cache:
 *    [n_layers, batch, n_kv_heads, max_seq_len, head_dim]
 *
 *  Any existing cache is freed first.
 * ═══════════════════════════════════════════════════════════════ */
VoxCPMError ralm_setup_cache(RALM* ralm, int batch_size) {
    if (!ralm) return VOXCPM_ERR_INTERNAL;

    // Free any existing cache
    tensor_free(ralm->cache_k);
    tensor_free(ralm->cache_v);
    ralm->cache_k = NULL;
    ralm->cache_v = NULL;

    int shape[5] = {
        ralm->n_layers,
        batch_size,
        ralm->n_kv_heads,
        ralm->max_seq_len,
        ralm->head_dim
    };

    ralm->cache_k = tensor_create(5, shape);
    ralm->cache_v = tensor_create(5, shape);
    if (!ralm->cache_k || !ralm->cache_v) {
        tensor_free(ralm->cache_k);
        tensor_free(ralm->cache_v);
        ralm->cache_k = NULL;
        ralm->cache_v = NULL;
        return VOXCPM_ERR_OOM;
    }

    ralm->cache_len = 0;
    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 *  ralm_cache_clear — Reset KV cache length to zero.
 *
 *  Does not free the cache buffers; just resets the write position
 *  so the next forward pass overwrites from the start.
 * ═══════════════════════════════════════════════════════════════ */
void ralm_cache_clear(RALM* ralm) {
    if (ralm) {
        ralm->cache_len = 0;
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  ralm_forward — Full sequence prefill (or continuation).
 *
 *  Processes a batch of input sequences through all RALM layers.
 *  If no KV cache exists yet (first call), it is allocated automatically.
 *  The KV cache is written at position ralm->cache_len, allowing
 *  continuation from a previous prefill.
 *
 *  Key differences from tslm_forward:
 *    1. No token embedding — x is already embedded
 *    2. no_rope=true — freqs_cis is ignored (NULL passed to attention)
 *    3. 8 layers instead of 28
 *
 *  Input:
 *    x:         [batch, seq, d_model]     — input embeddings
 *    freqs_cis: [max_seq, head_dim/2, 2]  — precomputed RoPE (ignored if no_rope)
 *    out:       [batch, seq, d_model]     — output hidden states
 *
 *  After forward, ralm->cache_len is incremented by seq.
 * ═══════════════════════════════════════════════════════════════ */
VoxCPMError ralm_forward(RALM* ralm, const Tensor* x,
                          const Tensor* freqs_cis, Tensor* out)
{
    // ─── Input validation ─────────────────────────────────────────
    if (!ralm || !x || !out) {
        LOG_ERROR("ralm_forward: NULL input");
        return VOXCPM_ERR_INTERNAL;
    }

    if (x->ndim != 3 || out->ndim != 3) {
        LOG_ERROR("ralm_forward: expected 3D tensors, got x.ndim=%d out.ndim=%d",
                  x->ndim, out->ndim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    int batch   = x->shape[0];
    int seq     = x->shape[1];
    int d_model = x->shape[2];

    if (d_model != ralm->d_model) {
        LOG_ERROR("ralm_forward: d_model mismatch: x=%d, ralm=%d",
                  d_model, ralm->d_model);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (out->shape[0] != batch || out->shape[1] != seq ||
        out->shape[2] != d_model) {
        LOG_ERROR("ralm_forward: output shape mismatch");
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    if (seq <= 0) {
        LOG_ERROR("ralm_forward: sequence length must be > 0, got %d", seq);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    // ─── Set up KV cache if not already allocated ─────────────────
    if (!ralm->cache_k || !ralm->cache_v) {
        VoxCPMError err = ralm_setup_cache(ralm, batch);
        if (err != VOXCPM_SUCCESS) {
            LOG_ERROR("ralm_forward: failed to set up KV cache");
            return err;
        }
    }

    // Validate cache batch matches input batch
    if (ralm->cache_k->shape[1] != batch) {
        LOG_ERROR("ralm_forward: cache batch mismatch, "
                  "cache has %d, input has %d",
                  ralm->cache_k->shape[1], batch);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    int n_layers    = ralm->n_layers;
    int n_kv_heads  = ralm->n_kv_heads;
    int max_seq_len = ralm->max_seq_len;
    int head_dim    = ralm->head_dim;
    int pos         = ralm->cache_len;

    // Check total length doesn't exceed max_seq_len
    if (pos + seq > max_seq_len) {
        LOG_ERROR("ralm_forward: sequence too long: pos=%d + seq=%d > max=%d",
                  pos, seq, max_seq_len);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    // ─── Temporary buffers (two alternating work buffers) ─────────
    Tensor* buf1 = tensor_clone(x);     // owned copy of input
    Tensor* buf2 = tensor_create(3, (int[]){ batch, seq, d_model });
    if (!buf1 || !buf2) {
        tensor_free(buf1);
        tensor_free(buf2);
        LOG_ERROR("ralm_forward: OOM for work buffers");
        return VOXCPM_ERR_OOM;
    }

    // ─── Causal mask (only needed when seq > 1) ───────────────────
    Tensor* mask = NULL;
    if (seq > 1) {
        mask = create_causal_mask(seq, pos);
        if (!mask) {
            tensor_free(buf1);
            tensor_free(buf2);
            LOG_ERROR("ralm_forward: OOM for causal mask");
            return VOXCPM_ERR_OOM;
        }
    }

    // When no_rope is true, pass NULL to attention_forward so RoPE is skipped.
    // (attention_forward checks freqs_cis for NULL before applying.)
    const Tensor* rope = ralm->no_rope ? NULL : freqs_cis;

    VoxCPMError err   = VOXCPM_SUCCESS;
    Tensor*     cur   = buf1;
    Tensor*     nxt   = buf2;

    // ─── Layer loop ───────────────────────────────────────────────
    for (int i = 0; i < n_layers; i++) {
        // Create non-owning 4D cache slice views for this layer
        Tensor* layer_k = layer_cache_slice(
            ralm->cache_k, i, batch, n_kv_heads, max_seq_len, head_dim);
        Tensor* layer_v = layer_cache_slice(
            ralm->cache_v, i, batch, n_kv_heads, max_seq_len, head_dim);

        if (!layer_k || !layer_v) {
            tensor_free(layer_k);
            tensor_free(layer_v);
            err = VOXCPM_ERR_OOM;
            LOG_ERROR("ralm_forward: OOM for layer %d cache slice", i);
            break;
        }

        // Forward through the transformer block
        err = transformer_block_forward(
            &ralm->layers[i], cur, layer_k, layer_v,
            mask, rope, pos, nxt);

        // Free cache slice views (non-owning; does not free cache data)
        tensor_free(layer_k);
        tensor_free(layer_v);

        if (err != VOXCPM_SUCCESS) {
            LOG_ERROR("ralm_forward: transformer_block_forward "
                      "failed at layer %d", i);
            break;
        }

        // Swap buffers for next layer
        Tensor* tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    // ─── Final output norm ────────────────────────────────────────
    if (err == VOXCPM_SUCCESS) {
        // cur now holds the final hidden state after all layers
        err = rms_norm_forward(ralm->output_norm, cur, out);
        if (err != VOXCPM_SUCCESS) {
            LOG_ERROR("ralm_forward: final rms_norm_forward failed");
        } else {
            // Update cache length only on success
            ralm->cache_len = pos + seq;
        }
    }

    // ─── Cleanup ─────────────────────────────────────────────────
    tensor_free(buf1);
    tensor_free(buf2);
    tensor_free(mask);

    return err;
}

/* ═══════════════════════════════════════════════════════════════
 *  ralm_forward_step — Single-token decode with KV cache.
 *
 *  Processes one new token using the existing KV cache (append mode).
 *  No causal mask is needed because a single query attends to all cached
 *  keys (trivially causal).
 *
 *  Input:
 *    x:               [batch, d_model]           — single token embedding
 *    position_id:     absolute position of this token
 *    freqs_cis_single:[1, head_dim/2, 2]         — RoPE (ignored if no_rope)
 *    out:             [batch, d_model]           — output hidden state
 *
 *  After forward_step, ralm->cache_len is set to position_id + 1.
 * ═══════════════════════════════════════════════════════════════ */
VoxCPMError ralm_forward_step(RALM* ralm, const Tensor* x,
                               int position_id,
                               const Tensor* freqs_cis_single,
                               Tensor* out)
{
    // ─── Input validation ─────────────────────────────────────────
    if (!ralm || !x || !out) {
        LOG_ERROR("ralm_forward_step: NULL input");
        return VOXCPM_ERR_INTERNAL;
    }

    if (x->ndim != 2 || out->ndim != 2) {
        LOG_ERROR("ralm_forward_step: expected 2D tensors, got x.ndim=%d out.ndim=%d",
                  x->ndim, out->ndim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    int batch   = x->shape[0];
    int d_model = x->shape[1];

    if (d_model != ralm->d_model) {
        LOG_ERROR("ralm_forward_step: d_model mismatch: x=%d, ralm=%d",
                  d_model, ralm->d_model);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (out->shape[0] != batch || out->shape[1] != d_model) {
        LOG_ERROR("ralm_forward_step: output shape mismatch");
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    // Cache must already exist for step mode
    if (!ralm->cache_k || !ralm->cache_v) {
        LOG_ERROR("ralm_forward_step: KV cache not set up. "
                  "Call ralm_forward or ralm_setup_cache first.");
        return VOXCPM_ERR_INTERNAL;
    }

    if (ralm->cache_k->shape[1] != batch) {
        LOG_ERROR("ralm_forward_step: cache batch mismatch, "
                  "cache has %d, input has %d",
                  ralm->cache_k->shape[1], batch);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    if (position_id < 0 || position_id >= ralm->max_seq_len) {
        LOG_ERROR("ralm_forward_step: position_id %d out of range [0, %d)",
                  position_id, ralm->max_seq_len);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    int n_layers    = ralm->n_layers;
    int n_kv_heads  = ralm->n_kv_heads;
    int max_seq_len = ralm->max_seq_len;
    int head_dim    = ralm->head_dim;

    // ─── Create 3D views with seq=1 for attention_forward ─────────
    //   x:  [batch, d_model]  →  view: [batch, 1, d_model]
    //   out: [batch, d_model] →  view: [batch, 1, d_model]
    int view_3d[3] = { batch, 1, d_model };

    Tensor* x_3d   = tensor_create_from_buffer(3, view_3d, x->data);
    Tensor* out_3d = tensor_create_from_buffer(3, view_3d, out->data);
    if (!x_3d || !out_3d) {
        tensor_free(x_3d);
        tensor_free(out_3d);
        LOG_ERROR("ralm_forward_step: OOM for 3D views");
        return VOXCPM_ERR_OOM;
    }

    // ─── Work buffer for inter-layer communication ────────────────
    Tensor* hidden = tensor_create(3, view_3d);
    if (!hidden) {
        tensor_free(x_3d);
        tensor_free(out_3d);
        LOG_ERROR("ralm_forward_step: OOM for hidden buffer");
        return VOXCPM_ERR_OOM;
    }

    // Copy input into work buffer (first layer uses hidden as input)
    memcpy(hidden->data, x->data,
           (size_t)batch * (size_t)d_model * sizeof(float));

    // When no_rope is true, skip RoPE
    const Tensor* rope = ralm->no_rope ? NULL : freqs_cis_single;

    // ─── Layer loop ──────────────────────────────────────────────
    VoxCPMError err = VOXCPM_SUCCESS;

    for (int i = 0; i < n_layers; i++) {
        // Create non-owning 4D cache slice views for this layer
        Tensor* layer_k = layer_cache_slice(
            ralm->cache_k, i, batch, n_kv_heads, max_seq_len, head_dim);
        Tensor* layer_v = layer_cache_slice(
            ralm->cache_v, i, batch, n_kv_heads, max_seq_len, head_dim);

        if (!layer_k || !layer_v) {
            tensor_free(layer_k);
            tensor_free(layer_v);
            err = VOXCPM_ERR_OOM;
            LOG_ERROR("ralm_forward_step: OOM for layer %d cache slice", i);
            break;
        }

        // Forward through the transformer block with seq=1.
        // No causal mask needed for single-token decode.
        // Use hidden as both input and output (in-place supported).
        err = transformer_block_forward(
            &ralm->layers[i], hidden, layer_k, layer_v,
            NULL,                /* no mask for single-token decode */
            rope,
            position_id,
            hidden);             /* in-place: write back to hidden */

        tensor_free(layer_k);
        tensor_free(layer_v);

        if (err != VOXCPM_SUCCESS) {
            LOG_ERROR("ralm_forward_step: transformer_block_forward "
                      "failed at layer %d", i);
            break;
        }
    }

    // ─── Final output norm ────────────────────────────────────────
    if (err == VOXCPM_SUCCESS) {
        // Apply output RMS norm: hidden [batch, 1, d_model] -> out [batch, d_model]
        err = rms_norm_forward(ralm->output_norm, hidden, out_3d);
        if (err != VOXCPM_SUCCESS) {
            LOG_ERROR("ralm_forward_step: final rms_norm_forward failed");
        } else {
            // Update cache length on success
            ralm->cache_len = position_id + 1;
        }
    }

    // ─── Cleanup ─────────────────────────────────────────────────
    tensor_free(x_3d);
    tensor_free(out_3d);
    tensor_free(hidden);

    return err;
}

/* ═══════════════════════════════════════════════════════════════
 *  ralm_to_cuda — Upload RALM weights and KV cache to GPU.
 *
 *  Iterates through all tensor fields in the RALM structure and
 *  calls tensor_to_cuda() on each.  Handles the layer array,
 *  output norm, and KV cache tensors.
 * ═══════════════════════════════════════════════════════════════ */
VoxCPMError ralm_to_cuda(RALM* ralm) {
    if (!ralm) return VOXCPM_ERR_INTERNAL;
#ifdef VOXCPM_CUDA
    VoxCPMError err;

    /* Transformer layers */
    for (int i = 0; i < ralm->n_layers; i++) {
        err = transformer_block_to_cuda(&ralm->layers[i]);
        if (err) return err;
    }

    /* Output RMS norm */
    err = rms_norm_to_cuda(ralm->output_norm);
    if (err) return err;

    /* KV cache (may be NULL if not yet allocated) */
    if (ralm->cache_k) {
        err = tensor_to_cuda(ralm->cache_k);
        if (err) return err;
    }
    if (ralm->cache_v) {
        err = tensor_to_cuda(ralm->cache_v);
        if (err) return err;
    }

    return VOXCPM_SUCCESS;
#else
    (void)ralm;
    return VOXCPM_ERR_CUDA_NOT_FOUND;
#endif
}

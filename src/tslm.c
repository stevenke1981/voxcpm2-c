// tslm.c — TSLM forward pass implementation
// VoxCPM2-C Project
// License: Apache-2.0
//
// Phase 2: tslm_forward (prefill) and tslm_forward_step (decode).
//
// Architecture:
//   TSLM (Text-Speech Language Model) is a 28-layer MiniCPM-4 autoregressive
//   transformer decoder.  It is the core backbone of the VoxCPM2 model.
//
//   tslm_forward:      Full sequence prefill (or continuation). Processes
//                      [batch, seq, d_model] through all layers, applies
//                      output RMS norm, and updates KV cache.
//
//   tslm_forward_step: Single-token decode. Uses existing KV cache to
//                      compute one output token. No mask needed (seq=1
//                      attention is trivially causal).

#include "model.h"
#include "platform.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 *  Internal: Create a 4D non-owning view into a layer's KV cache slice.
 *
 *  The TSLM cache_k / cache_v is 5D:
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

    /* Use the cache's actual max_seq_len (shape[3]) for offset computation,
     * NOT the passed-in max_seq_len which may differ (e.g., after OOM workaround).
     * This ensures each layer's slice points to the correct memory region. */
    int actual_seq = cache_5d->shape[3];
    size_t layer_size = (size_t)batch * (size_t)n_kv_heads *
                        (size_t)actual_seq * (size_t)head_dim;
    float* data = cache_5d->data + (size_t)layer_idx * layer_size;

    /* Use the actual cache seq len for the 4D view shape, so that
     * attention_forward uses the correct cache capacity for offset calculations.
     * (passed-in max_seq_len may differ due to OOM workaround). */
    int shape_4d[4] = { batch, n_kv_heads, actual_seq, head_dim };
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
 *  tslm_forward — Full sequence prefill (or continuation).
 *
 *  Processes a batch of input sequences through all TSLM layers.
 *  If no KV cache exists yet (first call), it is allocated automatically.
 *  The KV cache is written at position tslm->cache_len, allowing
 *  continuation from a previous prefill (e.g., after audio prompt).
 *
 *  Input:
 *    x:         [batch, seq, d_model]     — input embeddings
 *    freqs_cis: [max_seq, head_dim/2, 2]  — precomputed RoPE (may be NULL)
 *    out:       [batch, seq, d_model]     — output hidden states
 *
 *  After forward, tslm->cache_len is incremented by seq.
 * ═══════════════════════════════════════════════════════════════ */
VoxCPMError tslm_forward(TSLM* tslm, const Tensor* x,
                          const Tensor* freqs_cis, Tensor* out)
{
    // ─── Input validation ─────────────────────────────────────────
    LOG_INFO("TRACE tslm_forward ENTER");
    if (!tslm || !x || !out) {
        LOG_ERROR("tslm_forward: NULL input");
        return VOXCPM_ERR_INTERNAL;
    }

    if (x->ndim != 3 || out->ndim != 3) {
        LOG_ERROR("tslm_forward: expected 3D tensors, got x.ndim=%d out.ndim=%d",
                  x->ndim, out->ndim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    int batch   = x->shape[0];
    int seq     = x->shape[1];
    int d_model = x->shape[2];

    if (d_model != tslm->d_model) {
        LOG_ERROR("tslm_forward: d_model mismatch: x=%d, tslm=%d",
                  d_model, tslm->d_model);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (out->shape[0] != batch || out->shape[1] != seq ||
        out->shape[2] != d_model) {
        LOG_ERROR("tslm_forward: output shape mismatch");
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    if (seq <= 0) {
        LOG_ERROR("tslm_forward: sequence length must be > 0, got %d", seq);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    // ─── Set up KV cache if not already allocated ─────────────────
    if (!tslm->cache_k || !tslm->cache_v) {
        VoxCPMError err = tslm_setup_cache(tslm, batch);
        if (err != VOXCPM_SUCCESS) {
            LOG_ERROR("tslm_forward: failed to set up KV cache");
            return err;
        }
    }

    // Validate cache batch matches input batch
    if (tslm->cache_k->shape[1] != batch) {
        LOG_ERROR("tslm_forward: cache batch mismatch, "
                  "cache has %d, input has %d",
                  tslm->cache_k->shape[1], batch);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    int n_layers    = tslm->n_layers;
    int n_kv_heads  = tslm->n_kv_heads;
    int max_seq_len = tslm->max_seq_len;
    int head_dim    = tslm->head_dim;
    int pos         = tslm->cache_len;

    // Check total length doesn't exceed max_seq_len
    if (pos + seq > max_seq_len) {
        LOG_ERROR("tslm_forward: sequence too long: pos=%d + seq=%d > max=%d",
                  pos, seq, max_seq_len);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    // ─── Temporary buffers (two alternating work buffers) ─────────
    Tensor* buf1 = tensor_clone(x);     // owned copy of input
    Tensor* buf2 = tensor_create(3, (int[]){ batch, seq, d_model });
    if (!buf1 || !buf2) {
        tensor_free(buf1);
        tensor_free(buf2);
        LOG_ERROR("tslm_forward: OOM for work buffers");
        return VOXCPM_ERR_OOM;
    }

    // ─── Causal mask (only needed when seq > 1; single token has
    //      trivially causal attention with no masking needed) ───────
    Tensor* mask = NULL;
    if (seq > 1) {
        mask = create_causal_mask(seq, pos);
        if (!mask) {
            tensor_free(buf1);
            tensor_free(buf2);
            LOG_ERROR("tslm_forward: OOM for causal mask");
            return VOXCPM_ERR_OOM;
        }
    }

    VoxCPMError err   = VOXCPM_SUCCESS;
    Tensor*     cur   = buf1;
    Tensor*     nxt   = buf2;

    LOG_INFO("TRACE tslm: %d layers, pos=%d, seq=%d, batch=%d, d_model=%d, max_seq_len=%d, cache_k shape=[%d,%d,%d,%d,%d]",
             n_layers, pos, seq, batch, d_model, max_seq_len,
             tslm->cache_k->shape[0], tslm->cache_k->shape[1], tslm->cache_k->shape[2],
             tslm->cache_k->shape[3], tslm->cache_k->shape[4]);

    // ─── Layer loop ───────────────────────────────────────────────
    for (int i = 0; i < n_layers; i++) {
        if (i < 4) LOG_INFO("TRACE tslm: layer loop top i=%d", i);
        // Create non-owning 4D cache slice views for this layer
        Tensor* layer_k = layer_cache_slice(
            tslm->cache_k, i, batch, n_kv_heads, max_seq_len, head_dim);
        Tensor* layer_v = layer_cache_slice(
            tslm->cache_v, i, batch, n_kv_heads, max_seq_len, head_dim);

        if (!layer_k || !layer_v) {
            tensor_free(layer_k);
            tensor_free(layer_v);
            err = VOXCPM_ERR_OOM;
            LOG_ERROR("tslm_forward: OOM for layer %d cache slice", i);
            break;
        }
        if (i < 4) {
        LOG_INFO("TRACE tslm: cache slices created for layer %d k=%p v=%p k_shape=[%d,%d,%d,%d] layer_size=%zu",
                 i, (void*)layer_k, (void*)layer_v,
                 layer_k->shape[0], layer_k->shape[1], layer_k->shape[2], layer_k->shape[3],
                 (size_t)batch * (size_t)n_kv_heads * (size_t)max_seq_len * (size_t)head_dim);
    }

        if (i < 4) LOG_INFO("TRACE tslm: entering transformer_block_forward layer %d", i);

        // Forward through the transformer block.
        // cur (input) -> nxt (output after attention + FFN + residuals)
        err = transformer_block_forward(
            &tslm->layers[i], cur, layer_k, layer_v,
            mask, freqs_cis, pos, nxt);

        LOG_INFO("TRACE tslm: layer %d returned err=%d", i, err);

        // Free cache slice views (non-owning; does not free the actual cache data)
        LOG_INFO("TRACE tslm: freeing layer %d cache slices k=%p v=%p", i, (void*)layer_k, (void*)layer_v);
        tensor_free(layer_k);
        LOG_INFO("TRACE tslm: freed layer %d k slice", i);
        tensor_free(layer_v);
        LOG_INFO("TRACE tslm: freed layer %d v slice", i);

        if (err != VOXCPM_SUCCESS) {
            LOG_ERROR("tslm_forward: transformer_block_forward "
                      "failed at layer %d", i);
            break;
        }

        // Swap buffers for next layer
        LOG_INFO("TRACE tslm: swapping buffers for layer %d cur=%p nxt=%p", i, (void*)cur, (void*)nxt);
        Tensor* tmp = cur;
        cur = nxt;
        nxt = tmp;
        LOG_INFO("TRACE tslm: swapped done for layer %d cur=%p nxt=%p", i, (void*)cur, (void*)nxt);
    }

    // ─── Final output norm ────────────────────────────────────────
    if (err == VOXCPM_SUCCESS) {
        // cur now holds the final hidden state after all layers
        err = rms_norm_forward(tslm->output_norm, cur, out);
        if (err != VOXCPM_SUCCESS) {
            LOG_ERROR("tslm_forward: final rms_norm_forward failed");
        } else {
            // Update cache length only on success
            tslm->cache_len = pos + seq;
        }
    }

    // ─── Cleanup ─────────────────────────────────────────────────
    tensor_free(buf1);
    tensor_free(buf2);
    tensor_free(mask);

    return err;
}

/* ═══════════════════════════════════════════════════════════════
 *  tslm_forward_step — Single-token decode with KV cache.
 *
 *  Processes one new token using the existing KV cache (append mode).
 *  No causal mask is needed because a single query attends to all cached
 *  keys (trivially causal).
 *
 *  Input:
 *    x:               [batch, d_model]           — single token embedding
 *    position_id:     absolute position of this token
 *    freqs_cis_single:[1, head_dim/2, 2]         — RoPE for this position
 *    out:             [batch, d_model]           — output hidden state
 *
 *  After forward_step, tslm->cache_len is incremented by 1.
 * ═══════════════════════════════════════════════════════════════ */
VoxCPMError tslm_forward_step(TSLM* tslm, const Tensor* x,
                               int position_id,
                               const Tensor* freqs_cis_single,
                               Tensor* out)
{
    // ─── Input validation ─────────────────────────────────────────
    if (!tslm || !x || !out) {
        LOG_ERROR("tslm_forward_step: NULL input");
        return VOXCPM_ERR_INTERNAL;
    }

    if (x->ndim != 2 || out->ndim != 2) {
        LOG_ERROR("tslm_forward_step: expected 2D tensors, got x.ndim=%d out.ndim=%d",
                  x->ndim, out->ndim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    int batch   = x->shape[0];
    int d_model = x->shape[1];

    if (d_model != tslm->d_model) {
        LOG_ERROR("tslm_forward_step: d_model mismatch: x=%d, tslm=%d",
                  d_model, tslm->d_model);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (out->shape[0] != batch || out->shape[1] != d_model) {
        LOG_ERROR("tslm_forward_step: output shape mismatch");
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    // Cache must already exist for step mode
    if (!tslm->cache_k || !tslm->cache_v) {
        LOG_ERROR("tslm_forward_step: KV cache not set up. "
                  "Call tslm_forward or tslm_setup_cache first.");
        return VOXCPM_ERR_INTERNAL;
    }

    if (tslm->cache_k->shape[1] != batch) {
        LOG_ERROR("tslm_forward_step: cache batch mismatch, "
                  "cache has %d, input has %d",
                  tslm->cache_k->shape[1], batch);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    if (position_id < 0 || position_id >= tslm->max_seq_len) {
        LOG_ERROR("tslm_forward_step: position_id %d out of range [0, %d)",
                  position_id, tslm->max_seq_len);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    int n_layers    = tslm->n_layers;
    int n_kv_heads  = tslm->n_kv_heads;
    int max_seq_len = tslm->max_seq_len;
    int head_dim    = tslm->head_dim;

    // ─── Create 3D views with seq=1 for attention_forward ─────────
    //   x:  [batch, d_model]  →  view: [batch, 1, d_model]
    //   out: [batch, d_model] →  view: [batch, 1, d_model]
    int view_3d[3] = { batch, 1, d_model };

    Tensor* x_3d   = tensor_create_from_buffer(3, view_3d, x->data);
    Tensor* out_3d = tensor_create_from_buffer(3, view_3d, out->data);
    if (!x_3d || !out_3d) {
        tensor_free(x_3d);
        tensor_free(out_3d);
        LOG_ERROR("tslm_forward_step: OOM for 3D views");
        return VOXCPM_ERR_OOM;
    }

    // ─── Work buffer for inter-layer communication ────────────────
    Tensor* hidden = tensor_create(3, view_3d);
    if (!hidden) {
        tensor_free(x_3d);
        tensor_free(out_3d);
        LOG_ERROR("tslm_forward_step: OOM for hidden buffer");
        return VOXCPM_ERR_OOM;
    }

    // Copy input into work buffer (first layer uses hidden as input)
    memcpy(hidden->data, x->data, (size_t)batch * (size_t)d_model * sizeof(float));

    // ─── Layer loop ──────────────────────────────────────────────
    VoxCPMError err = VOXCPM_SUCCESS;

    for (int i = 0; i < n_layers; i++) {
        // Create non-owning 4D cache slice views for this layer
        Tensor* layer_k = layer_cache_slice(
            tslm->cache_k, i, batch, n_kv_heads, max_seq_len, head_dim);
        Tensor* layer_v = layer_cache_slice(
            tslm->cache_v, i, batch, n_kv_heads, max_seq_len, head_dim);

        if (!layer_k || !layer_v) {
            tensor_free(layer_k);
            tensor_free(layer_v);
            err = VOXCPM_ERR_OOM;
            LOG_ERROR("tslm_forward_step: OOM for layer %d cache slice", i);
            break;
        }

        // Forward through the transformer block with seq=1.
        // No causal mask needed for single-token decode.
        // Use hidden as both input and output (in-place supported).
        err = transformer_block_forward(
            &tslm->layers[i], hidden, layer_k, layer_v,
            NULL,                /* no mask for single-token decode */
            freqs_cis_single,
            position_id,
            hidden);             /* in-place: write back to hidden */

        tensor_free(layer_k);
        tensor_free(layer_v);

        if (err != VOXCPM_SUCCESS) {
            LOG_ERROR("tslm_forward_step: transformer_block_forward "
                      "failed at layer %d", i);
            break;
        }
    }

    // ─── Final output norm ────────────────────────────────────────
    if (err == VOXCPM_SUCCESS) {
        // Apply output RMS norm: hidden [batch, 1, d_model] -> out [batch, d_model]
        // Use the 3D view of out as target
        err = rms_norm_forward(tslm->output_norm, hidden, out_3d);
        if (err != VOXCPM_SUCCESS) {
            LOG_ERROR("tslm_forward_step: final rms_norm_forward failed");
        } else {
            // Update cache length on success.
            // cache_len is set to position_id + 1 (the new length).
            tslm->cache_len = position_id + 1;
        }
    }

    // ─── Cleanup ─────────────────────────────────────────────────
    tensor_free(x_3d);
    tensor_free(out_3d);
    tensor_free(hidden);

    return err;
}

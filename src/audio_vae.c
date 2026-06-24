// audio_vae.c — AudioVAE V2 decoder (P2-11)
// VoxCPM2-C Project
// License: Apache-2.0
//
// Implements AudioVAE V2 decoder forward pass:
//   latent [B, T, 64] -> waveform [B, samples] at 48kHz
//
// Architecture (from openbmb/VoxCPM2 audiovae.pth):
//   conv_in     (depthwise Conv1d 64->64, k=7)     — model.0
//   proj_up     (Conv1d 64->2048, k=1)              — model.1
//   6× DecoderBlock:                                 — model.2-7
//     Snake -> ConvTranspose1D -> 3× ResidualSubBlock
//     ResidualSubBlock: Snake -> DepthwiseConv(k=7) -> Snake -> PointwiseConv(k=1)
//   final_snake_alpha                                — model.8
//   conv_out    (Conv1d 32->1, k=7)                  — model.9
//
// Weight naming in .vxcpm:
//   audio_vae.decoder.conv_in.{weight_v,weight_g,bias}
//   audio_vae.decoder.proj_up.{weight_v,weight_g,bias}
//   audio_vae.decoder.decoder_blocks.{i}.snake_alpha
//   audio_vae.decoder.decoder_blocks.{i}.convtr.{weight_v,weight_g,bias}
//   audio_vae.decoder.decoder_blocks.{i}.res_blocks.{j}.snake_alpha1
//   audio_vae.decoder.decoder_blocks.{i}.res_blocks.{j}.conv_depthwise.{weight_v,weight_g,bias}
//   audio_vae.decoder.decoder_blocks.{i}.res_blocks.{j}.snake_alpha2
//   audio_vae.decoder.decoder_blocks.{i}.res_blocks.{j}.conv_pointwise.{weight_v,weight_g,bias}
//   audio_vae.decoder.final_snake_alpha
//   audio_vae.decoder.conv_out.{weight_v,weight_g,bias}

#include "model.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════ */
#define AUDIOVAE_SNAKE_BETA 1.0f
#define AUDIOVAE_WNORM_EPS  1e-8f

/* ── Decoder architecture parameters ── */
static const int DECODER_STRIDES[AUDIOVAE_NUM_DECODER_BLOCKS]  = {8, 6, 5, 2, 2, 2};
static const int DECODER_PADDINGS[AUDIOVAE_NUM_DECODER_BLOCKS] = {4, 3, 2, 1, 1, 1};

/* ═══════════════════════════════════════════════════════════════
 * Helper: Load a weight tensor by name from the weight index
 * ═══════════════════════════════════════════════════════════════ */
static Tensor* audio_vae_load_weight(const char* name, const uint8_t* mmap_data, const WeightIndex* idx) {
    const WeightEntry* entry = weight_index_find(idx, name);
    if (!entry) {
        LOG_ERROR("Weight not found: %s", name);
        return NULL;
    }
    Tensor* t = NULL;
    VoxCPMError err = weight_load_tensor(mmap_data, entry, &t);
    if (err != VOXCPM_SUCCESS) {
        LOG_ERROR("Failed to load weight: %s", name);
        return NULL;
    }
    return t;
}

static Tensor* audio_vae_load_weight_fp32(const char* name, const uint8_t* mmap_data, const WeightIndex* idx) {
    Tensor* t = audio_vae_load_weight(name, mmap_data, idx);
    if (!t) return NULL;
    VoxCPMError err = tensor_ensure_fp32(t);
    if (err != VOXCPM_SUCCESS) {
        LOG_ERROR("Failed to convert weight to fp32: %s", name);
        tensor_free(t);
        return NULL;
    }
    return t;
}

/* ═══════════════════════════════════════════════════════════════
 * WNConv Helper: compute effective_weight from weight_v and weight_g
 *
 * For is_transpose=false (regular Conv1d):
 *   weight_v shape: [out_channels, in_channels, kernel_size]
 *   weight_g shape: [out_channels]
 *   For each output channel o:
 *     norm = sqrt(sum_{c,k} weight_v[o,c,k]^2) + eps
 *     scale = weight_g[o] / norm
 *     effective_weight[o,c,k] = weight_v[o,c,k] * scale
 *
 * For is_transpose=true (ConvTranspose1d):
 *   weight_v shape: [in_channels, out_channels, kernel_size]
 *   weight_g shape: [in_channels]   (PyTorch weight_norm dim=0 convention)
 *   For each INPUT channel c:
 *     norm = sqrt(sum_{o,k} weight_v[c,o,k]^2) + eps
 *     scale = weight_g[c] / norm
 *     effective_weight[c,o,k] = weight_v[c,o,k] * scale
 * ═══════════════════════════════════════════════════════════════ */
static VoxCPMError wnconv_compute_effective_weight(WNConv* wn) {
    if (!wn || !wn->weight_v || !wn->weight_g) return VOXCPM_ERR_INTERNAL;

    Tensor* wv = wn->weight_v;
    Tensor* wg = wn->weight_g;

    /* Ensure weights are FP32 for direct float* access */
    VoxCPMError conv_err;
    conv_err = tensor_ensure_fp32(wv);
    if (conv_err) return conv_err;
    conv_err = tensor_ensure_fp32(wg);
    if (conv_err) return conv_err;

    int out_channels, in_channels, kernel_size;

    if (wn->is_transpose) {
        /* weight_v shape: [in_channels, out_channels, kernel_size] */
        in_channels   = (wv->ndim >= 1) ? wv->shape[0] : 1;
        out_channels  = (wv->ndim >= 2) ? wv->shape[1] : 1;
        kernel_size   = (wv->ndim >= 3) ? wv->shape[2] : 1;
    } else {
        /* weight_v shape: [out_channels, in_channels, kernel_size] */
        out_channels  = (wv->ndim >= 1) ? wv->shape[0] : 1;
        in_channels   = (wv->ndim >= 2) ? wv->shape[1] : 1;
        kernel_size   = (wv->ndim >= 3) ? wv->shape[2] : 1;
    }

    /* Verify weight_g dimension:
     *   non-transpose: weight_g has out_channels elements
     *   transpose:     weight_g has in_channels  elements (PyTorch dim=0 convention) */
    int wg_channels = (int)wg->size;
    int expected = wn->is_transpose ? in_channels : out_channels;
    if (wg_channels != expected) {
        LOG_WARN("wnconv: weight_g size %d != expected %d (is_transpose=%d)",
                 wg_channels, expected, (int)wn->is_transpose);
    }

    /* Allocate effective_weight with same shape as weight_v */
    Tensor* ew = tensor_clone(wv);
    if (!ew) return VOXCPM_ERR_OOM;

    if (wn->is_transpose) {
        /* ── ConvTranspose1d: norm per INPUT channel ── */
        /* weight_g has in_channels elements (dim=0 in PyTorch weight_norm) */
        for (int c = 0; c < in_channels; c++) {
            double sum_sq = 0.0;
            for (int o = 0; o < out_channels; o++) {
                for (int k = 0; k < kernel_size; k++) {
                    size_t idx = ((size_t)c * out_channels + (size_t)o) * kernel_size + (size_t)k;
                    float v = wv->data[idx];
                    sum_sq += (double)v * (double)v;
                }
            }
            float norm = sqrtf((float)sum_sq + AUDIOVAE_WNORM_EPS);
            int g_idx = (c < wg_channels) ? c : (wg_channels - 1); /* clamp safety */
            float scale = wg->data[g_idx] / norm;

            for (int o = 0; o < out_channels; o++) {
                for (int k = 0; k < kernel_size; k++) {
                    size_t idx = ((size_t)c * out_channels + (size_t)o) * kernel_size + (size_t)k;
                    ew->data[idx] = wv->data[idx] * scale;
                }
            }
        }
    } else {
        /* ── Regular Conv1d: norm per OUTPUT channel ── */
        for (int o = 0; o < out_channels; o++) {
            double sum_sq = 0.0;
            for (int c = 0; c < in_channels; c++) {
                for (int k = 0; k < kernel_size; k++) {
                    size_t idx = ((size_t)o * in_channels + (size_t)c) * kernel_size + (size_t)k;
                    float v = wv->data[idx];
                    sum_sq += (double)v * (double)v;
                }
            }
            float norm = sqrtf((float)sum_sq + AUDIOVAE_WNORM_EPS);
            int g_idx = (o < wg_channels) ? o : (wg_channels - 1);
            float scale = wg->data[g_idx] / norm;

            for (int c = 0; c < in_channels; c++) {
                for (int k = 0; k < kernel_size; k++) {
                    size_t idx = ((size_t)o * in_channels + (size_t)c) * kernel_size + (size_t)k;
                    ew->data[idx] = wv->data[idx] * scale;
                }
            }
        }
    }

    /* Free previous effective_weight if any, assign new one */
    tensor_free(wn->effective_weight);
    wn->effective_weight = ew;

    /* Set fields from weight shapes */
    wn->in_channels  = in_channels;
    wn->out_channels = out_channels;
    wn->kernel_size  = kernel_size;

    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * WNConv Helper: Create and load a WNConv from weight index
 * ═══════════════════════════════════════════════════════════════ */
static WNConv* wnconv_create_and_load(
    const char* name_prefix,
    bool is_transpose,
    const uint8_t* mmap_data,
    const WeightIndex* idx)
{
    WNConv* wn = (WNConv*)calloc(1, sizeof(WNConv));
    if (!wn) return NULL;

    wn->is_transpose = is_transpose;
    wn->has_bias = false;
    wn->groups = 1;
    wn->effective_weight = NULL;
    wn->bias = NULL;
    wn->weight_v = NULL;
    wn->weight_g = NULL;
    wn->stride = 1;
    wn->padding = 0;

    char key[256];

    /* Load weight_v */
    snprintf(key, sizeof(key), "%s.weight_v", name_prefix);
    wn->weight_v = audio_vae_load_weight(key, mmap_data, idx);
    if (!wn->weight_v) {
        LOG_ERROR("WNConv: missing weight_v for %s", name_prefix);
        free(wn);
        return NULL;
    }

    /* Load weight_g */
    snprintf(key, sizeof(key), "%s.weight_g", name_prefix);
    wn->weight_g = audio_vae_load_weight(key, mmap_data, idx);
    if (!wn->weight_g) {
        LOG_ERROR("WNConv: missing weight_g for %s", name_prefix);
        tensor_free(wn->weight_v);
        free(wn);
        return NULL;
    }

    /* Load bias (optional) */
    snprintf(key, sizeof(key), "%s.bias", name_prefix);
    wn->bias = audio_vae_load_weight(key, mmap_data, idx);
    wn->has_bias = (wn->bias != NULL);

    /* Compute effective weight */
    VoxCPMError err = wnconv_compute_effective_weight(wn);
    if (err != VOXCPM_SUCCESS) {
        LOG_ERROR("WNConv: failed to compute effective weight for %s", name_prefix);
        tensor_free(wn->weight_v);
        tensor_free(wn->weight_g);
        tensor_free(wn->bias);
        free(wn);
        return NULL;
    }

    LOG_DEBUG("WNConv loaded: %s (%s) %d->%d k=%d",
              name_prefix, is_transpose ? "transpose" : "regular",
              wn->in_channels, wn->out_channels, wn->kernel_size);

    return wn;
}

/* ═══════════════════════════════════════════════════════════════
 * WNConv: Free
 * ═══════════════════════════════════════════════════════════════ */
static void wnconv_free(WNConv* wn) {
    if (!wn) return;
    tensor_free(wn->weight_v);
    tensor_free(wn->weight_g);
    tensor_free(wn->bias);
    tensor_free(wn->effective_weight);
    free(wn);
}

/* ═══════════════════════════════════════════════════════════════
 * WNConv Forward (uses effective_weight)
 *
 * For is_transpose=false: Conv1D forward
 * For is_transpose=true:  Conv1DTranspose forward
 * ═══════════════════════════════════════════════════════════════ */
static VoxCPMError wnconv_forward(const WNConv* wn, const Tensor* x, Tensor* out) {
    if (!wn || !x || !out) return VOXCPM_ERR_INTERNAL;
    if (!wn->effective_weight) return VOXCPM_ERR_INTERNAL;

    if (wn->is_transpose) {
        Conv1DTranspose ct;
        ct.in_channels  = wn->in_channels;
        ct.out_channels = wn->out_channels;
        ct.kernel_size  = wn->kernel_size;
        ct.stride       = wn->stride;
        ct.padding      = wn->padding;
        ct.weight       = wn->effective_weight;
        ct.bias         = wn->bias;
        return conv1d_transpose_forward(&ct, x, out);
    } else {
        Conv1D c;
        c.in_channels  = wn->in_channels;
        c.out_channels = wn->out_channels;
        c.kernel_size  = wn->kernel_size;
        c.stride       = wn->stride;
        c.padding      = wn->padding;
        c.groups       = wn->groups;
        c.weight       = wn->effective_weight;
        c.bias         = wn->bias;
        return conv1d_forward(&c, x, out);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Snake forward (in-place)
 * ═══════════════════════════════════════════════════════════════ */
static VoxCPMError snake_forward(Tensor* x, const Tensor* alpha) {
    if (!x || !alpha) return VOXCPM_ERR_INTERNAL;
    /* x shape: [B, C, T], alpha shape: [1, C, 1] */
    int channels = x->shape[1];
    int T = x->shape[2];
    int batch = x->shape[0];
    float beta = AUDIOVAE_SNAKE_BETA;

    if (alpha->size < (size_t)channels) {
        LOG_ERROR("snake_forward: alpha size=%zu < channels=%d (x=[%d,%d,%d], alpha_ndim=%d shape=[%d,%d,%d])",
                  alpha->size, channels,
                  x->shape[0], x->shape[1], x->shape[2],
                  alpha->ndim,
                  alpha->ndim > 0 ? alpha->shape[0] : 0,
                  alpha->ndim > 1 ? alpha->shape[1] : 0,
                  alpha->ndim > 2 ? alpha->shape[2] : 0);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            float a = alpha->data[c];  /* alpha is [1, C, 1] */
            for (int t = 0; t < T; t++) {
                size_t idx = ((size_t)b * channels + (size_t)c) * T + (size_t)t;
                float val = x->data[idx];
                /* Snake: x + (1/beta) * sin^2(alpha * x) = x + (sin^2(alpha * x))/beta */
                float s = sinf(a * val);
                x->data[idx] = val + (s * s) / beta;
            }
        }
    }
    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * ResidualSubBlock forward
 *
 *   input [B, C, T]
 *     1. Snake(data, alpha1)
 *     2. Depthwise Conv1d (k=7, groups=C)
 *     3. Snake(result, alpha2)
 *     4. Pointwise Conv1d (k=1)
 *     5. result = input + output_of_pointwise  (skip connection)
 * ═══════════════════════════════════════════════════════════════ */
static VoxCPMError residual_sub_block_forward(
    const AudioVAEResBlock* rb,
    Tensor** px)
{
    VoxCPMError err;
    Tensor* x = *px;  /* [B, C, T] */
    int batch = x->shape[0];
    int channels = x->shape[1];
    int T_in = x->shape[2];

    /* Clone input for skip connection */
    Tensor* skip = tensor_clone(x);
    if (!skip) return VOXCPM_ERR_OOM;

    /* Step 1: Snake(data, alpha1) — in-place */
    if (rb->snake_alpha1) {
        err = snake_forward(x, rb->snake_alpha1);
        if (err) { tensor_free(skip); return err; }
    }

    /* Step 2: Depthwise Conv1d [B, C, T] -> [B, C, T'] (k=7, pad=3, stride=1) */
    {
        int T_out = (T_in + 2 * rb->conv_depthwise->padding
                     - rb->conv_depthwise->kernel_size)
                    / rb->conv_depthwise->stride + 1;
        Tensor* dw_out = tensor_create(3, (int[]){batch, channels, T_out});
        if (!dw_out) { tensor_free(skip); return VOXCPM_ERR_OOM; }

        err = wnconv_forward(rb->conv_depthwise, x, dw_out);
        tensor_free(x);
        if (err) { tensor_free(skip); tensor_free(dw_out); return err; }
        x = dw_out;
    }

    /* Step 3: Snake(x, alpha2) — in-place */
    if (rb->snake_alpha2) {
        err = snake_forward(x, rb->snake_alpha2);
        if (err) { tensor_free(skip); tensor_free(x); return err; }
    }

    /* Step 4: Pointwise Conv1d [B, C, T'] -> [B, C, T'] (k=1, pad=0, stride=1) */
    {
        int T_mid = x->shape[2];  /* same after depthwise with stride=1 */
        Tensor* pw_out = tensor_create(3, (int[]){batch, channels, T_mid});
        if (!pw_out) { tensor_free(skip); tensor_free(x); return VOXCPM_ERR_OOM; }

        err = wnconv_forward(rb->conv_pointwise, x, pw_out);
        tensor_free(x);
        if (err) { tensor_free(skip); tensor_free(pw_out); return err; }
        x = pw_out;
    }

    /* Step 5: Skip connection */
    err = tensor_add(x, skip, x);
    tensor_free(skip);
    if (err) { tensor_free(x); return err; }

    *px = x;
    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * DecoderBlock forward
 *
 *   Input [B, C_in, T_in]
 *     1. Snake(x, snake_alpha) — in-place
 *     2. ConvTranspose1D [B, C_in, T_in] -> [B, C_out, T_out]
 *     3. For each residual sub-block (3):
 *          apply residual_sub_block_forward
 * ═══════════════════════════════════════════════════════════════ */
static VoxCPMError decoder_block_forward(
    const AudioVAEDecoderBlock* block,
    Tensor** px)
{
    VoxCPMError err;
    Tensor* x = *px;
    int batch = x->shape[0];
    int T_in  = x->shape[2];

    LOG_DEBUG("decoder_block_forward: input=[%d,%d,%d] convtr=%d->%d k=%d stride=%d pad=%d",
              x->shape[0], x->shape[1], x->shape[2],
              block->convtr ? block->convtr->in_channels : -1,
              block->convtr ? block->convtr->out_channels : -1,
              block->convtr ? block->convtr->kernel_size : -1,
              block->convtr ? block->convtr->stride : -1,
              block->convtr ? block->convtr->padding : -1);

    /* Step 1: Snake activation (in-place) */
    if (block->snake_alpha) {
        err = snake_forward(x, block->snake_alpha);
        if (err) return err;
    }

    /* Step 2: ConvTranspose1D */
    int T_out = (T_in - 1) * block->convtr->stride
                - 2 * block->convtr->padding
                + block->convtr->kernel_size;
    int out_ch = block->convtr->out_channels;

    if (T_out <= 0) return VOXCPM_ERR_SHAPE_MISMATCH;

    Tensor* conv_out = tensor_create(3, (int[]){batch, out_ch, T_out});
    if (!conv_out) return VOXCPM_ERR_OOM;

    err = wnconv_forward(block->convtr, x, conv_out);
    tensor_free(x);
    if (err) { tensor_free(conv_out); return err; }
    x = conv_out;

    LOG_DEBUG("decoder_block_forward: after convtr=[%d,%d,%d]",
              x->shape[0], x->shape[1], x->shape[2]);

    /* Step 3: Residual sub-blocks */
    for (int j = 0; j < block->num_res_blocks; j++) {
        LOG_DEBUG("decoder_block_forward: res block %d input=[%d,%d,%d]",
                  j, x->shape[0], x->shape[1], x->shape[2]);
        err = residual_sub_block_forward(&block->res_blocks[j], &x);
        if (err) return err;
    }

    *px = x;
    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * EncoderBlock forward
 *
 *   Input [B, C_in, T_in]
 *     1. 3× ResidualSubBlock (Snake→Depthwise→Snake→Pointwise + skip)
 *     2. Snake(x, stride_alpha) — in-place
 *     3. Strided Conv1D [B, C_in, T_in] -> [B, C_out, T_out]
 * ═══════════════════════════════════════════════════════════════ */
static VoxCPMError encoder_block_forward(
    const AudioVAEEncoderBlock* block,
    Tensor** px)
{
    VoxCPMError err;
    Tensor* x = *px;
    int batch   = x->shape[0];
    int C_in    = x->shape[1];
    int T_in    = x->shape[2];

    LOG_DEBUG("encoder_block_forward: input=[%d,%d,%d] stride_conv=%d->%d k=%d s=%d p=%d",
              batch, C_in, T_in,
              block->stride_conv ? block->stride_conv->in_channels : -1,
              block->stride_conv ? block->stride_conv->out_channels : -1,
              block->stride_conv ? block->stride_conv->kernel_size : -1,
              block->stride_conv ? block->stride_conv->stride : -1,
              block->stride_conv ? block->stride_conv->padding : -1);

    /* Step 1: 3× residual sub-blocks (same as decoder) */
    for (int j = 0; j < block->num_res_blocks; j++) {
        err = residual_sub_block_forward(&block->res_blocks[j], &x);
        if (err) return err;
    }

    /* Step 2: Snake activation (in-place) */
    if (block->stride_alpha) {
        err = snake_forward(x, block->stride_alpha);
        if (err) return err;
    }

    /* Step 3: Strided Conv1D (downsample) */
    {
        int T_out = (T_in + 2 * block->stride_conv->padding
                     - block->stride_conv->kernel_size)
                    / block->stride_conv->stride + 1;
        int out_ch = block->stride_conv->out_channels;

        if (T_out <= 0) return VOXCPM_ERR_SHAPE_MISMATCH;

        Tensor* conv_out = tensor_create(3, (int[]){batch, out_ch, T_out});
        if (!conv_out) return VOXCPM_ERR_OOM;

        err = wnconv_forward(block->stride_conv, x, conv_out);
        tensor_free(x);
        if (err) { tensor_free(conv_out); return err; }
        x = conv_out;
    }

    LOG_DEBUG("encoder_block_forward: output=[%d,%d,%d]",
              x->shape[0], x->shape[1], x->shape[2]);

    *px = x;
    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * AudioVAEResBlock: Free
 * ═══════════════════════════════════════════════════════════════ */
static void audio_vae_res_block_free(AudioVAEResBlock* rb) {
    if (!rb) return;
    wnconv_free(rb->conv_depthwise);
    wnconv_free(rb->conv_pointwise);
    tensor_free(rb->snake_alpha1);
    tensor_free(rb->snake_alpha2);
}

/* ═══════════════════════════════════════════════════════════════
 * AudioVAEDecoderBlock: Free
 * ═══════════════════════════════════════════════════════════════ */
static void audio_vae_decoder_block_free(AudioVAEDecoderBlock* block) {
    if (!block) return;
    wnconv_free(block->convtr);
    tensor_free(block->snake_alpha);
    for (int j = 0; j < block->num_res_blocks; j++) {
        audio_vae_res_block_free(&block->res_blocks[j]);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * AudioVAEEncoderBlock: Free
 * ═══════════════════════════════════════════════════════════════ */
static void audio_vae_encoder_block_free(AudioVAEEncoderBlock* block) {
    if (!block) return;
    wnconv_free(block->stride_conv);
    tensor_free(block->stride_alpha);
    for (int j = 0; j < block->num_res_blocks; j++) {
        audio_vae_res_block_free(&block->res_blocks[j]);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Public API — audio_vae_create
 * ═══════════════════════════════════════════════════════════════ */
AudioVAE* audio_vae_create(const VoxCPMConfig* config, const uint8_t* mmap_data, const WeightIndex* idx) {
    if (!config || !mmap_data || !idx) return NULL;

    AudioVAE* vae = (AudioVAE*)calloc(1, sizeof(AudioVAE));
    if (!vae) return NULL;

    /* Set dimensions from config */
    vae->latent_dim       = config->latent_dim;
    vae->sample_rate      = config->vae_encode_sr;
    vae->out_sample_rate  = config->sample_rate;
    vae->patch_size       = config->patch_size;
    vae->chunk_size       = 640;      /* 2 * 5 * 8 * 8 (encoder rates) */
    vae->decode_chunk_size = 1920;    /* 8 * 6 * 5 * 2 * 2 * 2 (decoder rates) */

    /* Allocate encoder (stub for Phase 3) */
    vae->encoder = (AudioVAEEncoder*)calloc(1, sizeof(AudioVAEEncoder));
    if (!vae->encoder) { free(vae); return NULL; }
    vae->encoder->encoder_dim = 128;
    vae->encoder->latent_dim = config->latent_dim;

    /* Allocate decoder */
    vae->decoder = (AudioVAEDecoder*)calloc(1, sizeof(AudioVAEDecoder));
    if (!vae->decoder) {
        free(vae->encoder);
        free(vae);
        return NULL;
    }

    AudioVAEDecoder* dec = vae->decoder;

    /* ── Load conv_in (model.0): depthwise Conv1d 64->64, k=7 ── */
    dec->conv_in = wnconv_create_and_load(
        "audio_vae.decoder.conv_in", false, mmap_data, idx);
    if (!dec->conv_in) {
        LOG_ERROR("audio_vae_create: failed to load conv_in");
        audio_vae_free(vae);
        return NULL;
    }
    dec->conv_in->stride  = 1;
    dec->conv_in->padding = 3;  /* SAME padding for k=7 */
    /* Fix depthwise: weight_v shape is [64, 1, 7], so in_channels was set to 1,
     * but the actual input has latent_dim=64 channels with groups=latent_dim. */
    dec->conv_in->in_channels = vae->latent_dim;
    dec->conv_in->groups      = vae->latent_dim;

    /* ── Load proj_up (model.1): projection 64->2048, k=1 ── */
    dec->proj_up = wnconv_create_and_load(
        "audio_vae.decoder.proj_up", false, mmap_data, idx);
    if (!dec->proj_up) {
        LOG_ERROR("audio_vae_create: failed to load proj_up");
        audio_vae_free(vae);
        return NULL;
    }
    dec->proj_up->stride  = 1;
    dec->proj_up->padding = 0;

    /* ── Load decoder blocks 0-5 (model.2-7) ── */
    for (int i = 0; i < AUDIOVAE_NUM_DECODER_BLOCKS; i++) {
        AudioVAEDecoderBlock* blk = &dec->decoder_blocks[i];
        blk->num_res_blocks = AUDIOVAE_RES_BLOCKS_PER_BLOCK;

        char key[256];

        /* Load snake_alpha (block.0.alpha) */
        snprintf(key, sizeof(key),
                 "audio_vae.decoder.decoder_blocks.%d.snake_alpha", i);
        blk->snake_alpha = audio_vae_load_weight_fp32(key, mmap_data, idx);
        if (!blk->snake_alpha) {
            LOG_ERROR("audio_vae_create: missing snake_alpha for block %d", i);
            audio_vae_free(vae);
            return NULL;
        }

        /* Load convtr (block.1) — ConvTranspose1D */
        snprintf(key, sizeof(key),
                 "audio_vae.decoder.decoder_blocks.%d.convtr", i);
        blk->convtr = wnconv_create_and_load(key, true, mmap_data, idx);
        if (!blk->convtr) {
            LOG_ERROR("audio_vae_create: failed to load convtr for block %d", i);
            audio_vae_free(vae);
            return NULL;
        }
        blk->convtr->stride  = DECODER_STRIDES[i];
        blk->convtr->padding = DECODER_PADDINGS[i];

        /* Load residual sub-blocks (block.2, block.3, block.4) */
        for (int j = 0; j < AUDIOVAE_RES_BLOCKS_PER_BLOCK; j++) {
            AudioVAEResBlock* rb = &blk->res_blocks[j];

            /* snake_alpha1 (block.{j+2}.block.0.alpha) */
            snprintf(key, sizeof(key),
                     "audio_vae.decoder.decoder_blocks.%d.res_blocks.%d.snake_alpha1", i, j);
            rb->snake_alpha1 = audio_vae_load_weight_fp32(key, mmap_data, idx);
            if (!rb->snake_alpha1) {
                LOG_ERROR("audio_vae_create: missing snake_alpha1 for block %d res %d", i, j);
                audio_vae_free(vae);
                return NULL;
            }

            /* conv_depthwise (block.{j+2}.block.1) — depthwise Conv1d, k=7 */
            snprintf(key, sizeof(key),
                     "audio_vae.decoder.decoder_blocks.%d.res_blocks.%d.conv_depthwise", i, j);
            rb->conv_depthwise = wnconv_create_and_load(key, false, mmap_data, idx);
            if (!rb->conv_depthwise) {
                LOG_ERROR("audio_vae_create: missing conv_depthwise for block %d res %d", i, j);
                audio_vae_free(vae);
                return NULL;
            }
            rb->conv_depthwise->stride  = 1;
            rb->conv_depthwise->padding = 3;  /* SAME for k=7 */
            rb->conv_depthwise->groups      = rb->conv_depthwise->out_channels; /* depthwise */
            rb->conv_depthwise->in_channels = rb->conv_depthwise->out_channels; /* fix: weight shape[1]=1 */

            /* snake_alpha2 (block.{j+2}.block.2.alpha) */
            snprintf(key, sizeof(key),
                     "audio_vae.decoder.decoder_blocks.%d.res_blocks.%d.snake_alpha2", i, j);
            rb->snake_alpha2 = audio_vae_load_weight_fp32(key, mmap_data, idx);
            if (!rb->snake_alpha2) {
                LOG_ERROR("audio_vae_create: missing snake_alpha2 for block %d res %d", i, j);
                audio_vae_free(vae);
                return NULL;
            }

            /* conv_pointwise (block.{j+2}.block.3) — pointwise Conv1d, k=1 */
            snprintf(key, sizeof(key),
                     "audio_vae.decoder.decoder_blocks.%d.res_blocks.%d.conv_pointwise", i, j);
            rb->conv_pointwise = wnconv_create_and_load(key, false, mmap_data, idx);
            if (!rb->conv_pointwise) {
                LOG_ERROR("audio_vae_create: missing conv_pointwise for block %d res %d", i, j);
                audio_vae_free(vae);
                return NULL;
            }
            rb->conv_pointwise->stride  = 1;
            rb->conv_pointwise->padding = 0;
        }
    }

    /* ── Load final_snake_alpha (model.8) ── */
    dec->final_snake_alpha = audio_vae_load_weight_fp32(
        "audio_vae.decoder.final_snake_alpha", mmap_data, idx);
    if (!dec->final_snake_alpha) {
        LOG_ERROR("audio_vae_create: missing final_snake_alpha");
        audio_vae_free(vae);
        return NULL;
    }

    /* ── Load conv_out (model.9): 32->1, k=7 ── */
    dec->conv_out = wnconv_create_and_load(
        "audio_vae.decoder.conv_out", false, mmap_data, idx);
    if (!dec->conv_out) {
        LOG_ERROR("audio_vae_create: failed to load conv_out");
        audio_vae_free(vae);
        return NULL;
    }
    dec->conv_out->stride  = 1;
    dec->conv_out->padding = 3;  /* SAME for k=7 */

    /* Set sample rate conditioning (defaults, not yet implemented) */
    dec->sr_bin_boundaries[0] = 20000;
    dec->sr_bin_boundaries[1] = 30000;
    dec->sr_bin_boundaries[2] = 40000;
    dec->sr_bin_buckets = 4;
    dec->has_sr_cond = false;
    dec->sr_cond_layers = 0;

    /* ═══════════════════════════════════════════════════════════
     * Encoder weight loading
     * ═══════════════════════════════════════════════════════════ */

    /* conv_in: audio_vae.encoder.block.0 — (1→128, k=7, stride=1, pad=3) */
    vae->encoder->conv_in = wnconv_create_and_load(
        "audio_vae.encoder.block.0", false, mmap_data, idx);
    if (!vae->encoder->conv_in) {
        LOG_ERROR("audio_vae_create: failed to load encoder conv_in");
        audio_vae_free(vae);
        return NULL;
    }
    vae->encoder->conv_in->stride  = 1;
    vae->encoder->conv_in->padding = 3;  /* SAME: k=7 */

    /* Encoder blocks 0-3: audio_vae.encoder.block.{1-4} */
    /* rates=[2,5,8,8], channels expand 128→256→512→1024→2048 */
    int enc_rates[4] = {2, 5, 8, 8};
    int enc_channels[5] = {128, 256, 512, 1024, 2048};

    for (int i = 0; i < 4; i++) {
        AudioVAEEncoderBlock* blk = &vae->encoder->blocks[i];
        blk->num_res_blocks = AUDIOVAE_RES_BLOCKS_PER_BLOCK;  /* 3 */

        char key[256];
        int C_in = enc_channels[i];

        /* Load 3 residual sub-blocks */
        for (int j = 0; j < AUDIOVAE_RES_BLOCKS_PER_BLOCK; j++) {
            AudioVAEResBlock* rb = &blk->res_blocks[j];

            /* snake_alpha1: block.{i+1}.block.{j}.block.0.alpha */
            snprintf(key, sizeof(key),
                "audio_vae.encoder.block.%d.block.%d.block.0.alpha", i+1, j);
            rb->snake_alpha1 = audio_vae_load_weight_fp32(key, mmap_data, idx);
            if (!rb->snake_alpha1) {
                LOG_ERROR("audio_vae_create: missing enc snake_alpha1 block %d sub %d", i, j);
                audio_vae_free(vae);
                return NULL;
            }

            /* conv_depthwise: block.{i+1}.block.{j}.block.1 */
            snprintf(key, sizeof(key),
                "audio_vae.encoder.block.%d.block.%d.block.1", i+1, j);
            rb->conv_depthwise = wnconv_create_and_load(key, false, mmap_data, idx);
            if (!rb->conv_depthwise) {
                LOG_ERROR("audio_vae_create: missing enc depthwise block %d sub %d", i, j);
                audio_vae_free(vae);
                return NULL;
            }
            rb->conv_depthwise->stride  = 1;
            rb->conv_depthwise->padding = 3;  /* SAME for k=7 */
            rb->conv_depthwise->groups      = C_in;
            rb->conv_depthwise->in_channels = C_in;

            /* snake_alpha2: block.{i+1}.block.{j}.block.2.alpha */
            snprintf(key, sizeof(key),
                "audio_vae.encoder.block.%d.block.%d.block.2.alpha", i+1, j);
            rb->snake_alpha2 = audio_vae_load_weight_fp32(key, mmap_data, idx);
            if (!rb->snake_alpha2) {
                LOG_ERROR("audio_vae_create: missing enc snake_alpha2 block %d sub %d", i, j);
                audio_vae_free(vae);
                return NULL;
            }

            /* conv_pointwise: block.{i+1}.block.{j}.block.3 */
            snprintf(key, sizeof(key),
                "audio_vae.encoder.block.%d.block.%d.block.3", i+1, j);
            rb->conv_pointwise = wnconv_create_and_load(key, false, mmap_data, idx);
            if (!rb->conv_pointwise) {
                LOG_ERROR("audio_vae_create: missing enc pointwise block %d sub %d", i, j);
                audio_vae_free(vae);
                return NULL;
            }
            rb->conv_pointwise->stride  = 1;
            rb->conv_pointwise->padding = 0;
        }

        /* stride_alpha: block.{i+1}.block.3.alpha */
        snprintf(key, sizeof(key),
            "audio_vae.encoder.block.%d.block.3.alpha", i+1);
        blk->stride_alpha = audio_vae_load_weight_fp32(key, mmap_data, idx);
        if (!blk->stride_alpha) {
            LOG_ERROR("audio_vae_create: missing stride_alpha for block %d", i);
            audio_vae_free(vae);
            return NULL;
        }

        /* stride_conv: block.{i+1}.block.4 */
        snprintf(key, sizeof(key),
            "audio_vae.encoder.block.%d.block.4", i+1);
        blk->stride_conv = wnconv_create_and_load(key, false, mmap_data, idx);
        if (!blk->stride_conv) {
            LOG_ERROR("audio_vae_create: missing stride_conv for block %d", i);
            audio_vae_free(vae);
            return NULL;
        }
        int kernel = 2 * enc_rates[i];
        blk->stride_conv->stride  = enc_rates[i];
        blk->stride_conv->padding = kernel / 2;  /* SAME: p = k // 2 */
    }

    /* fc_mu: audio_vae.encoder.fc_mu */
    vae->encoder->fc_mu = wnconv_create_and_load(
        "audio_vae.encoder.fc_mu", false, mmap_data, idx);
    if (!vae->encoder->fc_mu) {
        LOG_ERROR("audio_vae_create: missing fc_mu");
        audio_vae_free(vae);
        return NULL;
    }
    vae->encoder->fc_mu->stride  = 1;
    vae->encoder->fc_mu->padding = 1;  /* k=3 SAME */

    /* fc_logvar: audio_vae.encoder.fc_logvar */
    vae->encoder->fc_logvar = wnconv_create_and_load(
        "audio_vae.encoder.fc_logvar", false, mmap_data, idx);
    if (!vae->encoder->fc_logvar) {
        LOG_ERROR("audio_vae_create: missing fc_logvar");
        audio_vae_free(vae);
        return NULL;
    }
    vae->encoder->fc_logvar->stride  = 1;
    vae->encoder->fc_logvar->padding = 1;  /* k=3 SAME */

    LOG_INFO("AudioVAE V2 created: %d decoder blocks with %d res blocks each, "
             "conv_in=%d->%d proj_up=%d->%d conv_out=%d->%d",
             AUDIOVAE_NUM_DECODER_BLOCKS, AUDIOVAE_RES_BLOCKS_PER_BLOCK,
             dec->conv_in->in_channels, dec->conv_in->out_channels,
             dec->proj_up->in_channels, dec->proj_up->out_channels,
             dec->conv_out->in_channels, dec->conv_out->out_channels);

    LOG_INFO("AudioVAE encoder loaded: conv_in=%d->%d fc_mu=%d->%d fc_logvar=%d->%d",
             vae->encoder->conv_in->in_channels, vae->encoder->conv_in->out_channels,
             vae->encoder->fc_mu->in_channels, vae->encoder->fc_mu->out_channels,
             vae->encoder->fc_logvar->in_channels, vae->encoder->fc_logvar->out_channels);

    return vae;
}

/* ═══════════════════════════════════════════════════════════════
 * Public API — audio_vae_free
 * ═══════════════════════════════════════════════════════════════ */
void audio_vae_free(AudioVAE* vae) {
    if (!vae) return;

    /* Free encoder */
    if (vae->encoder) {
        wnconv_free(vae->encoder->conv_in);
        for (int i = 0; i < 4; i++) {
            audio_vae_encoder_block_free(&vae->encoder->blocks[i]);
        }
        wnconv_free(vae->encoder->fc_mu);
        wnconv_free(vae->encoder->fc_logvar);
        free(vae->encoder);
    }

    /* Free decoder */
    if (vae->decoder) {
        AudioVAEDecoder* dec = vae->decoder;

        wnconv_free(dec->conv_in);
        wnconv_free(dec->proj_up);

        for (int i = 0; i < AUDIOVAE_NUM_DECODER_BLOCKS; i++) {
            audio_vae_decoder_block_free(&dec->decoder_blocks[i]);
        }

        tensor_free(dec->final_snake_alpha);
        wnconv_free(dec->conv_out);

        free(vae->decoder);
    }

    free(vae);
}

/* ═══════════════════════════════════════════════════════════════
 * Public API — audio_vae_decode
 *
 * Decodes latent features to waveform at 48kHz.
 *
 * Input:  latent  [batch, time, latent_dim]   (16kHz-rate features)
 * Output: waveform [batch, samples]            (48kHz waveform)
 *
 * Forward pass:
 *   1. Permute latent from [B, T, D] to [B, D, T]
 *   2. conv_in:   [B, 64, T] -> [B, 64, T]   (depthwise, SAME)
 *   3. proj_up:   [B, 64, T] -> [B, 2048, T]  (projection)
 *   4. For each decoder block (6):
 *      a. Snake activation
 *      b. ConvTranspose1D (upsample by stride)
 *      c. 3× ResidualSubBlock (Snake→Depthwise→Snake→Pointwise + skip)
 *   5. Snake (final_snake_alpha)
 *   6. conv_out: [B, 32, T_out] -> [B, 1, T_out]
 *   7. Tanh activation
 *   8. Copy to [B, T_out] waveform output
 * ═══════════════════════════════════════════════════════════════ */
VoxCPMError audio_vae_decode(
    const AudioVAE* vae,
    const Tensor* latent,
    Tensor* waveform)
{
    if (!vae || !latent || !waveform) return VOXCPM_ERR_INTERNAL;
    if (!vae->decoder || !vae->decoder->conv_in || !vae->decoder->proj_up
        || !vae->decoder->conv_out)
        return VOXCPM_ERR_INTERNAL;

    /* ── Input validation ── */
    if (latent->ndim != 3) {
        LOG_ERROR("VAE V01: latent->ndim=%d != 3", latent->ndim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    int batch = latent->shape[0];
    int time  = latent->shape[1];
    int dim   = latent->shape[2];
    if (dim != vae->latent_dim) {
        LOG_ERROR("VAE V02: dim=%d != latent_dim=%d", dim, vae->latent_dim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    int wf_ndim = waveform->ndim;
    if (wf_ndim != 1 && wf_ndim != 2) {
        LOG_ERROR("VAE V03: wf_ndim=%d", wf_ndim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    int wf_batch = (wf_ndim == 2) ? waveform->shape[0] : 1;
    if (wf_batch != batch) {
        LOG_ERROR("VAE V04: wf_batch=%d != batch=%d", wf_batch, batch);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    VoxCPMError err = VOXCPM_SUCCESS;
    Tensor* h = NULL;  /* current activation tensor */

    /* Step 1: Permute latent from [B, T, D] to [B, D, T] */
    h = tensor_create(3, (int[]){batch, vae->latent_dim, time});
    if (!h) { err = VOXCPM_ERR_OOM; goto cleanup; }

    {
        int axes[3] = {0, 2, 1};  /* [B, T, D] -> [B, D, T] */
        err = tensor_permute(latent, h, axes);
        if (err) { LOG_ERROR("VAE P01: permute [%d,%d,%d] axes(0,2,1) failed",
                             latent->shape[0], latent->shape[1], latent->shape[2]);
                   goto cleanup; }
    }

    /* Step 2: conv_in — depthwise [B, 64, T] output same shape */
    {
        int T_in  = h->shape[2];
        int out_c = vae->decoder->conv_in->out_channels;
        int T_out = (T_in + 2 * vae->decoder->conv_in->padding
                     - vae->decoder->conv_in->kernel_size)
                    / vae->decoder->conv_in->stride + 1;
        Tensor* conv_out = tensor_create(3, (int[]){batch, out_c, T_out});
        if (!conv_out) { err = VOXCPM_ERR_OOM; goto cleanup; }

        err = wnconv_forward(vae->decoder->conv_in, h, conv_out);
        tensor_free(h);
        h = conv_out;
        if (err) { LOG_ERROR("VAE P02: wnconv_forward conv_in failed"); goto cleanup; }
    }

    /* Step 3: proj_up — [B, 64, T] -> [B, 2048, T] */
    {
        int T_in  = h->shape[2];
        int out_c = vae->decoder->proj_up->out_channels;
        int T_out = (T_in + 2 * vae->decoder->proj_up->padding
                     - vae->decoder->proj_up->kernel_size)
                    / vae->decoder->proj_up->stride + 1;
        Tensor* conv_out = tensor_create(3, (int[]){batch, out_c, T_out});
        if (!conv_out) { err = VOXCPM_ERR_OOM; goto cleanup; }

        err = wnconv_forward(vae->decoder->proj_up, h, conv_out);
        tensor_free(h);
        h = conv_out;
        if (err) { LOG_ERROR("VAE P03: proj_up wnconv failed"); goto cleanup; }

        LOG_DEBUG("  proj_up: [%d, %d, %d] -> [%d, %d, %d]",
                  batch, vae->decoder->conv_in->out_channels, T_in,
                  batch, out_c, T_out);
    }

    /* Step 4: Decoder blocks 0-5 */
    for (int i = 0; i < AUDIOVAE_NUM_DECODER_BLOCKS; i++) {
        AudioVAEDecoderBlock* blk = &vae->decoder->decoder_blocks[i];
        if (!blk->convtr) {
            LOG_ERROR("VAE P04: decoder block %d missing convtr", i);
            err = VOXCPM_ERR_INTERNAL;
            goto cleanup;
        }

        err = decoder_block_forward(blk, &h);
        if (err) { LOG_ERROR("VAE P05: decoder_block_forward block %d failed", i); goto cleanup; }
    }

    /* Step 5: Snake before conv_out */
    if (vae->decoder->final_snake_alpha) {
        err = snake_forward(h, vae->decoder->final_snake_alpha);
        if (err) { LOG_ERROR("VAE P06: final snake failed"); goto cleanup; }
    }

    /* Step 6: conv_out: [B, C, T_out] -> [B, 1, T_out] */
    {
        int T_in  = h->shape[2];
        int out_c = vae->decoder->conv_out->out_channels;
        int T_out = (T_in + 2 * vae->decoder->conv_out->padding
                     - vae->decoder->conv_out->kernel_size)
                    / vae->decoder->conv_out->stride + 1;
        Tensor* conv_out = tensor_create(3, (int[]){batch, out_c, T_out});
        if (!conv_out) { err = VOXCPM_ERR_OOM; goto cleanup; }

        err = wnconv_forward(vae->decoder->conv_out, h, conv_out);
        tensor_free(h);
        h = conv_out;
        if (err) { LOG_ERROR("VAE P07: conv_out wnconv failed"); goto cleanup; }
    }

    /* Step 7: Tanh */
    err = tensor_tanh(h);
    if (err) { LOG_ERROR("VAE P08: final tanh failed"); goto cleanup; }

    /* Step 8: Copy to waveform output
     *   h shape: [B, 1, T_out] -> data layout: [b, 0, t]
     */
    {
        int T_out = h->shape[2];
        size_t copy_bytes = (size_t)batch * (size_t)T_out * sizeof(float);

        if (wf_ndim == 2) {
            if (waveform->shape[1] != T_out) {
                LOG_ERROR("VAE P09: waveform shape[1]=%d != T_out=%d", waveform->shape[1], T_out);
                err = VOXCPM_ERR_SHAPE_MISMATCH;
                goto cleanup;
            }
            memcpy(waveform->data, h->data, copy_bytes);
        } else {
            if (batch != 1) {
                LOG_ERROR("audio_vae_decode: 1D waveform requires batch=1, got %d", batch);
                err = VOXCPM_ERR_SHAPE_MISMATCH;
                goto cleanup;
            }
            size_t single_bytes = (size_t)T_out * sizeof(float);
            if (waveform->size < (size_t)T_out) {
                LOG_ERROR("VAE P10: waveform size=%zu < T_out=%d", waveform->size, T_out);
                err = VOXCPM_ERR_SHAPE_MISMATCH;
                goto cleanup;
            }
            memcpy(waveform->data, h->data, single_bytes);
        }
    }

cleanup:
    tensor_free(h);
    return err;
}

/* ═══════════════════════════════════════════════════════════════
 * Public API — audio_vae_encode
 *
 * Encodes waveform to latent at 16kHz rate.
 *
 * Input:  waveform  [batch, samples]   (16kHz PCM waveform)
 * Output: latent    [batch, T_enc, latent_dim]
 *
 * Forward pass:
 *   1. Reshape waveform [B, T] → [B, 1, T]
 *   2. conv_in:   [B, 1, T] → [B, encoder_dim, T]     (k=7, SAME)
 *   3. 4× Encoder blocks (each: 3×ResSubBlock → Snake → StridedConv)
 *        rate: 2 → 5 → 8 → 8, channels: 128→256→512→1024→2048
 *      After block 3: [B, 2048, T_enc]
 *   4. fc_mu:     [B, 2048, T_enc] → [B, latent_dim, T_enc] (k=3, SAME)
 *   5. Permute from [B, D, T] to [B, T, D]
 * ═══════════════════════════════════════════════════════════════ */
VoxCPMError audio_vae_encode(
    const AudioVAE* vae,
    const Tensor* waveform,
    Tensor* latent)
{
    if (!vae || !waveform || !latent) return VOXCPM_ERR_INTERNAL;
    if (!vae->encoder || !vae->encoder->conv_in) return VOXCPM_ERR_INTERNAL;
    AudioVAEEncoder* enc = vae->encoder;

    /* ── Input validation ── */
    int wf_ndim = waveform->ndim;
    if (wf_ndim != 1 && wf_ndim != 2) {
        LOG_ERROR("VAE ENC01: waveform ndim=%d", wf_ndim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    int batch = (wf_ndim == 2) ? waveform->shape[0] : 1;
    int T     = (wf_ndim == 2) ? waveform->shape[1] : (int)waveform->size;
    if (T <= 0) {
        LOG_ERROR("VAE ENC02: empty waveform T=%d", T);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    /* Validate latent shape — expect [B, T_enc, latent_dim] */
    if (latent->ndim != 3) {
        LOG_ERROR("VAE ENC03: latent ndim=%d != 3", latent->ndim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (latent->shape[0] != batch) {
        LOG_ERROR("VAE ENC04: latent batch=%d != waveform batch=%d", latent->shape[0], batch);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }
    if (latent->shape[2] != vae->latent_dim) {
        LOG_ERROR("VAE ENC05: latent dim=%d != %d", latent->shape[2], vae->latent_dim);
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    VoxCPMError err = VOXCPM_SUCCESS;
    Tensor* h = NULL;  /* current activation tensor */

    /* Step 1: Reshape waveform [B, T] → [B, 1, T] */
    h = tensor_create(3, (int[]){batch, 1, T});
    if (!h) { err = VOXCPM_ERR_OOM; goto cleanup; }

    memcpy(h->data, waveform->data, (size_t)batch * (size_t)T * sizeof(float));

    LOG_DEBUG("VAE ENC: conv_in input [%d,%d,%d]", batch, 1, T);

    /* Step 2: conv_in [B, 1, T] → [B, encoder_dim, T] (k=7, SAME) */
    {
        int out_c = enc->conv_in->out_channels;  /* 128 */
        int T_out = (T + 2 * enc->conv_in->padding
                     - enc->conv_in->kernel_size)
                    / enc->conv_in->stride + 1;
        Tensor* conv_out = tensor_create(3, (int[]){batch, out_c, T_out});
        if (!conv_out) { err = VOXCPM_ERR_OOM; goto cleanup; }

        err = wnconv_forward(enc->conv_in, h, conv_out);
        tensor_free(h);
        if (err) { tensor_free(conv_out); LOG_ERROR("VAE ENC10: conv_in failed"); goto cleanup; }
        h = conv_out;
    }

    LOG_DEBUG("VAE ENC: after conv_in [%d,%d,%d]", h->shape[0], h->shape[1], h->shape[2]);

    /* Steps 3-6: Encoder blocks 0-3 (each downsamples) */
    for (int i = 0; i < 4; i++) {
        err = encoder_block_forward(&enc->blocks[i], &h);
        if (err) { LOG_ERROR("VAE ENC11: encoder block %d failed", i); goto cleanup; }
        LOG_DEBUG("VAE ENC: after block %d [%d,%d,%d]", i, h->shape[0], h->shape[1], h->shape[2]);
    }

    /* h is now [B, 2048, T_enc] */
    int T_enc = h->shape[2];

    /* Validate latent T dimension matches encoder output */
    if (latent->shape[1] != T_enc) {
        LOG_ERROR("VAE ENC06: latent T_enc=%d != encoder T_enc=%d", latent->shape[1], T_enc);
        err = VOXCPM_ERR_SHAPE_MISMATCH;
        goto cleanup;
    }

    /* Step 7: fc_mu [B, 2048, T_enc] → [B, latent_dim, T_enc] (k=3, SAME) */
    {
        int mu_c = enc->fc_mu->out_channels;  /* latent_dim */
        Tensor* mu_out = tensor_create(3, (int[]){batch, mu_c, T_enc});
        if (!mu_out) { err = VOXCPM_ERR_OOM; goto cleanup; }

        err = wnconv_forward(enc->fc_mu, h, mu_out);
        tensor_free(h);
        if (err) { tensor_free(mu_out); goto cleanup; }
        h = mu_out;
    }

    /* Step 8: Permute from [B, D, T] → [B, T, D] to match latent output layout */
    {
        int axes[3] = {0, 2, 1};  /* [B, D, T] -> [B, T, D] */
        err = tensor_permute(h, latent, axes);
        tensor_free(h);
        h = NULL;
        if (err) { LOG_ERROR("VAE ENC20: permute failed"); goto cleanup; }
    }

    LOG_DEBUG("VAE ENC: output latent [%d,%d,%d]",
              latent->shape[0], latent->shape[1], latent->shape[2]);
    return VOXCPM_SUCCESS;

cleanup:
    tensor_free(h);
    return err;
}

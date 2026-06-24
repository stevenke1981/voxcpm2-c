#ifndef NN_H
#define NN_H

/*
 * nn.h — Neural network layer definitions
 * VoxCPM2-C Project
 * License: Apache-2.0
 *
 * Transformer block, attention, FFN (SwiGLU), and normalization layers.
 */

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════ */
#define NN_MAX_LAYERS 64

/* ═══════════════════════════════════════════════════════════════
 * RMS Norm (standalone)
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    Tensor* weight;    /* [d_model] */
    float   eps;
} RmsNorm;

RmsNorm* rms_norm_create(int d_model, float eps);
void rms_norm_free(RmsNorm* norm);
VoxCPMError rms_norm_forward(const RmsNorm* norm, const Tensor* x, Tensor* out);

/* ═══════════════════════════════════════════════════════════════
 * Layer Norm (standalone)
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    Tensor* weight;    /* [d_model] */
    Tensor* bias;      /* [d_model] (optional) */
    float   eps;
} LayerNorm;

LayerNorm* layer_norm_create(int d_model, float eps, bool use_bias);
void layer_norm_free(LayerNorm* norm);
VoxCPMError layer_norm_forward(const LayerNorm* norm, const Tensor* x, Tensor* out);

/* ═══════════════════════════════════════════════════════════════
 * SwiGLU Feed-Forward Network (MiniCPM-4 style)
 *
 * FFN(x) = (silu(x @ W1) ⊙ (x @ W3)) @ W2
 * Three weight matrices (not the traditional two).
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    Tensor* w1;    /* [d_model, d_ff]  — gate */
    Tensor* w2;    /* [d_ff, d_model]  — down */
    Tensor* w3;    /* [d_model, d_ff]  — up */
    int d_model;
    int d_ff;
} SwiGLU;

SwiGLU* swiglu_create(int d_model, int d_ff);
void swiglu_free(SwiGLU* ff);
VoxCPMError swiglu_forward(const SwiGLU* ff, const Tensor* x, Tensor* out);

/* ═══════════════════════════════════════════════════════════════
 * Multi-Head Attention
 *
 * Supports:
 *   - Multi-Head / Grouped-Query Attention (GQA)
 *   - RoPE (applied externally via tensor_rotary_emb)
 *   - KV cache (external buffers passed in)
 *   - Causal mask (optional, pass NULL for non-causal)
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    Tensor* wq;          /* [d_model, n_heads * head_dim] */
    Tensor* wk;          /* [d_model, n_kv_heads * head_dim] */
    Tensor* wv;          /* [d_model, n_kv_heads * head_dim] */
    Tensor* wo;          /* [n_heads * head_dim, d_model] */

    int d_model;
    int n_heads;
    int n_kv_heads;
    int head_dim;
} Attention;

Attention* attention_create(int d_model, int n_heads, int n_kv_heads);
void attention_free(Attention* attn);

/* Standard attention forward:
 *   x:      [batch, seq, d_model] — input
 *   cache_k: [batch, n_kv_heads, max_seq, head_dim] — KV cache for keys
 *   cache_v: [batch, n_kv_heads, max_seq, head_dim] — KV cache for values
 *   mask:   [seq, seq] — causal mask (lower triangular), or NULL
 *   freqs_cis: precomputed RoPE frequencies
 *   pos:    starting position offset for RoPE
 *   out:    [batch, seq, d_model] — output
 */
VoxCPMError attention_forward(
    const Attention* attn,
    const Tensor* x,
    Tensor* cache_k,
    Tensor* cache_v,
    const Tensor* mask,
    const Tensor* freqs_cis,
    int pos,
    Tensor* out
);

/* Cross-attention forward (for RALM):
 *   x:      [batch, seq, d_model] — decoder input (queries)
 *   enc:    [batch, enc_seq, d_model] — encoder output (keys/values)
 *   out:    [batch, seq, d_model]
 */
VoxCPMError cross_attention_forward(
    const Attention* attn,
    const Tensor* x,
    const Tensor* enc,
    Tensor* out
);

/* ═══════════════════════════════════════════════════════════════
 * Transformer Block (standard pre-norm architecture)
 *
 * Used by: TSLM (24 layers), LocEnc (1 layer)
 *
 * Architecture:
 *   x → RMSNorm → Attention → residual +
 *   → RMSNorm → SwiGLU → residual +
 *   → out
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    RmsNorm*    rms_attn;       /* pre-attention norm */
    Attention*  attn;           /* multi-head attention */
    RmsNorm*    rms_ffn;        /* pre-FFN norm */
    SwiGLU*     ffn;            /* SwiGLU FFN */
} TransformerBlock;

TransformerBlock* transformer_block_create(
    int d_model, int n_heads, int n_kv_heads, int d_ff, float norm_eps
);
void transformer_block_free(TransformerBlock* block);

VoxCPMError transformer_block_forward(
    const TransformerBlock* block,
    const Tensor* x,
    Tensor* cache_k,
    Tensor* cache_v,
    const Tensor* mask,
    const Tensor* freqs_cis,
    int pos,
    Tensor* out
);

/* ═══════════════════════════════════════════════════════════════
 * Diffusion Transformer Block (for LocDiT)
 *
 * Used in the diffusion model with adaptive layer norm (adaLN)
 * and cross-attention for condition injection.
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    RmsNorm*       rms_attn;
    Attention*     attn;
    RmsNorm*       rms_cross;
    Attention*     cross_attn;  /* cross-attention to condition */
    RmsNorm*       rms_ffn;
    SwiGLU*        ffn;

    /* Adaptive layer norm scale/shift (from timestep embedding) */
    Tensor* adaLN_gamma;   /* [6 * d_model] — combined scale/shift */
} DiTBlock;

DiTBlock* dit_block_create(
    int d_model, int n_heads, int n_kv_heads, int d_ff, float norm_eps
);
void dit_block_free(DiTBlock* block);

VoxCPMError dit_block_forward(
    const DiTBlock* block,
    const Tensor* x,
    const Tensor* cond,
    const Tensor* adaLN_params,  /* [batch, 6 * d_model] or NULL */
    Tensor* out
);

/* ═══════════════════════════════════════════════════════════════
 * Conv1D (for AudioVAE V2)
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    Tensor* weight;       /* [out_channels, in_channels/groups, kernel_size] */
    Tensor* bias;         /* [out_channels] (optional) */
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int groups;           /* 1=regular, out_channels=depthwise */
} Conv1D;

Conv1D* conv1d_create(
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding
);
void conv1d_free(Conv1D* conv);

/* x shape: [batch, in_channels, time], out shape: [batch, out_channels, out_time] */
VoxCPMError conv1d_forward(const Conv1D* conv, const Tensor* x, Tensor* out);

/* ═══════════════════════════════════════════════════════════════
 * Transpose Conv1D (upsampling for AudioVAE V2)
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    Tensor* weight;       /* [in_channels, out_channels, kernel_size] */
    Tensor* bias;         /* [out_channels] (optional) */
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
} Conv1DTranspose;

Conv1DTranspose* conv1d_transpose_create(
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding
);
void conv1d_transpose_free(Conv1DTranspose* conv);

VoxCPMError conv1d_transpose_forward(
    const Conv1DTranspose* conv, const Tensor* x, Tensor* out
);

/* ═══════════════════════════════════════════════════════════════
 * Snake activation (for AudioVAE V2)
 * snake(x) = x + (1 / beta) * sin^2(beta * x)
 * ═══════════════════════════════════════════════════════════════ */
VoxCPMError tensor_snake(Tensor* t, float beta);

#ifdef __cplusplus
}
#endif

#endif /* NN_H */

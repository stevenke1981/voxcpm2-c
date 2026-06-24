#ifndef MODEL_H
#define MODEL_H

/*
 * model.h — VoxCPM2 model structure definitions
 * VoxCPM2-C Project
 * License: Apache-2.0
 *
 * Defines the complete VoxCPM2 model architecture:
 *   LocEnc (12 layers) → TSLM (28 layers) → [RALM (8 layers)] → LocDiT (12 layers) → AudioVAE V2
 *
 * Architecture (from openbmb/VoxCPM2 config.json):
 *   TSLM:   MiniCPM-4, 28 layers, hidden=2048, heads=16, kv_heads=2, intermediate=6144
 *   RALM:   Residual LM, 8 layers, hidden=2048 (same head config)
 *   LocEnc: 12 layers, hidden=1024, heads=16, kv_heads=2, intermediate=4096
 *   LocDiT: 12 layers, hidden=1024, heads=16, kv_heads=2, intermediate=4096
 *   AudioVAE V2: 16kHz latent → 48kHz waveform, latent_dim=64, patch_size=4
 */

#include "nn.h"
#include "platform.h"
#include "voxcpm.h"
#include "tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 * .vxcpm Weight Format Constants
 * ═══════════════════════════════════════════════════════════════ */

#define VXCPM_MAGIC          0x56435850  /* "VXCP" */
#define VXCPM_HEADER_SIZE    64
#define VXCPM_META_ENTRY_SIZE 32
#define VXCPM_VERSION        1

/* Data type codes */
#define VXCPM_DTYPE_FP32     0
#define VXCPM_DTYPE_BF16     1
#define VXCPM_DTYPE_FP16     2
#define VXCPM_DTYPE_Q4_0     3
#define VXCPM_DTYPE_Q4_1     4

/* ═══════════════════════════════════════════════════════════════
 * .vxcpm On-Disk Structures
 * ═══════════════════════════════════════════════════════════════ */

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;              /* 0x56435850 = "VXCP" */
    uint32_t version;            /* format version (1) */
    uint32_t num_tensors;        /* number of tensors in file */
    uint32_t reserved[10];       /* reserved, must be 0 */
    uint32_t header_checksum;    /* CRC32C of bytes 0–55 (with this field = 0) */
    uint32_t data_checksum;      /* CRC32C of metadata + string table + data */
} VxcpmHeader;

typedef struct {
    uint32_t name_offset;        /* byte offset into string table */
    uint32_t name_length;        /* name string length (bytes, excl. null) */
    uint32_t dtype;              /* data type code */
    uint32_t ndim;               /* number of dimensions (1-4) */
    uint32_t shape[4];           /* shape [d0, d1, d2, d3] (unused = 1) */
} VxcpmTensorMeta;
#pragma pack(pop)

/* ═══════════════════════════════════════════════════════════════
 * In-Memory Tensor Index
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    char*    name;               /* weight name (from string table) */
    uint32_t dtype;              /* data type code */
    uint32_t ndim;
    uint32_t shape[4];
    uint64_t data_offset;        /* byte offset from file start to data */
    uint64_t data_size;          /* size of data in bytes */
} WeightEntry;

typedef struct WeightIndex {
    WeightEntry* entries;        /* sorted by name */
    int          count;          /* number of entries */
    int          capacity;
} WeightIndex;

/* ═══════════════════════════════════════════════════════════════
 * Model Configuration
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int     d_model;                /* hidden dimension (2048) */
    int     n_heads;                /* number of query heads (16) */
    int     n_kv_heads;             /* number of key/value heads for GQA (2) */
    int     d_ff;                   /* feed-forward intermediate size (6144) */
    int     n_layers_tslm;          /* TSLM layers (28) */
    int     n_layers_ralm;          /* RALM / residual LM layers (8) */
    int     n_layers_loc_dit;       /* LocDiT layers (12) */
    int     n_layers_loc_enc;       /* LocEnc encoder layers (12) */
    int     max_seq_len;            /* maximum sequence length (32768) */
    int     vocab_size;             /* vocabulary size (73448) */
    int     latent_dim;             /* feat_dim for diffusion (64) */
    int     patch_size;             /* patch size (4) */
    int     sample_rate;            /* output sample rate (48000) */
    int     vae_encode_sr;          /* VAE encode sample rate (16000) */
    int     n_threads;              /* CPU threads */
    bool    use_gpu;                /* enable GPU acceleration */
    float   rope_theta;             /* RoPE theta (10000.0) */
    float   norm_eps;               /* normalization epsilon (1e-5) */
    int     head_dim;               /* head dimension (128 = kv_channels) */
    int     enc_hidden;             /* encoder hidden dim (1024) */
    int     enc_d_ff;               /* encoder intermediate (4096) */
    int     enc_n_heads;            /* encoder heads (16) */
    int     enc_n_layers;           /* encoder layers (12) */
    int     dit_hidden;             /* DiT hidden dim (1024) */
    int     dit_d_ff;               /* DiT intermediate (4096) */
    int     dit_n_heads;            /* DiT heads (16) */
    int     dit_n_layers;           /* DiT layers (12) */
    int     enc_head_dim;           /* encoder head dim (128) */
    int     dit_head_dim;           /* DiT head dim (128) */
} VoxCPMConfig;

/* Return default configuration. */
VoxCPMConfig voxcpm_config_default(void);

/* ═══════════════════════════════════════════════════════════════
 * Local Encoder (LocEnc) — VoxCPMLocEnc
 *
 * 12-layer transformer encoder that encodes audio features into local features.
 *   input:  audio_feats [batch, time, patch, feat_dim]  (feat_dim=64)
 *   output: features [batch, time, enc_hidden]          (enc_hidden=1024)
 *
 * Architecture (from voxcpm2.py):
 *   - in_proj: Linear(64 -> 1024) maps patches to hidden dim
 *   - special_token: learnable [CLS] token prepended per time step
 *   - encoder: MiniCPMModel(12 layers, hidden=1024, 16 heads, GQA=2)
 *   - Output: CLS token (first token) from each group = [batch, time, 1024]
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    Tensor*          in_proj_weight;   /* [enc_hidden, feat_dim] input projection */
    Tensor*          in_proj_bias;     /* [enc_hidden] */
    Tensor*          special_token;    /* [1, 1, 1, enc_hidden] learnable CLS token */
    TransformerBlock* layers;          /* array of 12 transformer blocks */
    int               n_layers;
    RmsNorm*          output_norm;    /* final RMS norm [enc_hidden] */
    int               d_model;        /* enc_hidden (1024) */
    int               patch_size;     /* (4) */
} LocEnc;

LocEnc* loc_enc_create(const VoxCPMConfig* config, const uint8_t* mmap_data, const WeightIndex* idx);
void loc_enc_free(LocEnc* enc);

/* Encode audio features. */
VoxCPMError loc_enc_forward(
    const LocEnc* enc,
    const Tensor* audio_feats,      /* [batch, time, patch, feat_dim] */
    Tensor* out                     /* [batch, time, enc_hidden] */
);

/* ═══════════════════════════════════════════════════════════════
 * Text-Speech Language Model (TSLM / base_lm)
 *
 * MiniCPM-4 backbone — 28-layer autoregressive transformer decoder.
 *   input:  features [batch, seq, d_model]        (d_model=2048)
 *   output: hidden [batch, seq, d_model]
 *
 * Architecture (from config.json lm_config):
 *   - embed_tokens: Embedding(vocab_size=73448, d_model=2048) — NO lm_head
 *   - 28 layers, each with:
 *     - input_layernorm (RMS norm, eps=1e-5)
 *     - self_attn: 16 Q heads, 2 KV heads (GQA), head_dim=128 (kv_channels)
 *     - MLP: SwiGLU with gate/up/down, intermediate=6144
 *     - post_attention_layernorm (RMS norm)
 *   - final norm: RMS norm
 *   - LongRoPE with dynamic scaling (short/long factors)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    Tensor*           embed_weight;    /* token embedding [vocab_size, d_model] */
    TransformerBlock* layers;          /* array of n_layers (28) blocks */
    int               n_layers;
    RmsNorm*          output_norm;     /* final RMS norm [d_model] */
    int               d_model;
    int               vocab_size;

    /* KV cache (owned) — StaticKVCache layout */
    Tensor* cache_k;                  /* [n_layers, batch, n_kv_heads, max_seq, head_dim] */
    Tensor* cache_v;                  /* [n_layers, batch, n_kv_heads, max_seq, head_dim] */
    int     cache_len;                /* current cache length (position) */
    int     max_seq_len;
    int     n_kv_heads;
    int     head_dim;
} TSLM;

TSLM* tslm_create(const VoxCPMConfig* config, const uint8_t* mmap_data, const WeightIndex* idx);
void tslm_free(TSLM* tslm);

/* Forward pass through all TSLM layers (prefill mode).
 * On first call, cache must be empty. After prefill, cache contains
 * the full context so subsequent forward_step calls can decode token by token. */
VoxCPMError tslm_forward(
    TSLM* tslm,
    const Tensor* x,                 /* [batch, seq, d_model] — input embeddings */
    const Tensor* freqs_cis,         /* precomputed RoPE cos/sin [seq, head_dim] */
    Tensor* out                      /* [batch, seq, d_model] — hidden states */
);

/* Single token forward step (decoding mode, uses KV cache). */
VoxCPMError tslm_forward_step(
    TSLM* tslm,
    const Tensor* x,                 /* [batch, d_model] — single token embedding */
    int position_id,
    const Tensor* freqs_cis_single,  /* [1, head_dim] — RoPE for this position */
    Tensor* out                      /* [batch, d_model] — output hidden state */
);

/* Clear KV cache (call between different generations). */
void tslm_cache_clear(TSLM* tslm);

/* Set up KV cache (after config is known). */
VoxCPMError tslm_setup_cache(TSLM* tslm, int batch_size);

/* ═══════════════════════════════════════════════════════════════
 * Reference-Aware Language Model (RALM / residual_lm)
 *
 * 8-layer autoregressive decoder (same architecture as TSLM but fewer layers).
 * Used as residual acoustic LM in voice cloning pipeline.
 *   input:  features [batch, seq, d_model]        (d_model=2048)
 *   output: hidden [batch, seq, d_model]
 *
 * Architecture (from config.json):
 *   - residual_lm_num_layers = 8
 *   - Same head config as TSLM (16 heads, GQA=2, head_dim=128)
 *   - intermediate_size = 6144
 *   - No embedding (vocab_size = 0)
 *   - no_rope = true (from residual_lm_no_rope)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    TransformerBlock* layers;         /* array of n_layers (8) blocks */
    int               n_layers;
    RmsNorm*          output_norm;    /* final RMS norm [d_model] */
    int               d_model;

    /* KV cache (owned) */
    Tensor* cache_k;                  /* [n_layers, batch, n_kv_heads, max_seq, head_dim] */
    Tensor* cache_v;
    int     cache_len;
    int     max_seq_len;
    int     n_kv_heads;
    int     head_dim;
    bool    no_rope;                  /* true for residual_lm */
} RALM;

RALM* ralm_create(const VoxCPMConfig* config, const uint8_t* mmap_data, const WeightIndex* idx);
void ralm_free(RALM* ralm);

VoxCPMError ralm_forward(
    RALM* ralm,
    const Tensor* x,                 /* features [batch, seq, d_model] */
    const Tensor* freqs_cis,         /* may be NULL if no_rope */
    Tensor* out                      /* [batch, seq, d_model] */
);

VoxCPMError ralm_forward_step(
    RALM* ralm,
    const Tensor* x,                 /* [batch, d_model] */
    int position_id,
    const Tensor* freqs_cis_single,
    Tensor* out
);

void ralm_cache_clear(RALM* ralm);
VoxCPMError ralm_setup_cache(RALM* ralm, int batch_size);

/* ═══════════════════════════════════════════════════════════════
 * Local Diffusion Transformer (LocDiT) — VoxCPMLocDiTV2
 *
 * 12-layer diffusion transformer that denoises latent representations.
 * Uses DDIM sampler (Euler solver, log-norm noise schedule).
 *   input:  x [batch, feat_dim, time], mu [batch, dit_hidden],
 *           cond [batch, feat_dim, time'], t [batch], dt [batch]
 *   output: denoised [batch, feat_dim, time]
 *
 * Architecture (from config.json dit_config):
 *   - in_proj: Linear(feat_dim=64 -> dit_hidden=1024)
 *   - cond_proj: Linear(feat_dim=64 -> dit_hidden=1024)
 *   - out_proj: Linear(dit_hidden=1024 -> feat_dim=64)
 *   - Sinusoidal time embedding -> time_mlp MLP (SiLU)
 *   - delta_time_mlp MLP
 *   - decoder: MiniCPMModel(12 layers, hidden=1024, 16 heads, GQA=2)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    Tensor*           in_proj_weight;   /* [dit_hidden, feat_dim] */
    Tensor*           in_proj_bias;     /* [dit_hidden] */
    Tensor*           cond_proj_weight; /* [dit_hidden, feat_dim] */
    Tensor*           cond_proj_bias;   /* [dit_hidden] */
    Tensor*           out_proj_weight;  /* [feat_dim, dit_hidden] */
    Tensor*           out_proj_bias;    /* [feat_dim] */

    /* Time embeddings */
    Tensor*           time_mlp_1_weight; /* [dit_hidden, dit_hidden] */
    Tensor*           time_mlp_1_bias;
    Tensor*           time_mlp_2_weight; /* [dit_hidden, dit_hidden] */
    Tensor*           time_mlp_2_bias;
    Tensor*           delta_mlp_1_weight;
    Tensor*           delta_mlp_1_bias;
    Tensor*           delta_mlp_2_weight;
    Tensor*           delta_mlp_2_bias;

    TransformerBlock* layers;          /* array of n_layers (12) DiT decoder blocks */
    int               n_layers;
    RmsNorm*          output_norm;     /* decoder final RMS norm [dit_hidden] */

    int               d_model;         /* dit_hidden (1024) */
    int               feat_dim;        /* 64 */
} LocDiT;

LocDiT* loc_dit_create(const VoxCPMConfig* config, const uint8_t* mmap_data, const WeightIndex* idx);
void loc_dit_free(LocDiT* dit);

VoxCPMError loc_dit_forward(
    const LocDiT* dit,
    const Tensor* x,                 /* noisy latent [batch, feat_dim, time] */
    const Tensor* mu,                /* conditioning [batch, dit_hidden] */
    const Tensor* cond,              /* prefix condition [batch, feat_dim, time'] */
    const Tensor* t,                 /* timestep [batch] */
    const Tensor* dt,                /* delta time [batch] */
    Tensor* out                      /* denoised [batch, feat_dim, time] */
);

/* Diffusion inference: denoise from noise to clean latent. */
VoxCPMError loc_dit_sample(
    const LocDiT* dit,
    const Tensor* mu,                /* [batch, dit_hidden] */
    const Tensor* cond,              /* [batch, feat_dim, time'] */
    int n_timesteps,
    float cfg_value,
    Tensor* out                      /* [batch, feat_dim, patch_size] */
);

/* ═══════════════════════════════════════════════════════════════
 * AudioVAE V2
 *
 * Asymmetric VAE: 16kHz input -> 48kHz output.
 * Uses 1D convolutions with Snake activation, weight_norm, and upsampling.
 *
 */

#define AUDIOVAE_NUM_DECODER_BLOCKS   6
#define AUDIOVAE_RES_BLOCKS_PER_BLOCK 3

/* Forward declaration for Conv1D wrapper (weight_norm aware) */
typedef struct {
    Tensor* weight_v;                /* weight vector (actual weights) */
    Tensor* weight_g;                /* weight gain (scalar per output channel) */
    Tensor* bias;                    /* bias (may be NULL) */
    Tensor* effective_weight;        /* cached normalized weight (computed from weight_v, weight_g) */
    int     in_channels;
    int     out_channels;
    int     kernel_size;
    int     stride;
    int     padding;
    int     groups;
    bool    has_bias;
    bool    is_transpose;
} WNConv;

typedef struct {
    WNConv*  conv_in;               /* first conv: (1 -> encoder_dim) kernel=7 */
    WNConv*  conv_blocks[4];        /* 4 encoder blocks (rates=[2,5,8,8]) */
    WNConv*  fc_mu;                 /* mu head conv: (enc_dim*16 -> latent_dim) kernel=3 */
    WNConv*  fc_logvar;             /* logvar head conv */

    int      encoder_dim;            /* 128 */
    int      latent_dim;             /* 64 */
} AudioVAEEncoder;

/* ── Residual sub-block: Snake → DepthwiseConv(k=7) → Snake → PointwiseConv(k=1) ── */
typedef struct {
    WNConv*  conv_depthwise;        /* depthwise Conv1d: k=7, groups=out_channels, no bias */
    WNConv*  conv_pointwise;        /* pointwise Conv1d: k=1, no bias */
    Tensor*  snake_alpha1;          /* Snake alpha before depthwise conv */
    Tensor*  snake_alpha2;          /* Snake alpha before pointwise conv */
} AudioVAEResBlock;

/* ── Decoder block: Snake → ConvTR → 3×ResidualSubBlock ── */
typedef struct {
    Tensor*          snake_alpha;   /* Snake alpha before convtr */
    WNConv*          convtr;        /* ConvTranspose1D for upsampling */
    AudioVAEResBlock res_blocks[3]; /* 3 residual sub-blocks */
    int              num_res_blocks;/* 3 */
} AudioVAEDecoderBlock;

typedef struct {
    WNConv*               conv_in;               /* depthwise Conv1d (64, 64, k=7) */
    WNConv*               proj_up;               /* projection Conv1d (64→2048, k=1) */
    AudioVAEDecoderBlock  decoder_blocks[6];     /* 6 upsampling decoder blocks */
    Tensor*               final_snake_alpha;     /* Snake alpha before conv_out */
    WNConv*               conv_out;              /* final Conv1d (32→1, k=7) + Tanh */

    /* Sample rate conditioning */
    int      sr_bin_boundaries[3];  /* [20000, 30000, 40000] */
    int      sr_bin_buckets;        /* 4 */
    bool     has_sr_cond;
    int      sr_cond_layers;        /* number of decoder blocks with SR cond */
} AudioVAEDecoder;

typedef struct {
    AudioVAEEncoder* encoder;
    AudioVAEDecoder* decoder;

    int               latent_dim;      /* 64 */
    int               sample_rate;     /* 16000 (encode) */
    int               out_sample_rate; /* 48000 (decode) */
    int               chunk_size;      /* 640 = prod(encoder_rates) */
    int               decode_chunk_size; /* 1920 = prod(decoder_rates) */
    int               patch_size;      /* 4 */
} AudioVAE;

AudioVAE* audio_vae_create(const VoxCPMConfig* config, const uint8_t* mmap_data, const WeightIndex* idx);
void audio_vae_free(AudioVAE* vae);

/* Decode latent to waveform. */
VoxCPMError audio_vae_decode(
    const AudioVAE* vae,
    const Tensor* latent,            /* [batch, time, latent_dim] */
    Tensor* waveform                 /* [batch, samples] */
);

/* Encode waveform to latent (for cloning). */
VoxCPMError audio_vae_encode(
    const AudioVAE* vae,
    const Tensor* waveform,          /* [batch, samples] */
    Tensor* latent                   /* [batch, time, latent_dim] */
);

/* ═══════════════════════════════════════════════════════════════
 * VoxCPM Full Model
 * ═══════════════════════════════════════════════════════════════ */

struct VoxCPMModel {
    VoxCPMConfig config;

    /* Sub-models */
    LocEnc*     loc_enc;
    TSLM*       tslm;
    RALM*       ralm;
    LocDiT*     loc_dit;
    AudioVAE*   audio_vae;

    /* Embeddings */
    Tensor*     text_embed;           /* token embedding (shared or separate) */
    Tensor*     audio_embed;          /* audio prompt embedding */

    /* LM → DiT projection (mu extraction) */
    Tensor*     lm_to_dit_weight;     /* [dit_hidden, d_model] */
    Tensor*     lm_to_dit_bias;       /* [dit_hidden] */

    /* Encoder → LM projection (audio context) */
    Tensor*     enc_to_lm_proj_weight;    /* [d_model, enc_hidden] */
    Tensor*     enc_to_lm_proj_bias;      /* [d_model] */

    /* Residual → DiT projection (for cond from text hidden) */
    Tensor*     res_to_dit_proj_weight;   /* [dit_hidden, d_model] */
    Tensor*     res_to_dit_proj_bias;     /* [dit_hidden] */

    /* Fusion: concatenated hidden+enc → hidden */
    Tensor*     fusion_concat_proj_weight; /* [d_model, d_model*2] */
    Tensor*     fusion_concat_proj_bias;   /* [d_model] */

    /* Tokenizer */
    Tokenizer*  tokenizer;            /* BPE tokenizer */

    /* Precomputed RoPE frequencies */
    Tensor*     freqs_cis;            /* [max_seq_len, head_dim/2, 2] */

    /* Weights buffer (mmap) */
    MmapFile*   weights_mmap;
    WeightIndex* weight_index;        /* sorted index of weight entries */

    /* AudioVAE weights (separate .vxcpm file) */
    MmapFile*   audiovae_mmap;
    WeightIndex* audiovae_index;

    /* Thread pool (for parallel layer computation) */
    ThreadPool* thread_pool;

    /* Generation state */
    volatile bool cancel_requested;
    int            timeout_ms;

    /* GPU state */
    bool        gpu_enabled;
    int         gpu_device_id;
    void*       gpu_context;
};

/* ═══════════════════════════════════════════════════════════════
 * Weight Index Management (model.c)
 * ═══════════════════════════════════════════════════════════════ */

/* Build a WeightIndex from mmap'd .vxcpm file data.
 * Parses header, builds sorted index of all tensors.
 * Returns pointer to heap-allocated index, or NULL on error. */
WeightIndex* weight_index_build(const uint8_t* data, size_t data_size);

/* Free a WeightIndex. */
void weight_index_free(WeightIndex* idx);

/* Find a weight entry by exact name (binary search).
 * Returns pointer to entry, or NULL if not found. */
const WeightEntry* weight_index_find(const WeightIndex* idx, const char* name);

/* Load a weight tensor from its mmap'd WeightEntry.
 * Allocates a new tensor of the correct shape and converts dtype as needed.
 * Returns VOXCPM_SUCCESS or error code. */
VoxCPMError weight_load_tensor(
    const uint8_t* data,
    const WeightEntry* entry,
    Tensor** out_tensor
);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_H */

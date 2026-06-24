// model.c — Model lifecycle, weight loading, and inference pipeline
// VoxCPM2-C Project
// License: Apache-2.0
//
// Phase 1: Weight loading (.vxcpm binary format), model lifecycle.
// Inference pipeline will be implemented in Phase 2.

#include "model.h"
#include "audio.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

/* ═══════════════════════════════════════════════════════════════
 * CRC32-C (Castagnoli polynomial 0x1EDC6F41)
 * ═══════════════════════════════════════════════════════════════ */

static uint32_t crc32c_table[256];
static int crc32c_initialized = 0;

static void crc32c_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1)
                crc = 0x82F63B78 ^ (crc >> 1);
            else
                crc >>= 1;
        }
        crc32c_table[i] = crc;
    }
    crc32c_initialized = 1;
}

static uint32_t crc32c_compute(const uint8_t* data, size_t len, uint32_t crc) {
    if (!crc32c_initialized) crc32c_init();
    crc ^= 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc = crc32c_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

/* ═══════════════════════════════════════════════════════════════
 * Public API — Error string
 * ═══════════════════════════════════════════════════════════════ */

const char* voxcpm_error_string(VoxCPMError err) {
    switch (err) {
        case VOXCPM_SUCCESS:             return "Success";
        case VOXCPM_ERR_FILE_NOT_FOUND:  return "File not found";
        case VOXCPM_ERR_INVALID_MODEL:   return "Invalid model file";
        case VOXCPM_ERR_INVALID_TEXT:    return "Invalid text";
        case VOXCPM_ERR_INVALID_AUDIO:   return "Invalid audio";
        case VOXCPM_ERR_SHAPE_MISMATCH:  return "Shape mismatch";
        case VOXCPM_ERR_OOM:             return "Out of memory";
        case VOXCPM_ERR_GPU:             return "GPU error";
        case VOXCPM_ERR_INTERNAL:        return "Internal error";
        case VOXCPM_ERR_UNSUPPORTED:     return "Unsupported operation";
        case VOXCPM_ERR_TIMEOUT:         return "Timeout";
        case VOXCPM_ERR_CUDA_NOT_FOUND:  return "CUDA not available";
        default:                         return "Unknown error";
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Public API — Version
 * ═══════════════════════════════════════════════════════════════ */

const char* voxcpm_version(void) {
    return VOXCPM_VERSION;
}

/* ═══════════════════════════════════════════════════════════════
 * Public API — GPU detection
 * ═══════════════════════════════════════════════════════════════ */

int voxcpm_gpu_count(void) {
#ifdef VOXCPM_CUDA
    // TODO: Query CUDA device count
    return 0;
#else
    return 0;
#endif
}

void voxcpm_gpu_name(int device_id, char* buffer, int buffer_size) {
    (void)device_id;
    if (buffer && buffer_size > 0) {
        buffer[0] = '\0';
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Public API — Config defaults
 * ═══════════════════════════════════════════════════════════════ */

VoxCPMModelConfig voxcpm_model_config_default(void) {
    VoxCPMModelConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.model_path = "models/voxcpm2-f16.vxcpm";
    cfg.n_threads = 4;
    cfg.use_gpu = false;
    cfg.gpu_device = 0;
    cfg.quantization = NULL;
    cfg.max_seq_len = 2048;
    cfg.gpu_memory_limit = 0;
    cfg.verbosity = 2;
    return cfg;
}

VoxCPMGenConfig voxcpm_gen_config_default(void) {
    VoxCPMGenConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.text = "";
    cfg.reference_wav = NULL;
    cfg.prompt_text = NULL;
    cfg.voice_design = NULL;
    cfg.cfg_value = 2.0f;
    cfg.inference_timesteps = 10;
    cfg.denoise = true;
    cfg.normalize = true;
    cfg.seed = 0;
    cfg.max_new_tokens = 1024;
    cfg.temperature = 1.0f;
    cfg.timeout_ms = 0;
    return cfg;
}

/* ═══════════════════════════════════════════════════════════════
 * Public API — Audio
 * ═══════════════════════════════════════════════════════════════ */

void voxcpm_audio_free(VoxCPMAudio* audio) {
    if (audio && audio->samples) {
        free(audio->samples);
        audio->samples = NULL;
        audio->num_samples = 0;
    }
}

VoxCPMError voxcpm_audio_save(const VoxCPMAudio* audio, const char* path) {
    if (!audio || !path) return VOXCPM_ERR_INTERNAL;
    if (!audio->samples || audio->num_samples <= 0) return VOXCPM_ERR_INVALID_AUDIO;

    // WAV file header writing
    FILE* f = fopen(path, "wb");
    if (!f) return VOXCPM_ERR_FILE_NOT_FOUND;
    int sample_rate = audio->sample_rate > 0 ? audio->sample_rate : 48000;
    int bits_per_sample = 16;
    int channels = 1;
    int data_size = audio->num_samples * (bits_per_sample / 8);
    int fmt_size = 16;

    // RIFF header
    fwrite("RIFF", 1, 4, f);
    uint32_t chunk_size = (uint32_t)(36 + data_size);
    fwrite(&chunk_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    // fmt subchunk
    fwrite("fmt ", 1, 4, f);
    fwrite(&fmt_size, 4, 1, f);
    uint16_t audio_format = 1; // PCM
    fwrite(&audio_format, 2, 1, f);
    uint16_t num_channels = (uint16_t)channels;
    fwrite(&num_channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    uint32_t byte_rate = (uint32_t)(sample_rate * channels * (bits_per_sample / 8));
    fwrite(&byte_rate, 4, 1, f);
    uint16_t block_align = (uint16_t)(channels * (bits_per_sample / 8));
    fwrite(&block_align, 2, 1, f);
    uint16_t bits = (uint16_t)bits_per_sample;
    fwrite(&bits, 2, 1, f);

    // data subchunk
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);

    // Convert float [-1,1] to int16 and write
    for (int i = 0; i < audio->num_samples; i++) {
        float s = audio->samples[i];
        if (s < -1.0f) s = -1.0f;
        if (s > 1.0f) s = 1.0f;
        int16_t sample = (int16_t)(s * 32767.0f);
        fwrite(&sample, 2, 1, f);
    }

    fclose(f);
    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * Public API — Model lifecycle (stubs for Phase 0)
 * ═══════════════════════════════════════════════════════════════ */

// Forward declarations
static Tensor* load_weight(const char* name, const uint8_t* mmap_data, const WeightIndex* idx);
static TransformerBlock* transformer_block_from_weights(
    const char* prefix, const VoxCPMConfig* config, int head_dim,
    const uint8_t* mmap_data, const WeightIndex* idx);
static TransformerBlock* enc_block_from_weights(
    const char* prefix, const VoxCPMConfig* config,
    const uint8_t* mmap_data, const WeightIndex* idx);

VoxCPMModel* voxcpm_create(const VoxCPMModelConfig* config, VoxCPMError* err) {
    if (!config) {
        if (err) *err = VOXCPM_ERR_INTERNAL;
        return NULL;
    }

    // Allocate model
    VoxCPMModel* model = (VoxCPMModel*)calloc(1, sizeof(VoxCPMModel));
    if (!model) {
        if (err) *err = VOXCPM_ERR_OOM;
        return NULL;
    }

    // Set config
    model->config = voxcpm_config_default();
    model->config.n_threads = config->n_threads > 0 ? config->n_threads : 4;
    model->config.use_gpu = config->use_gpu;
    model->gpu_enabled = config->use_gpu;
    model->gpu_device_id = config->gpu_device;
    model->timeout_ms = 0;
    model->cancel_requested = false;

    // Check if model file exists
    if (config->model_path) {
        if (!file_exists(config->model_path)) {
            LOG_ERROR("Model file not found: %s", config->model_path);
            voxcpm_free(model);
            if (err) *err = VOXCPM_ERR_FILE_NOT_FOUND;
            return NULL;
        }

        // Map weight file
        model->weights_mmap = mmap_open(config->model_path);
        if (!model->weights_mmap) {
            LOG_ERROR("Failed to mmap: %s", config->model_path);
            voxcpm_free(model);
            if (err) *err = VOXCPM_ERR_FILE_NOT_FOUND;
            return NULL;
        }

        LOG_INFO("Model loaded: %s (%zu bytes)", config->model_path, model->weights_mmap->size);

        // Build weight index
        model->weight_index = weight_index_build(
            (const uint8_t*)model->weights_mmap->addr,
            model->weights_mmap->size);

        if (!model->weight_index) {
            LOG_ERROR("Failed to parse .vxcpm weight file: %s", config->model_path);
            voxcpm_free(model);
            if (err) *err = VOXCPM_ERR_INVALID_MODEL;
            return NULL;
        }

        LOG_INFO("Weight index built: %d tensors", model->weight_index->count);
    }

    // Create thread pool
    model->thread_pool = thread_pool_create(model->config.n_threads);

    // Create sub-models from weights
    if (model->weight_index && model->weights_mmap) {
        const uint8_t* data = (const uint8_t*)model->weights_mmap->addr;
        const WeightIndex* idx = model->weight_index;

        LOG_INFO("Creating TSLM (base_lm)...");
        model->tslm = tslm_create(&model->config, data, idx);
        if (!model->tslm) LOG_WARN("TSLM creation failed (missing weights?)");

        LOG_INFO("Creating RALM (residual_lm)...");
        model->ralm = ralm_create(&model->config, data, idx);
        if (!model->ralm) LOG_WARN("RALM creation failed");

        LOG_INFO("Creating LocDiT...");
        model->loc_dit = loc_dit_create(&model->config, data, idx);
        if (!model->loc_dit) LOG_WARN("LocDiT creation failed");

        // Load LM→DiT projection weights for mu extraction
        model->lm_to_dit_weight = load_weight("lm_to_dit_proj.weight", data, idx);
        model->lm_to_dit_bias   = load_weight("lm_to_dit_proj.bias", data, idx);
        if (!model->lm_to_dit_weight) LOG_WARN("lm_to_dit_proj.weight not found");

        // Load encoder→LM projection (audio context into LM space)
        model->enc_to_lm_proj_weight = load_weight("enc_to_lm_proj.weight", data, idx);
        model->enc_to_lm_proj_bias   = load_weight("enc_to_lm_proj.bias", data, idx);
        if (!model->enc_to_lm_proj_weight) LOG_WARN("enc_to_lm_proj.weight not found");

        // Load residual→DiT projection (text hidden → dit_hidden for cond)
        model->res_to_dit_proj_weight = load_weight("res_to_dit_proj.weight", data, idx);
        model->res_to_dit_proj_bias   = load_weight("res_to_dit_proj.bias", data, idx);
        if (!model->res_to_dit_proj_weight) LOG_WARN("res_to_dit_proj.weight not found");

        // Load fusion_concat projection (hidden+enc → hidden)
        model->fusion_concat_proj_weight = load_weight("fusion_concat_proj.weight", data, idx);
        model->fusion_concat_proj_bias   = load_weight("fusion_concat_proj.bias", data, idx);
        if (!model->fusion_concat_proj_weight) LOG_WARN("fusion_concat_proj.weight not found");

        // Create LocEnc (feat_encoder) for audio feature encoding
        LOG_INFO("Creating LocEnc (feat_encoder)...");
        model->loc_enc = loc_enc_create(&model->config, data, idx);
        if (!model->loc_enc) LOG_WARN("LocEnc creation failed");

        // AudioVAE loads from separate companion file
        // See: load_audiovae_separate below

        // Precompute RoPE frequencies
        model->freqs_cis = tensor_create(3, (int[]){
            model->config.max_seq_len,
            model->config.head_dim / 2,
            2
        });
        if (model->freqs_cis) {
            VoxCPMError rope_err = tensor_precompute_freqs_cis(
                model->config.max_seq_len,
                model->config.head_dim,
                model->config.rope_theta,
                model->freqs_cis
            );
            if (rope_err != VOXCPM_SUCCESS) {
                tensor_free(model->freqs_cis);
                model->freqs_cis = NULL;
            }
        }

        // Load AudioVAE from companion file (e.g., model_audiovae.vxcpm)
        if (config->model_path) {
            // Derive audiovae path: replace ".vxcpm" with "_audiovae.vxcpm"
            const char* model_path = config->model_path;
            size_t path_len = strlen(model_path);
            char* av_path = (char*)malloc(path_len + 16);
            if (av_path) {
                // Check for .vxcpm extension
                const char* dot = strstr(model_path, ".vxcpm");
                if (dot) {
                    size_t base_len = (size_t)(dot - model_path);
                    memcpy(av_path, model_path, base_len);
                    memcpy(av_path + base_len, "_audiovae.vxcpm", 16);
                } else {
                    // Fallback: append _audiovae.vxcpm
                    memcpy(av_path, model_path, path_len);
                    memcpy(av_path + path_len, "_audiovae.vxcpm", 16);
                }

                if (file_exists(av_path)) {
                    model->audiovae_mmap = mmap_open(av_path);
                    if (model->audiovae_mmap) {
                        model->audiovae_index = weight_index_build(
                            (const uint8_t*)model->audiovae_mmap->addr,
                            model->audiovae_mmap->size);
                        if (model->audiovae_index) {
                            LOG_INFO("AudioVAE loaded: %s (%d tensors)",
                                     av_path, model->audiovae_index->count);
                            model->audio_vae = audio_vae_create(
                                &model->config,
                                (const uint8_t*)model->audiovae_mmap->addr,
                                model->audiovae_index);
                            if (!model->audio_vae) {
                                LOG_WARN("AudioVAE creation from %s failed", av_path);
                            }
                        } else {
                            LOG_WARN("AudioVAE weight index parse failed: %s", av_path);
                        }
                    } else {
                        LOG_WARN("Failed to mmap AudioVAE file: %s", av_path);
                    }
                } else {
                    LOG_INFO("AudioVAE companion file not found: %s", av_path);
                }
                free(av_path);
            }
        }
    }

    // Initialize remaining NULL fields
    model->text_embed = NULL;
    model->audio_embed = NULL;
    model->tokenizer = NULL;
    model->gpu_context = NULL;

    // Initialize tokenizer from weights or fallback file
    if (model->weight_index && model->weights_mmap) {
        // First try loading from .vxcpm weight data
        model->tokenizer = tokenizer_create_from_weights(
            (const uint8_t*)model->weights_mmap->addr,
            model->weight_index);

        // Fallback: try tokenizer.bin alongside model file
        if (!model->tokenizer && config->model_path) {
            // Derive tokenizer.bin path from model path
            // Replace model filename with "tokenizer.bin"
            const char* model_path = config->model_path;
            const char* last_sep = strrchr(model_path, '/');
            const char* last_sep2 = strrchr(model_path, '\\');
            if (last_sep2 > last_sep) last_sep = last_sep2;

            char* dir = NULL;
            if (last_sep) {
                size_t dir_len = (size_t)(last_sep - model_path);
                dir = (char*)malloc(dir_len + 1);
                if (dir) {
                    memcpy(dir, model_path, dir_len);
                    dir[dir_len] = '\0';
                }
            } else {
                dir = (char*)malloc(2);
                if (dir) {
                    dir[0] = '.';
                    dir[1] = '\0';
                }
            }

            if (dir) {
                char* tok_path = path_join(dir, "tokenizer.bin");
                if (tok_path) {
                    if (file_exists(tok_path)) {
                        model->tokenizer = tokenizer_create_from_file(tok_path);
                        if (model->tokenizer) {
                            LOG_INFO("Tokenizer loaded from: %s", tok_path);
                        } else {
                            LOG_WARN("Failed to load tokenizer from: %s", tok_path);
                        }
                    } else {
                        LOG_INFO("Tokenizer file not found: %s (using UNK-only fallback)", tok_path);
                    }
                    free(tok_path);
                }
                free(dir);
            }
        } else if (model->tokenizer) {
            LOG_INFO("Tokenizer loaded from model weights");
        } else {
            LOG_WARN("No tokenizer found in weights or file");
        }
    }

    // ─── Release mmap regions ──────────────────────────────
    // All weight data has been deep-copied into tensors via weight_load_tensor.
    // The mmap regions (4.36 GB fp16 model → 8.72 GB fp32 tensors) are no longer
    // needed and can be freed to reduce memory pressure.
    if (model->weights_mmap) {
        mmap_close(model->weights_mmap);
        model->weights_mmap = NULL;
    }
    if (model->audiovae_mmap) {
        mmap_close(model->audiovae_mmap);
        model->audiovae_mmap = NULL;
    }

    if (err) *err = VOXCPM_SUCCESS;
    return model;
}

void voxcpm_free(VoxCPMModel* model) {
    if (!model) return;

    loc_enc_free(model->loc_enc);
    tslm_free(model->tslm);
    ralm_free(model->ralm);
    loc_dit_free(model->loc_dit);
    audio_vae_free(model->audio_vae);
    tensor_free(model->text_embed);
    tensor_free(model->audio_embed);
    tensor_free(model->lm_to_dit_weight);
    tensor_free(model->lm_to_dit_bias);
    tensor_free(model->enc_to_lm_proj_weight);
    tensor_free(model->enc_to_lm_proj_bias);
    tensor_free(model->res_to_dit_proj_weight);
    tensor_free(model->res_to_dit_proj_bias);
    tensor_free(model->fusion_concat_proj_weight);
    tensor_free(model->fusion_concat_proj_bias);
    tensor_free(model->freqs_cis);
    weight_index_free(model->weight_index);
    model->weight_index = NULL;
    weight_index_free(model->audiovae_index);
    model->audiovae_index = NULL;
    mmap_close(model->audiovae_mmap);
    mmap_close(model->weights_mmap);
    thread_pool_free(model->thread_pool);
    tokenizer_free(model->tokenizer);
    model->tokenizer = NULL;
    free(model);
}

/* ═══════════════════════════════════════════════════════════════
 * GPU upload — recursively upload all model weights to GPU
 * ═══════════════════════════════════════════════════════════════ */
#ifdef VOXCPM_CUDA
VoxCPMError voxcpm_to_cuda(VoxCPMModel* model) {
    if (!model) return VOXCPM_ERR_INTERNAL;

    VoxCPMError err;

    // Initialize CUDA runtime
    err = tensor_cuda_init();
    if (err) {
        LOG_ERROR("GPU init failed: err=%d", err);
        return err;
    }

    // Top-level tensor fields
    // Large weight matrices: upload to GPU for tensor_matmul_nt acceleration.
    // Bias tensors and freqs_cis stay on CPU — they are read directly by
    // CPU code (bias addition, tensor_rotary_emb). Tiny size, no GPU benefit.
    err = tensor_to_cuda(model->lm_to_dit_weight);
    if (err) LOG_WARN("lm_to_dit_weight upload failed: err=%d", err);
    // lm_to_dit_bias — keep on CPU
    err = tensor_to_cuda(model->enc_to_lm_proj_weight);
    if (err) LOG_WARN("enc_to_lm_proj_weight upload failed: err=%d", err);
    // enc_to_lm_proj_bias — keep on CPU
    err = tensor_to_cuda(model->res_to_dit_proj_weight);
    if (err) LOG_WARN("res_to_dit_proj_weight upload failed: err=%d", err);
    // res_to_dit_proj_bias — keep on CPU
    err = tensor_to_cuda(model->fusion_concat_proj_weight);
    if (err) LOG_WARN("fusion_concat_proj_weight upload failed: err=%d", err);
    // fusion_concat_proj_bias — keep on CPU
    // freqs_cis — keep on CPU (read by tensor_rotary_emb)
    if (model->text_embed) {
        err = tensor_to_cuda(model->text_embed);
        if (err) LOG_WARN("text_embed upload failed: err=%d", err);
    }
    if (model->audio_embed) {
        err = tensor_to_cuda(model->audio_embed);
        if (err) LOG_WARN("audio_embed upload failed: err=%d", err);
    }

    // Sub-module uploads
    if (model->tslm) {
        // tslm_to_cuda keeps embed_weight and KV cache on CPU (CPU reads them directly)
        err = tslm_to_cuda(model->tslm);
        if (err) LOG_WARN("tslm_to_cuda failed: err=%d", err);
    }
    if (model->ralm) {
        err = ralm_to_cuda(model->ralm);
        if (err) LOG_WARN("ralm_to_cuda failed: err=%d", err);
    }
    if (model->loc_enc) {
        err = loc_enc_to_cuda(model->loc_enc);
        if (err) LOG_WARN("loc_enc_to_cuda failed: err=%d", err);
    }
    if (model->loc_dit) {
        err = loc_dit_to_cuda(model->loc_dit);
        if (err) LOG_WARN("loc_dit_to_cuda failed: err=%d", err);
    }
    // AudioVAE uses custom convolution (not tensor_matmul_nt) that reads
    // weight->data directly on CPU. Keep all AudioVAE weights on CPU.
    // AudioVAE encode/decode are a small portion of total inference compute.

    LOG_INFO("GPU upload complete");
    return VOXCPM_SUCCESS;
}
#else
VoxCPMError voxcpm_to_cuda(VoxCPMModel* model) {
    (void)model;
    LOG_WARN("GPU upload not available (compiled without VOXCPM_CUDA)");
    return VOXCPM_ERR_CUDA_NOT_FOUND;
}
#endif /* VOXCPM_CUDA */

char* voxcpm_model_info(const VoxCPMModel* model) {
    if (!model) return NULL;

    // Build JSON info string
    const VoxCPMConfig* cfg = &model->config;
    int len = snprintf(NULL, 0,
        "{"
        "\"d_model\":%d,"
        "\"n_layers_tslm\":%d,"
        "\"n_layers_ralm\":%d,"
        "\"n_layers_loc_dit\":%d,"
        "\"n_heads\":%d,"
        "\"n_kv_heads\":%d,"
        "\"max_seq_len\":%d,"
        "\"sample_rate\":%d,"
        "\"gpu_enabled\":%s"
        "}",
        cfg->d_model,
        cfg->n_layers_tslm,
        cfg->n_layers_ralm,
        cfg->n_layers_loc_dit,
        cfg->n_heads,
        cfg->n_kv_heads,
        cfg->max_seq_len,
        cfg->sample_rate,
        model->gpu_enabled ? "true" : "false"
    );
    if (len < 0) return NULL;

    char* result = (char*)malloc((size_t)len + 1);
    if (result) {
        snprintf(result, (size_t)len + 1,
            "{"
            "\"d_model\":%d,"
            "\"n_layers_tslm\":%d,"
            "\"n_layers_ralm\":%d,"
            "\"n_layers_loc_dit\":%d,"
            "\"n_heads\":%d,"
            "\"n_kv_heads\":%d,"
            "\"max_seq_len\":%d,"
            "\"sample_rate\":%d,"
            "\"gpu_enabled\":%s"
            "}",
            cfg->d_model,
            cfg->n_layers_tslm,
            cfg->n_layers_ralm,
            cfg->n_layers_loc_dit,
            cfg->n_heads,
            cfg->n_kv_heads,
            cfg->max_seq_len,
            cfg->sample_rate,
            model->gpu_enabled ? "true" : "false"
        );
    }
    return result;
}

/* ═══════════════════════════════════════════════════════════════
 * Public API — Generation
 * ═══════════════════════════════════════════════════════════════ */

VoxCPMError voxcpm_generate(
    VoxCPMModel* model,
    const VoxCPMGenConfig* config,
    VoxCPMAudio* output)
{
    if (!model || !config || !output) return VOXCPM_ERR_INTERNAL;
    if (!config->text || config->text[0] == '\0') return VOXCPM_ERR_INVALID_TEXT;

    // Check required sub-models exist
    if (!model->tslm || !model->loc_dit || !model->audio_vae) {
        LOG_ERROR("Model not fully loaded: missing sub-models");
        return VOXCPM_ERR_INTERNAL;
    }
    if (!model->tokenizer) {
        LOG_ERROR("Tokenizer not loaded");
        return VOXCPM_ERR_INTERNAL;
    }
    if (!model->freqs_cis) {
        LOG_ERROR("RoPE frequencies not precomputed");
        return VOXCPM_ERR_INTERNAL;
    }

    VoxCPMError err = VOXCPM_SUCCESS;
    const int B = 1;  // single batch for now
    const int F = model->config.latent_dim;  // 64
    const int P = model->config.patch_size;  // 4
    const int D = model->config.d_model;      // 2048
    const int DH = model->config.dit_hidden;  // 1024

    // ─── Step 1: Tokenize ───────────────────────────────────
    int buf_size = tokenizer_encode(model->tokenizer, config->text, NULL, 0);
    if (buf_size <= 0) {
        LOG_ERROR("Tokenization failed or empty text");
        return VOXCPM_ERR_INVALID_TEXT;
    }

    int32_t* tokens = (int32_t*)malloc((size_t)buf_size * sizeof(int32_t));
    if (!tokens) return VOXCPM_ERR_OOM;

    int n_tokens = tokenizer_encode(model->tokenizer, config->text, tokens, buf_size);
    if (n_tokens <= 0) {
        LOG_ERROR("Tokenization failed");
        free(tokens);
        return VOXCPM_ERR_INVALID_TEXT;
    }
    LOG_INFO("Tokenized text: %d tokens", n_tokens);

    // ─── Step 2: Embed tokens ───────────────────────────────
    LOG_INFO("Tokenized: %d tokens, creating hidden tensor [%d,%d,%d]...",
             n_tokens, B, n_tokens, D);
    // embed_weight: [vocab_size, d_model]
    Tensor* hidden = tensor_create(3, (int[]){ B, n_tokens, D });
    if (!hidden) { free(tokens); return VOXCPM_ERR_OOM; }

    Tensor* embed = model->tslm->embed_weight;
    for (int s = 0; s < n_tokens; s++) {
        int token_id = tokens[s];
        if (token_id < 0 || token_id >= embed->shape[0]) {
            token_id = 0;  // fallback to UNK
        }
        float* dst = hidden->data + (size_t)s * (size_t)D;
        if (embed->is_fp16) {
            const uint16_t* src16 = embed->data_fp16 + (size_t)token_id * (size_t)D;
            for (int j = 0; j < D; j++) {
                dst[j] = fp16_to_fp32(src16[j]);
            }
        } else {
            float* src = embed->data + (size_t)token_id * (size_t)D;
            memcpy(dst, src, (size_t)D * sizeof(float));
        }
    }
    free(tokens);

    // ─── Step 3: Setup KV caches ────────────────────────────
    // Use a smaller max_seq_len for KV cache to avoid OOM from heap fragmentation.
    // TTS only needs a few hundred tokens; 2048 is more than generous.
    // Keep max_seq_len at 2048 — do NOT restore the original 32768 — because
    // tslm_forward / layer_cache_slice use tslm->max_seq_len to compute per-layer
    // offsets into the 5D cache tensor. A mismatch causes out-of-bounds access.
    const int kv_max_seq = 2048;
    model->tslm->max_seq_len = kv_max_seq;
    LOG_INFO("Setting up TSLM KV cache (n_layers=%d, kv_heads=%d, max_seq=%d, head_dim=%d)...",
             model->tslm->n_layers, model->tslm->n_kv_heads,
             model->tslm->max_seq_len, model->tslm->head_dim);
    err = tslm_setup_cache(model->tslm, B);
    if (err) { LOG_ERROR("TSLM cache setup failed (OOM)"); tensor_free(hidden); return err; }
    LOG_INFO("TSLM KV cache created");

    if (model->ralm) {
        model->ralm->max_seq_len = kv_max_seq;
        LOG_INFO("Setting up RALM KV cache (n_layers=%d, kv_heads=%d, max_seq=%d, head_dim=%d)...",
                 model->ralm->n_layers, model->ralm->n_kv_heads,
                 model->ralm->max_seq_len, model->ralm->head_dim);
        err = ralm_setup_cache(model->ralm, B);
        if (err) { LOG_ERROR("RALM cache setup failed (OOM)"); tensor_free(hidden); return err; }
        LOG_INFO("RALM KV cache created");
    }

    // ─── Step 4: TSLM prefill ───────────────────────────────
    Tensor* freqs_cis = model->freqs_cis;

    LOG_INFO("TSLM forward: hidden [%d,%d,%d]...", B, n_tokens, D);
    err = tslm_forward(model->tslm, hidden, freqs_cis, hidden);
    if (err) { LOG_ERROR("tslm_forward failed: err=%d", err); tensor_free(hidden); return err; }

    // ─── Step 5: RALM forward (residual) ────────────────────
    if (model->ralm) {
        Tensor* ralm_out = tensor_create(3, (int[]){ B, n_tokens, D });
        if (!ralm_out) { LOG_ERROR("ralm_out OOM"); tensor_free(hidden); return VOXCPM_ERR_OOM; }

        LOG_INFO("RALM forward...");
        err = ralm_forward(model->ralm, hidden, NULL, ralm_out);
        if (err) { LOG_ERROR("ralm_forward failed: err=%d", err); tensor_free(hidden); tensor_free(ralm_out); return err; }

        // Add residual: hidden += ralm_out
        err = tensor_add(hidden, ralm_out, hidden);
        tensor_free(ralm_out);
        if (err) { LOG_ERROR("tensor_add failed: err=%d", err); tensor_free(hidden); return err; }
    }

    // ─── Step 6: Save text hidden for future fusion ──────────
    // clone before freeing — we still need hidden for mu extraction below
    Tensor* text_hidden = tensor_clone(hidden);
    if (!text_hidden) { LOG_ERROR("text_hidden OOM"); tensor_free(hidden); return VOXCPM_ERR_OOM; }

    // ─── Step 7: Extract mu vector from text hidden ─────────
    Tensor* last_hidden = tensor_create(2, (int[]){ B, D });
    if (!last_hidden) { tensor_free(hidden); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
    {
        float* src = hidden->data + (size_t)(n_tokens - 1) * (size_t)D;
        memcpy(last_hidden->data, src, (size_t)D * sizeof(float));
    }
    tensor_free(hidden);  // hidden no longer needed

    Tensor* mu = NULL;
    if (model->lm_to_dit_weight) {
        mu = tensor_create(2, (int[]){ B, DH });
        if (!mu) { tensor_free(last_hidden); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
        err = tensor_matmul_nt(last_hidden, model->lm_to_dit_weight, mu);
        if (err) { tensor_free(last_hidden); tensor_free(mu); tensor_free(text_hidden); return err; }
        if (model->lm_to_dit_bias) {
            if (model->lm_to_dit_bias->is_fp16) {
                for (int i = 0; i < DH; i++)
                    mu->data[i] += fp16_to_fp32(model->lm_to_dit_bias->data_fp16[i]);
            } else {
                for (int i = 0; i < DH; i++)
                    mu->data[i] += model->lm_to_dit_bias->data[i];
            }
        }
    } else {
        mu = tensor_create(2, (int[]){ B, DH });
        if (!mu) { tensor_free(last_hidden); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
        memcpy(mu->data, last_hidden->data, (size_t)DH * sizeof(float));
        LOG_WARN("lm_to_dit_proj not found; using first-half slice as mu");
    }
    tensor_free(last_hidden);

    // ─── Step 8: Autoregressive patch generation loop ────────
    // Determine number of patches from max_new_tokens.
    // Each patch = P VAE frames = ~0.16s of 48kHz audio.
    const int n_patches = config->max_new_tokens > 0
        ? (config->max_new_tokens + P - 1) / P  // ceil division
        : 4;   // default: ~0.6 seconds

    LOG_INFO("Generating %d patches (%d tokens, mu=[%d,%d], text_len=%d)...",
             n_patches, config->max_new_tokens, B, DH, text_hidden->shape[1]);

    // Compute samples-per-patch from AudioVAE decoder strides
    int samples_per_patch = P * 1920; // fallback
    if (model->audio_vae && model->audio_vae->decoder) {
        AudioVAEDecoder* dec = model->audio_vae->decoder;
        int T = P;
        for (int i = 0; i < AUDIOVAE_NUM_DECODER_BLOCKS; i++) {
            const WNConv* ct = dec->decoder_blocks[i].convtr;
            T = (T - 1) * ct->stride - 2 * ct->padding + ct->kernel_size;
        }
        samples_per_patch = T;
    }

    // Allocate output buffer (accumulate all patches)
    size_t max_total = (size_t)n_patches * (size_t)samples_per_patch;
    float* all_samples = (float*)malloc(max_total * sizeof(float));
    if (!all_samples) { tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
    size_t total_samples = 0;

    // Patch loop — cond and acc_latent managed across iterations
    Tensor* cond = NULL;
    Tensor* acc_latent = NULL;
    for (int p = 0; p < n_patches; p++) {
        LOG_INFO("Patch %d/%d...", p + 1, n_patches);

        // ─── Build cond ───
        // Patch 0: empty cond (no audio context yet)
        // Patches 1+: cond set from previous iteration's audio encoding
        if (p == 0) {
            cond = tensor_create(3, (int[]){ B, F, 0 });
            if (!cond) { free(all_samples); tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
        }
        // else: cond is already set from the encoding step at end of previous iteration

        // ─── LocDiT diffusion sample ───
        Tensor* latent = tensor_create(3, (int[]){ B, F, P });
        if (!latent) { tensor_free(cond); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }

        err = loc_dit_sample(
            model->loc_dit, mu, cond,
            config->inference_timesteps > 0 ? config->inference_timesteps : 10,
            config->cfg_value > 0.0f ? config->cfg_value : 2.0f,
            latent
        );
        tensor_free(cond); cond = NULL;
        if (err) { LOG_ERROR("loc_dit_sample failed at patch %d", p); tensor_free(latent); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return err; }

        LOG_INFO("Patch %d: latent permute...", p);
        // ─── Permute latent for AudioVAE ───
        Tensor* latent_vae = tensor_create(3, (int[]){ B, P, F });
        if (!latent_vae) { tensor_free(latent); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
        err = tensor_permute(latent, latent_vae, (int[]){ 0, 2, 1 });
        tensor_free(latent);
        if (err) { tensor_free(latent_vae); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return err; }

        LOG_INFO("Patch %d: AudioVAE decode...", p);
        // ─── AudioVAE decode ───
        Tensor* waveform = tensor_create(1, (int[]){ B * samples_per_patch });
        if (!waveform) { tensor_free(latent_vae); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
        if (model->audio_vae) {
            err = audio_vae_decode(model->audio_vae, latent_vae, waveform);
        } else {
            tensor_zero(waveform);
            err = VOXCPM_SUCCESS;
        }
        LOG_INFO("Patch %d: AudioVAE decode done, err=%d", p, err);
        if (err) { tensor_free(latent_vae); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return err; }

        // ─── Accumulate latent for autoregressive conditioning ──
        // Use latent_vae (before free) as AR context for next patches.
        // Also update mu via LocEnc → fusion → RALM.
        if (p + 1 < n_patches) {
            int T_enc_prev = acc_latent ? acc_latent->shape[1] : 0;
            Tensor* new_acc = tensor_create(3, (int[]){ B, T_enc_prev + P, F });
            if (!new_acc) {
                tensor_free(latent_vae); tensor_free(waveform);
                free(all_samples); tensor_free(mu); tensor_free(text_hidden);
                return VOXCPM_ERR_OOM;
            }
            if (acc_latent) {
                memcpy(new_acc->data, acc_latent->data,
                       (size_t)B * (size_t)T_enc_prev * (size_t)F * sizeof(float));
                tensor_free(acc_latent);
            }
            memcpy(new_acc->data + (size_t)B * (size_t)T_enc_prev * (size_t)F,
                   latent_vae->data,
                   (size_t)B * (size_t)P * (size_t)F * sizeof(float));
            acc_latent = new_acc;

            int T_enc = acc_latent->shape[1];
            LOG_INFO("AR: accumulated latent [%d,%d,%d]", B, T_enc, F);

            // Build cond: permute [B,T_enc,F] → [B,F,T_enc]
            cond = tensor_create(3, (int[]){ B, F, T_enc });
            if (!cond) {
                tensor_free(latent_vae); tensor_free(waveform);
                free(all_samples); tensor_free(mu); tensor_free(text_hidden);
                return VOXCPM_ERR_OOM;
            }
            err = tensor_permute(acc_latent, cond, (int[]){ 0, 2, 1 });
            if (err) {
                LOG_ERROR("AR P01: permute failed");
                tensor_free(cond); tensor_free(latent_vae); tensor_free(waveform);
                free(all_samples); tensor_free(mu); tensor_free(text_hidden);
                return err;
            }

            // ─── Update mu via LocEnc → enc_to_lm_proj → fusion → RALM → res_to_dit_proj ───
            int T_groups = T_enc / P;
            if (T_groups > 0 && model->loc_enc && model->enc_to_lm_proj_weight
                && model->fusion_concat_proj_weight)
            {
                Tensor* loc_in = tensor_create(4, (int[]){ B, T_groups, P, F });
                if (!loc_in) {
                    tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform);
                    free(all_samples); tensor_free(mu); tensor_free(text_hidden);
                    return VOXCPM_ERR_OOM;
                }
                memcpy(loc_in->data, acc_latent->data,
                       (size_t)B * (size_t)T_groups * (size_t)P * (size_t)F * sizeof(float));

                Tensor* enc_out = tensor_create(3, (int[]){ B, T_groups, DH });
                if (!enc_out) { tensor_free(loc_in); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
                err = loc_enc_forward(model->loc_enc, loc_in, enc_out);
                tensor_free(loc_in);
                if (err) { tensor_free(enc_out); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return err; }

                // enc_to_lm_proj [B,T_groups,1024] → [B,T_groups,2048]
                Tensor* enc_proj = tensor_create(3, (int[]){ B, T_groups, D });
                if (!enc_proj) { tensor_free(enc_out); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
                {   Tensor* fi  = tensor_create_from_buffer(2, (int[]){B*T_groups, DH}, enc_out->data);
                    Tensor* fo  = tensor_create_from_buffer(2, (int[]){B*T_groups, D}, enc_proj->data);
                    if (!fi||!fo) { tensor_free(fi); tensor_free(fo); tensor_free(enc_out); tensor_free(enc_proj); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
                    err = tensor_matmul_nt(fi, model->enc_to_lm_proj_weight, fo);
                    tensor_free(fi); tensor_free(fo);
                    if (err) { tensor_free(enc_out); tensor_free(enc_proj); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return err; }
                    if (model->enc_to_lm_proj_bias) {
                        float* b = model->enc_to_lm_proj_bias->data;
                        for (int bi = 0; bi < B*T_groups; bi++) {
                            float* r = enc_proj->data + (size_t)bi * D;
                            for (int j = 0; j < D; j++) r[j] += b[j];
                        }
                    }
                }
                tensor_free(enc_out);

                // Fusion: concat enc_proj with text_hidden
                int n_tok = text_hidden->shape[1];
                int fused_T = (n_tok < T_groups) ? n_tok : T_groups;
                Tensor* fused_in = tensor_create(3, (int[]){ B, fused_T, D*2 });
                if (!fused_in) { tensor_free(enc_proj); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
                for (int bi = 0; bi < B; bi++) {
                    for (int ti = 0; ti < fused_T; ti++) {
                        float* d = fused_in->data + (size_t)bi*fused_T*D*2 + (size_t)ti*D*2;
                        int ai = T_groups - fused_T + ti;
                        int txti = n_tok - fused_T + ti;
                        memcpy(d, enc_proj->data + (size_t)bi*T_groups*D + (size_t)ai*D, (size_t)D*sizeof(float));
                        memcpy(d+D, text_hidden->data + (size_t)bi*n_tok*D + (size_t)txti*D, (size_t)D*sizeof(float));
                    }
                }

                Tensor* fused_out = tensor_create(3, (int[]){ B, fused_T, D });
                if (!fused_out) { tensor_free(fused_in); tensor_free(enc_proj); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
                {   Tensor* fi2 = tensor_create_from_buffer(2, (int[]){B*fused_T, D*2}, fused_in->data);
                    Tensor* fo2 = tensor_create_from_buffer(2, (int[]){B*fused_T, D}, fused_out->data);
                    if (!fi2||!fo2) { tensor_free(fi2); tensor_free(fo2); tensor_free(fused_in); tensor_free(fused_out); tensor_free(enc_proj); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
                    err = tensor_matmul_nt(fi2, model->fusion_concat_proj_weight, fo2);
                    tensor_free(fi2); tensor_free(fo2);
                    if (err) { tensor_free(fused_in); tensor_free(fused_out); tensor_free(enc_proj); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return err; }
                    if (model->fusion_concat_proj_bias) {
                        float* b = model->fusion_concat_proj_bias->data;
                        for (int bi = 0; bi < B*fused_T; bi++) {
                            float* r = fused_out->data + (size_t)bi*D;
                            for (int j = 0; j < D; j++) r[j] += b[j];
                        }
                    }
                }
                tensor_free(fused_in);
                tensor_free(enc_proj);

                // RALM forward (residual)
                Tensor* ralm_out = tensor_create(3, (int[]){ B, fused_T, D });
                if (!ralm_out) { tensor_free(fused_out); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
                err = ralm_forward(model->ralm, fused_out, NULL, ralm_out);
                if (err) { tensor_free(ralm_out); tensor_free(fused_out); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return err; }
                err = tensor_add(fused_out, ralm_out, fused_out);
                tensor_free(ralm_out);
                if (err) { tensor_free(fused_out); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return err; }

                // Last hidden → res_to_dit_proj → new mu
                Tensor* last_fused = tensor_create(2, (int[]){ B, D });
                if (!last_fused) { tensor_free(fused_out); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
                memcpy(last_fused->data, fused_out->data + (size_t)(fused_T-1)*D, (size_t)D*sizeof(float));
                tensor_free(fused_out);

                if (model->res_to_dit_proj_weight) {
                    Tensor* new_mu = tensor_create(2, (int[]){ B, DH });
                    if (!new_mu) { tensor_free(last_fused); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
                    err = tensor_matmul_nt(last_fused, model->res_to_dit_proj_weight, new_mu);
                    tensor_free(last_fused);
                    if (err) { tensor_free(new_mu); tensor_free(latent_vae); tensor_free(cond); tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return err; }
                    if (model->res_to_dit_proj_bias) {
                        float* b = model->res_to_dit_proj_bias->data;
                        for (int i = 0; i < B*DH; i++) new_mu->data[i] += b[i];
                    }
                    tensor_free(mu);
                    mu = new_mu;
                    LOG_INFO("AR: mu updated via fusion→RALM→res_to_dit_proj");
                } else {
                    tensor_free(last_fused);
                }
            }
        }

        // ─── Now safe to free latent_vae ───
        tensor_free(latent_vae);

        // ─── Append to output ───
        int n_samp = (int)waveform->size;
        if (total_samples + (size_t)n_samp > max_total) {
            max_total = total_samples + (size_t)n_samp;
            float* tmp = (float*)realloc(all_samples, max_total * sizeof(float));
            if (!tmp) { tensor_free(waveform); free(all_samples); tensor_free(mu); tensor_free(text_hidden); return VOXCPM_ERR_OOM; }
            all_samples = tmp;
        }
        memcpy(all_samples + total_samples, waveform->data, (size_t)n_samp * sizeof(float));
        total_samples += (size_t)n_samp;

        tensor_free(waveform);
    }

    // Cleanup accumulated latent
    tensor_free(acc_latent);

    tensor_free(mu);
    tensor_free(text_hidden);

    // ─── Step 9: Normalize audio (peak to [-1, 1]) ──────────
    {
        Tensor* norm_t = tensor_create_from_buffer(1, (int[]){ (int)total_samples }, all_samples);
        if (norm_t) {
            audio_normalize(norm_t);
            tensor_free(norm_t);
        }
    }

    // ─── Step 10: Copy to VoxCPMAudio ───────────────────────
    output->num_samples = (int)total_samples;
    output->sample_rate = model->config.sample_rate;
    output->samples = all_samples;  // transfer ownership

    return VOXCPM_SUCCESS;
}

VoxCPMError voxcpm_generate_streaming(
    VoxCPMModel* model,
    const VoxCPMGenConfig* config,
    voxcpm_stream_callback callback,
    void* user_data)
{
    if (!model || !config || !callback) return VOXCPM_ERR_INTERNAL;

    LOG_INFO("voxcpm_generate_streaming: text=\"%s\" (stub)", config->text);

    // Generate 5 chunks of silence
    int sample_rate = 48000;
    int chunk_samples = sample_rate / 10; // 100ms chunks
    float* chunk = (float*)calloc((size_t)chunk_samples, sizeof(float));
    if (!chunk) return VOXCPM_ERR_OOM;

    for (int i = 0; i < 5; i++) {
        if (!callback(chunk, chunk_samples, user_data)) break;
    }

    free(chunk);
    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * Public API — Utility
 * ═══════════════════════════════════════════════════════════════ */

size_t voxcpm_memory_used(const VoxCPMModel* model) {
    (void)model;
    return 0; // Will be implemented properly
}

size_t voxcpm_gpu_memory_used(const VoxCPMModel* model) {
    (void)model;
    return 0;
}

size_t voxcpm_gpu_memory_total(const VoxCPMModel* model) {
    (void)model;
    return 0;
}

void voxcpm_set_timeout(VoxCPMModel* model, int timeout_ms) {
    if (model) model->timeout_ms = timeout_ms;
}

void voxcpm_cancel(VoxCPMModel* model) {
    if (model) model->cancel_requested = true;
}

VoxCPMError voxcpm_tts(const char* model_path, const char* text, const char* output_path) {
    VoxCPMModelConfig cfg = voxcpm_model_config_default();
    cfg.model_path = model_path;

    VoxCPMError err;
    VoxCPMModel* model = voxcpm_create(&cfg, &err);
    if (!model) return err;

    VoxCPMGenConfig gen_cfg = voxcpm_gen_config_default();
    gen_cfg.text = text;

    VoxCPMAudio audio;
    memset(&audio, 0, sizeof(audio));
    err = voxcpm_generate(model, &gen_cfg, &audio);

    if (err == VOXCPM_SUCCESS) {
        err = voxcpm_audio_save(&audio, output_path);
        voxcpm_audio_free(&audio);
    }

    voxcpm_free(model);
    return err;
}

/* ═══════════════════════════════════════════════════════════════
 * VoxCPMConfig defaults
 * ═══════════════════════════════════════════════════════════════ */

VoxCPMConfig voxcpm_config_default(void) {
    VoxCPMConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    /* VoxCPM2 actual architecture (from openbmb/VoxCPM2 config.json) */
    cfg.d_model = 2048;
    cfg.n_heads = 16;
    cfg.n_kv_heads = 2;
    cfg.d_ff = 6144;
    cfg.n_layers_tslm = 28;    /* base_lm: 28 layers */
    cfg.n_layers_ralm = 8;     /* residual_lm: 8 layers */
    cfg.n_layers_loc_dit = 12; /* DiT: 12 layers */
    cfg.n_layers_loc_enc = 12; /* encoder: 12 layers */
    cfg.max_seq_len = 32768;   /* max_position_embeddings */
    cfg.vocab_size = 73448;    /* actual vocab size */
    cfg.latent_dim = 64;       /* feat_dim */
    cfg.patch_size = 4;
    cfg.sample_rate = 48000;   /* AudioVAE output sample rate */
    cfg.vae_encode_sr = 16000; /* AudioVAE input sample rate */
    cfg.n_threads = 4;
    cfg.use_gpu = false;
    cfg.rope_theta = 10000.0f;
    cfg.norm_eps = 1e-5f;
    cfg.head_dim = 128;        /* kv_channels = 128 */
    /* Encoder (LocEnc) defaults */
    cfg.enc_hidden = 1024;
    cfg.enc_d_ff = 4096;
    cfg.enc_n_heads = 16;
    cfg.enc_n_layers = 12;
    cfg.enc_head_dim = 128;
    /* DiT defaults */
    cfg.dit_hidden = 1024;
    cfg.dit_d_ff = 4096;
    cfg.dit_n_heads = 8;
    cfg.dit_n_layers = 12;
    cfg.dit_head_dim = 128;
    return cfg;
}

/* ═══════════════════════════════════════════════════════════════
 * Helper: load a weight tensor by name from the weight index
 * ═══════════════════════════════════════════════════════════════ */

static Tensor* load_weight(const char* name, const uint8_t* mmap_data, const WeightIndex* idx) {
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

/* ═══════════════════════════════════════════════════════════════
 * Helper: create a transformer block from weight names
 * ═══════════════════════════════════════════════════════════════ */

static TransformerBlock* transformer_block_from_weights(
    const char* prefix,
    const VoxCPMConfig* config,
    int head_dim,
    const uint8_t* mmap_data,
    const WeightIndex* idx)
{
    TransformerBlock* block = (TransformerBlock*)calloc(1, sizeof(TransformerBlock));
    if (!block) return NULL;

    char key[256];

    /* Input RMS norm */
    snprintf(key, sizeof(key), "%s.input_layernorm.weight", prefix);
    Tensor* rms_w = load_weight(key, mmap_data, idx);
    if (!rms_w) { transformer_block_free(block); return NULL; }
    block->rms_attn = (RmsNorm*)calloc(1, sizeof(RmsNorm));
    if (!block->rms_attn) { transformer_block_free(block); return NULL; }
    block->rms_attn->weight = rms_w;
    block->rms_attn->eps = config->norm_eps;

    /* Attention */
    block->attn = (Attention*)calloc(1, sizeof(Attention));
    if (!block->attn) { transformer_block_free(block); return NULL; }
    block->attn->d_model = config->d_model;
    block->attn->n_heads = config->n_heads;
    block->attn->n_kv_heads = config->n_kv_heads;
    block->attn->head_dim = head_dim;

    snprintf(key, sizeof(key), "%s.self_attn.q_proj.weight", prefix);
    block->attn->wq = load_weight(key, mmap_data, idx);
    snprintf(key, sizeof(key), "%s.self_attn.k_proj.weight", prefix);
    block->attn->wk = load_weight(key, mmap_data, idx);
    snprintf(key, sizeof(key), "%s.self_attn.v_proj.weight", prefix);
    block->attn->wv = load_weight(key, mmap_data, idx);
    snprintf(key, sizeof(key), "%s.self_attn.o_proj.weight", prefix);
    block->attn->wo = load_weight(key, mmap_data, idx);

    if (!block->attn->wq || !block->attn->wk || !block->attn->wv || !block->attn->wo) {
        transformer_block_free(block); return NULL;
    }

    /* SwiGLU FFN */
    int d_ff = config->d_ff; /* default to lm_config intermediate */
    block->ffn = (SwiGLU*)calloc(1, sizeof(SwiGLU));
    if (!block->ffn) { transformer_block_free(block); return NULL; }
    block->ffn->d_model = config->d_model;
    block->ffn->d_ff = d_ff;

    snprintf(key, sizeof(key), "%s.mlp.gate_proj.weight", prefix);
    block->ffn->w1 = load_weight(key, mmap_data, idx);
    snprintf(key, sizeof(key), "%s.mlp.down_proj.weight", prefix);
    block->ffn->w2 = load_weight(key, mmap_data, idx);
    snprintf(key, sizeof(key), "%s.mlp.up_proj.weight", prefix);
    block->ffn->w3 = load_weight(key, mmap_data, idx);

    if (!block->ffn->w1 || !block->ffn->w2 || !block->ffn->w3) {
        transformer_block_free(block); return NULL;
    }

    /* Post-attention RMS norm */
    snprintf(key, sizeof(key), "%s.post_attention_layernorm.weight", prefix);
    rms_w = load_weight(key, mmap_data, idx);
    if (!rms_w) { transformer_block_free(block); return NULL; }
    block->rms_ffn = (RmsNorm*)calloc(1, sizeof(RmsNorm));
    if (!block->rms_ffn) { transformer_block_free(block); return NULL; }
    block->rms_ffn->weight = rms_w;
    block->rms_ffn->eps = config->norm_eps;

    return block;
}

/* ═══════════════════════════════════════════════════════════════
 * Helper: create encoder-style transformer block
 * (enc_hidden=1024, enc_d_ff=4096, enc_n_heads=16)
 * ═══════════════════════════════════════════════════════════════ */

static TransformerBlock* enc_block_from_weights(
    const char* prefix,
    const VoxCPMConfig* config,
    const uint8_t* mmap_data,
    const WeightIndex* idx)
{
    TransformerBlock* block = (TransformerBlock*)calloc(1, sizeof(TransformerBlock));
    if (!block) return NULL;
    memset(block, 0, sizeof(TransformerBlock));

    char key[256];

    snprintf(key, sizeof(key), "%s.input_layernorm.weight", prefix);
    Tensor* rms_w = load_weight(key, mmap_data, idx);
    if (!rms_w) { free(block); return NULL; }
    block->rms_attn = (RmsNorm*)calloc(1, sizeof(RmsNorm));
    block->rms_attn->weight = rms_w;
    block->rms_attn->eps = config->norm_eps;

    block->attn = (Attention*)calloc(1, sizeof(Attention));
    block->attn->d_model = config->enc_hidden;
    block->attn->n_heads = config->enc_n_heads;
    block->attn->n_kv_heads = config->n_kv_heads; /* same GQA pattern */
    block->attn->head_dim = config->enc_head_dim;

    snprintf(key, sizeof(key), "%s.self_attn.q_proj.weight", prefix);
    block->attn->wq = load_weight(key, mmap_data, idx);
    snprintf(key, sizeof(key), "%s.self_attn.k_proj.weight", prefix);
    block->attn->wk = load_weight(key, mmap_data, idx);
    snprintf(key, sizeof(key), "%s.self_attn.v_proj.weight", prefix);
    block->attn->wv = load_weight(key, mmap_data, idx);
    snprintf(key, sizeof(key), "%s.self_attn.o_proj.weight", prefix);
    block->attn->wo = load_weight(key, mmap_data, idx);

    block->ffn = (SwiGLU*)calloc(1, sizeof(SwiGLU));
    block->ffn->d_model = config->enc_hidden;
    block->ffn->d_ff = config->enc_d_ff;

    snprintf(key, sizeof(key), "%s.mlp.gate_proj.weight", prefix);
    block->ffn->w1 = load_weight(key, mmap_data, idx);
    snprintf(key, sizeof(key), "%s.mlp.down_proj.weight", prefix);
    block->ffn->w2 = load_weight(key, mmap_data, idx);
    snprintf(key, sizeof(key), "%s.mlp.up_proj.weight", prefix);
    block->ffn->w3 = load_weight(key, mmap_data, idx);

    snprintf(key, sizeof(key), "%s.post_attention_layernorm.weight", prefix);
    rms_w = load_weight(key, mmap_data, idx);
    if (!rms_w) { free(block); return NULL; }
    block->rms_ffn = (RmsNorm*)calloc(1, sizeof(RmsNorm));
    block->rms_ffn->weight = rms_w;
    block->rms_ffn->eps = config->norm_eps;

    return block;
}

/* ═══════════════════════════════════════════════════════════════
 * LocEnc — Create from weights
 * ═══════════════════════════════════════════════════════════════ */

LocEnc* loc_enc_create(const VoxCPMConfig* config, const uint8_t* mmap_data, const WeightIndex* idx) {
    if (!config || !mmap_data || !idx) return NULL;

    LocEnc* enc = (LocEnc*)calloc(1, sizeof(LocEnc));
    if (!enc) return NULL;

    enc->d_model = config->enc_hidden;
    enc->n_layers = config->enc_n_layers;
    enc->patch_size = config->patch_size;

    /* Load projection weights */
    enc->in_proj_weight = load_weight("feat_encoder.in_proj.weight", mmap_data, idx);
    enc->in_proj_bias = load_weight("feat_encoder.in_proj.bias", mmap_data, idx);

    /* Load special token */
    enc->special_token = load_weight("feat_encoder.special_token", mmap_data, idx);

    /* Create encoder layers (12) */
    enc->layers = (TransformerBlock*)calloc((size_t)enc->n_layers, sizeof(TransformerBlock));
    if (!enc->layers) { loc_enc_free(enc); return NULL; }

    for (int i = 0; i < enc->n_layers; i++) {
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "feat_encoder.encoder.layers.%d", i);
        TransformerBlock* block = enc_block_from_weights(prefix, config, mmap_data, idx);
        if (!block) { loc_enc_free(enc); return NULL; }
        enc->layers[i] = *block;
        free(block);
    }

    /* Load final norm */
    Tensor* norm_w = load_weight("feat_encoder.encoder.norm.weight", mmap_data, idx);
    if (!norm_w) { loc_enc_free(enc); return NULL; }
    enc->output_norm = (RmsNorm*)calloc(1, sizeof(RmsNorm));
    enc->output_norm->weight = norm_w;
    enc->output_norm->eps = config->norm_eps;

    return enc;
}

void loc_enc_free(LocEnc* enc) {
    if (!enc) return;
    tensor_free(enc->in_proj_weight);
    tensor_free(enc->in_proj_bias);
    tensor_free(enc->special_token);
    if (enc->layers) {
        for (int i = 0; i < enc->n_layers; i++) {
            transformer_block_free_sub(&enc->layers[i]);
        }
        free(enc->layers);
    }
    if (enc->output_norm) {
        tensor_free(enc->output_norm->weight);
        free(enc->output_norm);
    }
    free(enc);
}

/* ═══════════════════════════════════════════════════════════════
 * TSLM — Create from weights
 * ═══════════════════════════════════════════════════════════════ */

TSLM* tslm_create(const VoxCPMConfig* config, const uint8_t* mmap_data, const WeightIndex* idx) {
    if (!config || !mmap_data || !idx) return NULL;

    TSLM* tslm = (TSLM*)calloc(1, sizeof(TSLM));
    if (!tslm) return NULL;

    tslm->n_layers = config->n_layers_tslm;
    tslm->d_model = config->d_model;
    tslm->vocab_size = config->vocab_size;
    tslm->n_kv_heads = config->n_kv_heads;
    tslm->head_dim = config->head_dim;
    tslm->max_seq_len = config->max_seq_len;
    tslm->cache_len = 0;

    /* Load token embedding */
    tslm->embed_weight = load_weight("base_lm.embed_tokens.weight", mmap_data, idx);

    /* Create transformer layers (28) */
    tslm->layers = (TransformerBlock*)calloc((size_t)tslm->n_layers, sizeof(TransformerBlock));
    if (!tslm->layers) { tslm_free(tslm); return NULL; }

    for (int i = 0; i < tslm->n_layers; i++) {
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "base_lm.layers.%d", i);
        TransformerBlock* block = transformer_block_from_weights(prefix, config, config->head_dim, mmap_data, idx);
        if (!block) { tslm_free(tslm); return NULL; }
        tslm->layers[i] = *block;
        free(block);
    }

    /* Load final norm */
    Tensor* norm_w = load_weight("base_lm.norm.weight", mmap_data, idx);
    if (!norm_w) { tslm_free(tslm); return NULL; }
    tslm->output_norm = (RmsNorm*)calloc(1, sizeof(RmsNorm));
    tslm->output_norm->weight = norm_w;
    tslm->output_norm->eps = config->norm_eps;

    /* KV cache (allocated later by tslm_setup_cache) */
    tslm->cache_k = NULL;
    tslm->cache_v = NULL;

    return tslm;
}

void tslm_free(TSLM* tslm) {
    if (!tslm) return;
    tensor_free(tslm->embed_weight);
    if (tslm->layers) {
        for (int i = 0; i < tslm->n_layers; i++) {
            transformer_block_free_sub(&tslm->layers[i]);
        }
        free(tslm->layers);
    }
    if (tslm->output_norm) {
        tensor_free(tslm->output_norm->weight);
        free(tslm->output_norm);
    }
    tensor_free(tslm->cache_k);
    tensor_free(tslm->cache_v);
    free(tslm);
}

/* ═══════════════════════════════════════════════════════════════
 * RALM — Create from weights
 * ═══════════════════════════════════════════════════════════════ */

RALM* ralm_create(const VoxCPMConfig* config, const uint8_t* mmap_data, const WeightIndex* idx) {
    if (!config || !mmap_data || !idx) return NULL;

    RALM* ralm = (RALM*)calloc(1, sizeof(RALM));
    if (!ralm) return NULL;

    ralm->n_layers = config->n_layers_ralm;
    ralm->d_model = config->d_model;
    ralm->n_kv_heads = config->n_kv_heads;
    ralm->head_dim = config->head_dim;
    ralm->max_seq_len = config->max_seq_len;
    ralm->cache_len = 0;
    ralm->no_rope = true; /* residual_lm has no RoPE */

    /* Create transformer layers (8) */
    ralm->layers = (TransformerBlock*)calloc((size_t)ralm->n_layers, sizeof(TransformerBlock));
    if (!ralm->layers) { ralm_free(ralm); return NULL; }

    for (int i = 0; i < ralm->n_layers; i++) {
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "residual_lm.layers.%d", i);
        TransformerBlock* block = transformer_block_from_weights(prefix, config, config->head_dim, mmap_data, idx);
        if (!block) { ralm_free(ralm); return NULL; }
        ralm->layers[i] = *block;
        free(block);
    }

    /* Load final norm */
    Tensor* norm_w = load_weight("residual_lm.norm.weight", mmap_data, idx);
    if (!norm_w) { ralm_free(ralm); return NULL; }
    ralm->output_norm = (RmsNorm*)calloc(1, sizeof(RmsNorm));
    ralm->output_norm->weight = norm_w;
    ralm->output_norm->eps = config->norm_eps;

    ralm->cache_k = NULL;
    ralm->cache_v = NULL;

    return ralm;
}

void ralm_free(RALM* ralm) {
    if (!ralm) return;
    if (ralm->layers) {
        for (int i = 0; i < ralm->n_layers; i++) {
            transformer_block_free_sub(&ralm->layers[i]);
        }
        free(ralm->layers);
    }
    if (ralm->output_norm) {
        tensor_free(ralm->output_norm->weight);
        free(ralm->output_norm);
    }
    tensor_free(ralm->cache_k);
    tensor_free(ralm->cache_v);
    free(ralm);
}

/* ═══════════════════════════════════════════════════════════════
 * LocDiT — Create from weights
 * ═══════════════════════════════════════════════════════════════ */

LocDiT* loc_dit_create(const VoxCPMConfig* config, const uint8_t* mmap_data, const WeightIndex* idx) {
    if (!config || !mmap_data || !idx) return NULL;

    LocDiT* dit = (LocDiT*)calloc(1, sizeof(LocDiT));
    if (!dit) return NULL;

    dit->d_model = config->dit_hidden;
    dit->n_layers = config->dit_n_layers;
    dit->feat_dim = config->latent_dim;

    /* Load projections */
    dit->in_proj_weight = load_weight("feat_decoder.estimator.in_proj.weight", mmap_data, idx);
    dit->in_proj_bias = load_weight("feat_decoder.estimator.in_proj.bias", mmap_data, idx);
    dit->cond_proj_weight = load_weight("feat_decoder.estimator.cond_proj.weight", mmap_data, idx);
    dit->cond_proj_bias = load_weight("feat_decoder.estimator.cond_proj.bias", mmap_data, idx);
    dit->out_proj_weight = load_weight("feat_decoder.estimator.out_proj.weight", mmap_data, idx);
    dit->out_proj_bias = load_weight("feat_decoder.estimator.out_proj.bias", mmap_data, idx);

    /* Time embeddings */
    dit->time_mlp_1_weight = load_weight("feat_decoder.estimator.time_mlp.linear_1.weight", mmap_data, idx);
    dit->time_mlp_1_bias = load_weight("feat_decoder.estimator.time_mlp.linear_1.bias", mmap_data, idx);
    dit->time_mlp_2_weight = load_weight("feat_decoder.estimator.time_mlp.linear_2.weight", mmap_data, idx);
    dit->time_mlp_2_bias = load_weight("feat_decoder.estimator.time_mlp.linear_2.bias", mmap_data, idx);
    dit->delta_mlp_1_weight = load_weight("feat_decoder.estimator.delta_time_mlp.linear_1.weight", mmap_data, idx);
    dit->delta_mlp_1_bias = load_weight("feat_decoder.estimator.delta_time_mlp.linear_1.bias", mmap_data, idx);
    dit->delta_mlp_2_weight = load_weight("feat_decoder.estimator.delta_time_mlp.linear_2.weight", mmap_data, idx);
    dit->delta_mlp_2_bias = load_weight("feat_decoder.estimator.delta_time_mlp.linear_2.bias", mmap_data, idx);

    /* Create DiT decoder layers (12) — using encoder block helper since same hidden/ffn dimensions */
    dit->layers = (TransformerBlock*)calloc((size_t)dit->n_layers, sizeof(TransformerBlock));
    if (!dit->layers) { loc_dit_free(dit); return NULL; }

    for (int i = 0; i < dit->n_layers; i++) {
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "feat_decoder.estimator.decoder.layers.%d", i);
        TransformerBlock* block = enc_block_from_weights(prefix, config, mmap_data, idx);
        if (!block) { loc_dit_free(dit); return NULL; }
        dit->layers[i] = *block;
        free(block);
    }

    /* Deocoder final norm */
    Tensor* norm_w = load_weight("feat_decoder.estimator.decoder.norm.weight", mmap_data, idx);
    if (!norm_w) { loc_dit_free(dit); return NULL; }
    dit->output_norm = (RmsNorm*)calloc(1, sizeof(RmsNorm));
    dit->output_norm->weight = norm_w;
    dit->output_norm->eps = config->norm_eps;

    return dit;
}

void loc_dit_free(LocDiT* dit) {
    if (!dit) return;
    tensor_free(dit->in_proj_weight);
    tensor_free(dit->in_proj_bias);
    tensor_free(dit->cond_proj_weight);
    tensor_free(dit->cond_proj_bias);
    tensor_free(dit->out_proj_weight);
    tensor_free(dit->out_proj_bias);
    tensor_free(dit->time_mlp_1_weight);
    tensor_free(dit->time_mlp_1_bias);
    tensor_free(dit->time_mlp_2_weight);
    tensor_free(dit->time_mlp_2_bias);
    tensor_free(dit->delta_mlp_1_weight);
    tensor_free(dit->delta_mlp_1_bias);
    tensor_free(dit->delta_mlp_2_weight);
    tensor_free(dit->delta_mlp_2_bias);
    if (dit->layers) {
        for (int i = 0; i < dit->n_layers; i++) {
            transformer_block_free_sub(&dit->layers[i]);
        }
        free(dit->layers);
    }
    if (dit->output_norm) {
        tensor_free(dit->output_norm->weight);
        free(dit->output_norm);
    }
    free(dit);
}

/* ═══════════════════════════════════════════════════════════════
 * Forward stubs (will be implemented in Phase 2)
 * ═══════════════════════════════════════════════════════════════ */

void tslm_cache_clear(TSLM* tslm) {
    if (tslm) tslm->cache_len = 0;
}

VoxCPMError tslm_setup_cache(TSLM* tslm, int batch_size) {
    if (!tslm) return VOXCPM_ERR_INTERNAL;
    tensor_free(tslm->cache_k);
    tensor_free(tslm->cache_v);

    int shape[5] = {tslm->n_layers, batch_size,
                    tslm->n_kv_heads, tslm->max_seq_len,
                    tslm->head_dim};
    tslm->cache_k = tensor_create(5, shape);
    tslm->cache_v = tensor_create(5, shape);
    if (!tslm->cache_k || !tslm->cache_v) return VOXCPM_ERR_OOM;
    tslm->cache_len = 0;
    return VOXCPM_SUCCESS;
}

/* AudioVAE functions are implemented in src/audio_vae.c */

/* ═══════════════════════════════════════════════════════════════
 * Weight Index — Build from mmap'd .vxcpm data
 * ═══════════════════════════════════════════════════════════════ */

static int weight_entry_cmp(const void* a, const void* b) {
    return strcmp(((const WeightEntry*)a)->name, ((const WeightEntry*)b)->name);
}

WeightIndex* weight_index_build(const uint8_t* data, size_t data_size) {
    if (!data || data_size < VXCPM_HEADER_SIZE) return NULL;

    const VxcpmHeader* header = (const VxcpmHeader*)data;

    /* Validate magic */
    if (header->magic != VXCPM_MAGIC) {
        LOG_ERROR("Invalid .vxcpm magic: 0x%08X", header->magic);
        return NULL;
    }

    if (header->version != 1) {
        LOG_ERROR("Unsupported .vxcpm version: %u", header->version);
        return NULL;
    }

    /* Verify header checksum */
    uint8_t header_copy[VXCPM_HEADER_SIZE];
    memcpy(header_copy, data, VXCPM_HEADER_SIZE);
    /* Zero out checksum fields for verification */
    *(uint32_t*)(header_copy + 56) = 0;
    *(uint32_t*)(header_copy + 60) = 0;
    uint32_t calc_crc = crc32c_compute(header_copy, VXCPM_HEADER_SIZE, 0);
    if (calc_crc != header->header_checksum) {
        LOG_WARN("Header CRC mismatch: stored=0x%08X calc=0x%08X (non-fatal)",
                  header->header_checksum, calc_crc);
    }

    uint32_t num_tensors = header->num_tensors;
    if (num_tensors == 0) {
        LOG_ERROR("No tensors in .vxcpm file");
        return NULL;
    }

    /* Allocate index */
    WeightIndex* idx = (WeightIndex*)calloc(1, sizeof(WeightIndex));
    if (!idx) return NULL;

    idx->count = 0;
    idx->capacity = (int)num_tensors;
    idx->entries = (WeightEntry*)calloc((size_t)num_tensors, sizeof(WeightEntry));
    if (!idx->entries) {
        free(idx);
        return NULL;
    }

    /* Parse metadata entries */
    const VxcpmTensorMeta* meta_array =
        (const VxcpmTensorMeta*)(data + VXCPM_HEADER_SIZE);

    /* String table starts after metadata array */
    size_t string_table_offset = VXCPM_HEADER_SIZE +
                                  (size_t)num_tensors * VXCPM_META_ENTRY_SIZE;
    const char* string_table = (const char*)data + string_table_offset;

    /* First pass over meta_array: find where string table ends */
    size_t max_name_end = 0;
    for (uint32_t i = 0; i < num_tensors; i++) {
        const VxcpmTensorMeta* meta = &meta_array[i];
        size_t name_end = (size_t)meta->name_offset + (size_t)meta->name_length;
        if (name_end > max_name_end) max_name_end = name_end;
    }
    /* data_start must be max_name_end + 1 because max_name_end = max(name_offset + name_length)
     * points to the LAST NULL BYTE of the string table (name_length excludes null terminator).
     * Tensor data starts AFTER that null byte. */
    size_t data_start = string_table_offset + max_name_end + 1;

    /* Second pass: fill entries with correct data_offset (file order) */
    {
        uint64_t cum_offset = (uint64_t)data_start;
        for (uint32_t i = 0; i < num_tensors; i++) {
            const VxcpmTensorMeta* meta = &meta_array[i];
            const char* name_ptr = string_table + meta->name_offset;

            WeightEntry* entry = &idx->entries[idx->count];

            /* Copy name */
            size_t name_len = (size_t)meta->name_length;
            entry->name = (char*)malloc(name_len + 1);
            if (!entry->name) goto fail;
            memcpy(entry->name, name_ptr, name_len);
            entry->name[name_len] = '\0';

            entry->dtype = meta->dtype;
            entry->ndim = meta->ndim;
            memcpy(entry->shape, meta->shape, sizeof(uint32_t) * 4);

            /* Compute element count and raw size */
            uint64_t elem_count = 1;
            for (int d = 0; d < (int)meta->ndim; d++) {
                elem_count *= (uint64_t)meta->shape[d];
            }
            int elem_size = 4; /* default FP32 */
            switch (meta->dtype) {
                case VXCPM_DTYPE_BF16:
                case VXCPM_DTYPE_FP16:  elem_size = 2; break;
                case VXCPM_DTYPE_FP32:  elem_size = 4; break;
                case VXCPM_DTYPE_Q4_0:
                case VXCPM_DTYPE_Q4_1:  elem_size = 0; break; /* variable */
            }
            uint64_t raw_size = elem_count * (uint64_t)elem_size;

            entry->data_size = raw_size;
            entry->data_offset = cum_offset;
            cum_offset += raw_size;

            /* Debug: check for gaps (padding) */
            if (i > 0) {
                WeightEntry* prev = &idx->entries[i - 1];
                uint64_t prev_end = prev->data_offset + prev->data_size;
                if (entry->data_offset != prev_end) {
                    LOG_WARN("GAP: tensor #%d '%s' start=%llu prev '%s' end=%llu gap=%llu",
                             i, entry->name,
                             (unsigned long long)entry->data_offset,
                             prev->name,
                             (unsigned long long)prev_end,
                             (unsigned long long)(entry->data_offset - prev_end));
                }
            }

            idx->count++;
        }
    }

    /* Sort entries by name for binary search */
    qsort(idx->entries, (size_t)idx->count, sizeof(WeightEntry), weight_entry_cmp);

    return idx;

fail:
    weight_index_free(idx);
    return NULL;
}

void weight_index_free(WeightIndex* idx) {
    if (!idx) return;
    if (idx->entries) {
        for (int i = 0; i < idx->count; i++) {
            free(idx->entries[i].name);
        }
        free(idx->entries);
    }
    free(idx);
}

const WeightEntry* weight_index_find(const WeightIndex* idx, const char* name) {
    if (!idx || !name) return NULL;

    /* Binary search on sorted name array */
    int lo = 0, hi = idx->count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int cmp = strcmp(idx->entries[mid].name, name);
        if (cmp == 0) return &idx->entries[mid];
        if (cmp < 0) lo = mid + 1;
        else         hi = mid - 1;
    }
    return NULL;
}

VoxCPMError weight_load_tensor(
    const uint8_t* data,
    const WeightEntry* entry,
    Tensor** out_tensor)
{
    if (!data || !entry || !out_tensor) return VOXCPM_ERR_INTERNAL;

    /* Build tensor shape from entry */
    int shape[4];
    int ndim = (int)entry->ndim;
    if (ndim < 1 || ndim > 4) ndim = 1;
    for (int i = 0; i < ndim; i++) {
        shape[i] = (int)entry->shape[i];
    }

    /* Determine element count and size */
    int64_t num_elems = 1;
    for (int i = 0; i < ndim; i++) {
        num_elems *= shape[i];
    }

    /* Create tensor (always stored as float32 internally) */
    Tensor* t = tensor_create(ndim, shape);
    if (!t) return VOXCPM_ERR_OOM;

    const uint8_t* src = data + entry->data_offset;

    switch (entry->dtype) {
        case VXCPM_DTYPE_FP32: {
            /* Direct copy */
            memcpy(t->data, src, (size_t)num_elems * sizeof(float));
            break;
        }
        case VXCPM_DTYPE_FP16: {
            /* Keep as FP16 — no conversion to FP32. Store raw fp16 data in data_fp16.
             * Exception: 1D tensors (biases, norm weights) are tiny and are accessed
             * directly as float* in many places (add_bias_2d, etc.), so convert them
             * to fp32 immediately. */
            if (ndim == 1) {
                /* 1D tensor — convert to FP32 directly */
                const uint16_t* f16 = (const uint16_t*)src;
                float* dst = t->data;
                for (int64_t i = 0; i < num_elems; i++) {
                    dst[i] = fp16_to_fp32(f16[i]);
                }
            } else {
                /* 2D+ tensor — keep as FP16 for memory efficiency */
                size_t bytes = (size_t)num_elems * sizeof(uint16_t);
                free(t->data);  /* free the unused FP32 buffer */
                t->data = NULL;
                t->data_fp16 = (uint16_t*)malloc(bytes);
                if (!t->data_fp16) { tensor_free(t); return VOXCPM_ERR_OOM; }
                memcpy(t->data_fp16, src, bytes);
                t->is_fp16 = true;
            }
            break;
        }
        case VXCPM_DTYPE_BF16: {
            /* Convert bfloat16 -> float32 */
            const uint16_t* bf16 = (const uint16_t*)src;
            float* dst = t->data;
            for (int64_t i = 0; i < num_elems; i++) {
                uint32_t f = (uint32_t)bf16[i] << 16;
                memcpy(&dst[i], &f, sizeof(float));
            }
            break;
        }
        default: {
            tensor_free(t);
            return VOXCPM_ERR_UNSUPPORTED;
        }
    }

    /* Clamp NaN to zero — safety net for edge cases in BF16 conversion.
     * Skip for fp16 tensors: they have no fp32 data yet. */
    if (!t->is_fp16) {
        int nan_count = 0;
        for (int64_t i = 0; i < num_elems; i++) {
            if (isnan(t->data[i])) { t->data[i] = 0.0f; nan_count++; }
        }
        if (nan_count > 0) {
            LOG_WARN("NaN clamped: '%s' %d NaN (size=%lld)", entry->name, nan_count, (long long)num_elems);
        }
    }

    *out_tensor = t;
    return VOXCPM_SUCCESS;
}



// audio.c — WAV I/O and audio processing
// VoxCPM2-C Project
// License: Apache-2.0
//
// Implements WAV file reading/writing (16/24/32-bit PCM),
// resampling, normalization, noise gate, pre/de-emphasis.
//
// WAV Format Support:
//   - Standard PCM WAV (16-bit, 24-bit, 32-bit integer)
//   - RIFF chunk parsing (fmt, data, skip unknown chunks)
//   - Mono/stereo (convert stereo to mono by averaging)
//   - Always writes 16-bit PCM mono

#include "audio.h"
#include "platform.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 * Internal: WAV chunk reading helpers
 * ═══════════════════════════════════════════════════════════════ */

/* Portable file seek relative to current position.
 * WAV chunks may be up to ~4 GB; use 64-bit seek where available. */
static int file_skip(FILE* f, uint32_t bytes) {
#ifdef _MSC_VER
    return _fseeki64(f, (__int64)bytes, SEEK_CUR) == 0;
#elif defined(_WIN32)
    return _fseeki64(f, (__int64)bytes, SEEK_CUR) == 0;
#else
    return fseeko(f, (off_t)bytes, SEEK_CUR) == 0;
#endif
}

/* Read a 16-bit little-endian value from a buffer. */
static uint16_t read_le16(const uint8_t* buf) {
    return (uint16_t)((uint16_t)buf[0] | ((uint16_t)buf[1] << 8));
}

/* Read a 32-bit little-endian value from a buffer. */
static uint32_t read_le32(const uint8_t* buf) {
    return ((uint32_t)buf[0]) | ((uint32_t)buf[1] << 8)
         | ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24);
}

/* Convert a signed integer sample to float [-1.0, 1.0].
 * bits: 16, 24, or 32. For 24-bit, the sample is stored in 3 bytes
 * (little-endian) and is sign-extended to int32. */
static float sample_to_float(const uint8_t* buf, int bits) {
    if (bits == 16) {
        int16_t s = (int16_t)read_le16(buf);
        return (float)s * (1.0f / 32768.0f);
    } else if (bits == 24) {
        /* 24-bit sample in 3 bytes, little-endian */
        int32_t s = (int32_t)(((uint32_t)buf[0])
                            | ((uint32_t)buf[1] << 8)
                            | ((uint32_t)buf[2] << 16));
        /* Sign-extend from 24 bits */
        if (s & 0x800000) s |= ~0xFFFFFF;
        return (float)s * (1.0f / 8388608.0f);
    } else if (bits == 32) {
        int32_t s = (int32_t)read_le32(buf);
        return (float)s * (1.0f / 2147483648.0f);
    }
    return 0.0f;
}

/* ═══════════════════════════════════════════════════════════════
 * audio_read_wav — Read a WAV file into a float array
 *
 * Supported formats: PCM (format=1), 16/24/32-bit, mono/stereo.
 * Stereo is downmixed to mono by averaging channels.
 *
 * Returns malloc'd float array (caller must free), or NULL on error.
 * Sets *out_num_samples and *out_sample_rate on success.
 * ═══════════════════════════════════════════════════════════════ */
float* audio_read_wav(const char* path, int* out_num_samples, int* out_sample_rate) {
    if (!path) {
        LOG_ERROR("audio_read_wav: NULL path");
        if (out_num_samples) *out_num_samples = 0;
        if (out_sample_rate) *out_sample_rate = 0;
        return NULL;
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        LOG_ERROR("audio_read_wav: cannot open file: %s", path);
        if (out_num_samples) *out_num_samples = 0;
        if (out_sample_rate) *out_sample_rate = 0;
        return NULL;
    }

    /* Defaults */
    int sample_rate = 0;
    int bits_per_sample = 0;
    int num_channels = 0;
    uint32_t data_size = 0;
    int format_tag = 0;
    int got_fmt = 0;
    int got_data = 0;

    /* ── RIFF header ── */
    uint8_t riff_hdr[12];
    if (fread(riff_hdr, 1, 12, f) != 12) {
        LOG_ERROR("audio_read_wav: truncated RIFF header");
        fclose(f);
        if (out_num_samples) *out_num_samples = 0;
        if (out_sample_rate) *out_sample_rate = 0;
        return NULL;
    }

    if (memcmp(riff_hdr, "RIFF", 4) != 0) {
        LOG_ERROR("audio_read_wav: not a RIFF file");
        fclose(f);
        if (out_num_samples) *out_num_samples = 0;
        if (out_sample_rate) *out_sample_rate = 0;
        return NULL;
    }
    if (memcmp(riff_hdr + 8, "WAVE", 4) != 0) {
        LOG_ERROR("audio_read_wav: not a WAVE file");
        fclose(f);
        if (out_num_samples) *out_num_samples = 0;
        if (out_sample_rate) *out_sample_rate = 0;
        return NULL;
    }

    /* ── Parse RIFF chunks ── */
    while (1) {
        uint8_t chunk_hdr[8];
        size_t nread = fread(chunk_hdr, 1, 8, f);
        if (nread != 8) {
            /* Normal end of file or truncated; stop parsing */
            break;
        }

        uint32_t chunk_size = read_le32(chunk_hdr + 4);
        char chunk_id[5];
        memcpy(chunk_id, chunk_hdr, 4);
        chunk_id[4] = '\0';

        if (memcmp(chunk_hdr, "fmt ", 4) == 0) {
            /* fmt subchunk — at least 16 bytes */
            uint32_t read_size = chunk_size < 16 ? chunk_size : 16;
            uint8_t fmt_buf[16];
            if (fread(fmt_buf, 1, read_size, f) != read_size) {
                LOG_ERROR("audio_read_wav: truncated fmt chunk");
                break;
            }
            /* Skip remaining bytes of fmt chunk if larger */
            if (chunk_size > read_size) {
                file_skip(f, chunk_size - read_size);
            }

            format_tag     = (int)read_le16(fmt_buf + 0);
            num_channels   = (int)read_le16(fmt_buf + 2);
            sample_rate    = (int)read_le32(fmt_buf + 4);
            /* byte_rate at +8, block_align at +12 (unused here) */
            bits_per_sample = (int)read_le16(fmt_buf + 14);

            got_fmt = 1;

            /* Validate */
            if (format_tag != 1) {
                LOG_ERROR("audio_read_wav: unsupported format tag %d (only PCM=1)", format_tag);
                fclose(f);
                if (out_num_samples) *out_num_samples = 0;
                if (out_sample_rate) *out_sample_rate = 0;
                return NULL;
            }
            if (num_channels < 1 || num_channels > 2) {
                LOG_ERROR("audio_read_wav: unsupported channel count %d (only 1 or 2)", num_channels);
                fclose(f);
                if (out_num_samples) *out_num_samples = 0;
                if (out_sample_rate) *out_sample_rate = 0;
                return NULL;
            }
            if (bits_per_sample != 16 && bits_per_sample != 24 && bits_per_sample != 32) {
                LOG_ERROR("audio_read_wav: unsupported bits per sample %d", bits_per_sample);
                fclose(f);
                if (out_num_samples) *out_num_samples = 0;
                if (out_sample_rate) *out_sample_rate = 0;
                return NULL;
            }

        } else if (memcmp(chunk_hdr, "data", 4) == 0) {
            /* data subchunk */
            data_size = chunk_size;
            got_data = 1;
            break;  /* Stop parsing; data follows */

        } else {
            /* Skip unknown chunk */
            if (chunk_size > 0) {
                file_skip(f, chunk_size);
            }
        }
    }

    if (!got_fmt || !got_data) {
        LOG_ERROR("audio_read_wav: missing fmt or data chunk");
        fclose(f);
        if (out_num_samples) *out_num_samples = 0;
        if (out_sample_rate) *out_sample_rate = 0;
        return NULL;
    }

    /* ── Read sample data ── */
    int bytes_per_sample = bits_per_sample / 8;
    int total_frame_bytes = (int)data_size;

    /* Total frames (samples per channel) */
    int num_frames = total_frame_bytes / (bytes_per_sample * num_channels);
    if (num_frames <= 0) {
        LOG_ERROR("audio_read_wav: no audio data");
        fclose(f);
        if (out_num_samples) *out_num_samples = 0;
        if (out_sample_rate) *out_sample_rate = 0;
        return NULL;
    }

    /* Allocate output buffer (mono) */
    float* samples = (float*)malloc((size_t)num_frames * sizeof(float));
    if (!samples) {
        LOG_ERROR("audio_read_wav: OOM for %d samples", num_frames);
        fclose(f);
        if (out_num_samples) *out_num_samples = 0;
        if (out_sample_rate) *out_sample_rate = 0;
        return NULL;
    }

    /* Read frame by frame (handles arbitrary chunk sizes) */
    uint8_t* frame_buf = (uint8_t*)malloc((size_t)(bytes_per_sample * num_channels));
    if (!frame_buf) {
        LOG_ERROR("audio_read_wav: OOM for frame buffer");
        free(samples);
        fclose(f);
        if (out_num_samples) *out_num_samples = 0;
        if (out_sample_rate) *out_sample_rate = 0;
        return NULL;
    }

    for (int i = 0; i < num_frames; i++) {
        size_t n = fread(frame_buf, 1, (size_t)(bytes_per_sample * num_channels), f);
        if (n != (size_t)(bytes_per_sample * num_channels)) {
            /* Truncated read; fill remaining with silence */
            for (int j = i; j < num_frames; j++) {
                samples[j] = 0.0f;
            }
            break;
        }

        if (num_channels == 1) {
            samples[i] = sample_to_float(frame_buf, bits_per_sample);
        } else {
            /* Stereo: average channels */
            float left  = sample_to_float(frame_buf, bits_per_sample);
            float right = sample_to_float(frame_buf + bytes_per_sample, bits_per_sample);
            samples[i] = (left + right) * 0.5f;
        }
    }

    free(frame_buf);
    fclose(f);

    if (out_num_samples) *out_num_samples = num_frames;
    if (out_sample_rate) *out_sample_rate = sample_rate;

    return samples;
}

/* ═══════════════════════════════════════════════════════════════
 * audio_write_wav — Write float array to 16-bit PCM mono WAV
 *
 * Returns 0 on success, -1 on error.
 * ═══════════════════════════════════════════════════════════════ */
int audio_write_wav(const float* samples, int num_samples, int sample_rate,
                    const char* path) {
    if (!path || !samples) {
        LOG_ERROR("audio_write_wav: NULL parameter");
        return -1;
    }
    if (num_samples <= 0 || sample_rate <= 0) {
        LOG_ERROR("audio_write_wav: invalid num_samples=%d or sample_rate=%d",
                  num_samples, sample_rate);
        return -1;
    }

    FILE* f = fopen(path, "wb");
    if (!f) {
        LOG_ERROR("audio_write_wav: cannot create file: %s", path);
        return -1;
    }

    int bits_per_sample = 16;
    int channels = 1;
    int bytes_per_sample = bits_per_sample / 8;
    int data_size = num_samples * bytes_per_sample;
    int fmt_size = 16;

    /* RIFF header */
    if (fwrite("RIFF", 1, 4, f) != 4) goto write_error;
    uint32_t chunk_size = (uint32_t)(36 + data_size);
    if (fwrite(&chunk_size, 4, 1, f) != 1) goto write_error;
    if (fwrite("WAVE", 1, 4, f) != 4) goto write_error;

    /* fmt subchunk */
    if (fwrite("fmt ", 1, 4, f) != 4) goto write_error;
    if (fwrite(&fmt_size, 4, 1, f) != 1) goto write_error;
    uint16_t audio_format = 1; /* PCM */
    if (fwrite(&audio_format, 2, 1, f) != 1) goto write_error;
    uint16_t num_channels_u16 = (uint16_t)channels;
    if (fwrite(&num_channels_u16, 2, 1, f) != 1) goto write_error;
    if (fwrite(&sample_rate, 4, 1, f) != 1) goto write_error;
    uint32_t byte_rate = (uint32_t)(sample_rate * channels * bytes_per_sample);
    if (fwrite(&byte_rate, 4, 1, f) != 1) goto write_error;
    uint16_t block_align = (uint16_t)(channels * bytes_per_sample);
    if (fwrite(&block_align, 2, 1, f) != 1) goto write_error;
    uint16_t bits_u16 = (uint16_t)bits_per_sample;
    if (fwrite(&bits_u16, 2, 1, f) != 1) goto write_error;

    /* data subchunk */
    if (fwrite("data", 1, 4, f) != 4) goto write_error;
    if (fwrite(&data_size, 4, 1, f) != 1) goto write_error;

    /* Convert float [-1,1] to int16 and write */
    for (int i = 0; i < num_samples; i++) {
        float s = samples[i];
        if (s < -1.0f) s = -1.0f;
        if (s >  1.0f) s =  1.0f;
        int16_t sample_i16 = (int16_t)(s * 32767.0f);
        if (fwrite(&sample_i16, 2, 1, f) != 1) goto write_error;
    }

    fclose(f);
    return 0;

write_error:
    LOG_ERROR("audio_write_wav: write error on file: %s", path);
    fclose(f);
    return -1;
}

/* ═══════════════════════════════════════════════════════════════
 * Internal: linear interpolation helpers for audio_resample
 * ═══════════════════════════════════════════════════════════════ */

/* Get a pointer to the first sample given a tensor.
 * Handles both [batch, samples] and [samples] layouts. */
static float* tensor_audio_data(const Tensor* t, int* out_samples) {
    if (!t || !t->data) return NULL;

    if (t->ndim == 1) {
        *out_samples = t->shape[0];
        return t->data;
    } else if (t->ndim == 2) {
        /* [batch, samples] — use the first batch only */
        *out_samples = t->shape[1];
        return t->data; /* row-major: batch 0 starts at data[0] */
    }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════
 * audio_resample — Simple linear interpolation resampling
 *
 * Input tensor shape: [batch, samples] or [samples].
 * Output tensor: allocated with tensor_create, shape matches input
 * (batch dimension preserved).
 * Returns NULL on error.
 * ═══════════════════════════════════════════════════════════════ */
Tensor* audio_resample(const Tensor* input, int input_rate, int output_rate) {
    if (!input || !input->data) {
        LOG_ERROR("audio_resample: NULL input");
        return NULL;
    }
    if (input_rate <= 0 || output_rate <= 0) {
        LOG_ERROR("audio_resample: invalid rates (%d -> %d)", input_rate, output_rate);
        return NULL;
    }
    if (input_rate == output_rate) {
        /* No resampling needed — return a clone */
        return tensor_clone(input);
    }

    int input_samples = 0;
    float* in_data = tensor_audio_data(input, &input_samples);
    if (!in_data || input_samples <= 0) {
        LOG_ERROR("audio_resample: invalid input tensor shape");
        return NULL;
    }

    int batch = (input->ndim == 2) ? input->shape[0] : 1;

    /* Compute output length */
    /* output_samples = ceil(input_samples * output_rate / input_rate) */
    int64_t numerator   = (int64_t)input_samples * (int64_t)output_rate;
    int64_t denominator = (int64_t)input_rate;
    int output_samples = (int)((numerator + denominator - 1) / denominator);
    if (output_samples < 1) output_samples = 1;

    /* Create output tensor */
    Tensor* output = NULL;
    if (input->ndim == 1) {
        int s[1] = { output_samples };
        output = tensor_create(1, s);
    } else {
        int s[2] = { batch, output_samples };
        output = tensor_create(2, s);
    }
    if (!output) {
        LOG_ERROR("audio_resample: OOM for output tensor");
        return NULL;
    }

    float* out_data = output->data;

    /* Linear interpolation across all samples in the buffer.
     * For batched [batch, samples], we process each batch independently. */
    for (int b = 0; b < batch; b++) {
        float* batch_in  = in_data  + (int64_t)b * (int64_t)input_samples;
        float* batch_out = out_data + (int64_t)b * (int64_t)output_samples;

        for (int i = 0; i < output_samples; i++) {
            /* Find the corresponding input position */
            double src_pos = (double)i * (double)input_rate / (double)output_rate;
            int    idx     = (int)src_pos;
            double frac    = src_pos - (double)idx;

            if (idx >= input_samples - 1) {
                /* Past the end — use the last sample */
                batch_out[i] = batch_in[input_samples - 1];
            } else {
                float s0 = batch_in[idx];
                float s1 = batch_in[idx + 1];
                batch_out[i] = (float)((double)s0 + frac * ((double)s1 - (double)s0));
            }
        }
    }

    /* Propagate name */
    if (input->name[0] != '\0') {
        tensor_set_name(output, input->name);
    }

    return output;
}

/* ═══════════════════════════════════════════════════════════════
 * audio_normalize — Normalize audio to [-1.0, 1.0] (in-place)
 *
 * Finds the maximum absolute value and scales all samples
 * so that the max absolute value = 1.0.
 * If audio is silent (all zeros), does nothing.
 * ═══════════════════════════════════════════════════════════════ */
void audio_normalize(Tensor* audio) {
    if (!audio || !audio->data || audio->size == 0) return;

    /* Find max absolute value */
    float max_abs = 0.0f;
    for (size_t i = 0; i < audio->size; i++) {
        float v = fabsf(audio->data[i]);
        if (v > max_abs) max_abs = v;
    }

    /* If silent, do nothing */
    if (max_abs <= 0.0f) return;

    /* Scale */
    float scale = 1.0f / max_abs;
    for (size_t i = 0; i < audio->size; i++) {
        audio->data[i] *= scale;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * audio_noise_gate — Silence samples below threshold (in-place)
 *
 * Sets samples with |sample| < threshold to 0.
 * ═══════════════════════════════════════════════════════════════ */
void audio_noise_gate(Tensor* audio, float threshold) {
    if (!audio || !audio->data || audio->size == 0) return;

    /* Clamp negative threshold to positive */
    float t = (threshold < 0.0f) ? -threshold : threshold;

    for (size_t i = 0; i < audio->size; i++) {
        if (fabsf(audio->data[i]) < t) {
            audio->data[i] = 0.0f;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * audio_pre_emphasis — Apply pre-emphasis filter (in-place)
 *
 * y[t] = x[t] - coeff * x[t-1], with y[0] = x[0]
 * ═══════════════════════════════════════════════════════════════ */
void audio_pre_emphasis(Tensor* audio, float coeff) {
    if (!audio || !audio->data || audio->size == 0) return;

    float prev = audio->data[0];
    /* y[0] = x[0] — stays the same */

    for (size_t i = 1; i < audio->size; i++) {
        float curr = audio->data[i];
        audio->data[i] = audio->data[i] - coeff * prev;
        prev = curr;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * audio_de_emphasis — Apply de-emphasis filter (in-place)
 *
 * y[t] = x[t] + coeff * y[t-1], with y[0] = x[0]
 * ═══════════════════════════════════════════════════════════════ */
void audio_de_emphasis(Tensor* audio, float coeff) {
    if (!audio || !audio->data || audio->size == 0) return;

    /* y[0] = x[0] */
    float prev_y = audio->data[0];

    for (size_t i = 1; i < audio->size; i++) {
        float y = audio->data[i] + coeff * prev_y;
        audio->data[i] = y;
        prev_y = y;
    }
}

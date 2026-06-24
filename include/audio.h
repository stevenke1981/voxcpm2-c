#ifndef AUDIO_H
#define AUDIO_H

/*
 * audio.h — Audio I/O and processing
 * VoxCPM2-C Project
 * License: Apache-2.0
 *
 * WAV file reading/writing, audio preprocessing, and post-processing.
 */

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 * WAV I/O
 * ═══════════════════════════════════════════════════════════════ */

/* Read a WAV file into a float array.
 * Returns number of samples read, or -1 on error.
 * Caller must free() the returned samples pointer. */
float* audio_read_wav(const char* path, int* out_num_samples, int* out_sample_rate);

/* Write a float array as a WAV file.
 * samples should be in [-1.0, 1.0] range.
 * Returns 0 on success, -1 on error. */
int audio_write_wav(const float* samples, int num_samples, int sample_rate,
                    const char* path);

/* ═══════════════════════════════════════════════════════════════
 * Audio Processing
 * ═══════════════════════════════════════════════════════════════ */

/* Resample audio from one sample rate to another (simple linear interpolation). */
Tensor* audio_resample(const Tensor* input, int input_rate, int output_rate);

/* Normalize audio to [-1.0, 1.0] range (in-place). */
void audio_normalize(Tensor* audio);

/* Apply simple noise gate (in-place).
 * Silences samples below threshold. */
void audio_noise_gate(Tensor* audio, float threshold);

/* Apply pre-emphasis filter: y[t] = x[t] - coeff * x[t-1] */
void audio_pre_emphasis(Tensor* audio, float coeff);

/* Apply de-emphasis filter: y[t] = x[t] + coeff * y[t-1] */
void audio_de_emphasis(Tensor* audio, float coeff);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_H */

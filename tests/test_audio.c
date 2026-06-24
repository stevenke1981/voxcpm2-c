// test_audio.c — Audio I/O and processing unit tests
// VoxCPM2-C Project
// License: Apache-2.0
//
// Tests for WAV read/write, resampling, normalization,
// noise gate, pre-emphasis, and de-emphasis.

#include "audio.h"
#include "platform.h"
#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) do { printf("  TEST: %s ... ", name); } while(0)
#define PASS() do { printf("PASS\n"); g_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); g_failed++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* ═══════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════ */
#define TEST_WAV "test_audio_temp.wav"
#define SR 44100
#define TOLERANCE 1e-4f

/* ═══════════════════════════════════════════════════════════════
 * Helper: compare float arrays with tolerance
 * ═══════════════════════════════════════════════════════════════ */
static int float_arrays_equal(const float* a, const float* b, int n, float tol) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tol) return 0;
    }
    return 1;
}

/* ═══════════════════════════════════════════════════════════════
 * Tests
 * ═══════════════════════════════════════════════════════════════ */

static void test_write_read_roundtrip(void) {
    TEST("write/read roundtrip 16-bit PCM");

    int num_samples = 1000;
    float* original = (float*)malloc((size_t)num_samples * sizeof(float));
    ASSERT(original != NULL, "OOM");

    /* Generate a sine wave */
    for (int i = 0; i < num_samples; i++) {
        original[i] = sinf((float)i * 2.0f * 3.14159265f * 440.0f / (float)SR) * 0.5f;
    }

    /* Write */
    int ret = audio_write_wav(original, num_samples, SR, TEST_WAV);
    ASSERT(ret == 0, "write_wav returned error");

    /* Read back */
    int read_samples = 0, read_sr = 0;
    float* read_data = audio_read_wav(TEST_WAV, &read_samples, &read_sr);
    ASSERT(read_data != NULL, "read_wav returned NULL");
    ASSERT(read_samples == num_samples, "sample count mismatch");
    ASSERT(read_sr == SR, "sample rate mismatch");

    /* Compare (16-bit roundtrip has ~3e-5 quantization error) */
    ASSERT(float_arrays_equal(original, read_data, num_samples, 5e-4f),
           "sample values differ beyond tolerance");

    free(original);
    free(read_data);
    remove(TEST_WAV);
    PASS();
}

static void test_write_read_null_path(void) {
    TEST("write_wav with NULL path");
    float dummy[10] = {0};
    ASSERT(audio_write_wav(dummy, 10, 44100, NULL) == -1,
           "should return -1 for NULL path");
    PASS();
}

static void test_write_read_invalid_params(void) {
    TEST("write_wav with invalid params");
    float dummy[10] = {0};
    ASSERT(audio_write_wav(NULL, 10, 44100, TEST_WAV) == -1,
           "should return -1 for NULL samples");
    ASSERT(audio_write_wav(dummy, 0, 44100, TEST_WAV) == -1,
           "should return -1 for zero samples");
    ASSERT(audio_write_wav(dummy, 10, 0, TEST_WAV) == -1,
           "should return -1 for zero sample rate");
    PASS();
}

static void test_read_wav_nonexistent(void) {
    TEST("read_wav with nonexistent file");
    int ns = 0, sr = 0;
    float* data = audio_read_wav("nonexistent_file.wav", &ns, &sr);
    ASSERT(data == NULL, "should return NULL");
    ASSERT(ns == 0, "out_num_samples should be 0");
    ASSERT(sr == 0, "out_sample_rate should be 0");
    PASS();
}

static void test_read_wav_null_path(void) {
    TEST("read_wav with NULL path");
    int ns = 0, sr = 0;
    float* data = audio_read_wav(NULL, &ns, &sr);
    ASSERT(data == NULL, "should return NULL");
    ASSERT(ns == 0, "out_num_samples should be 0");
    ASSERT(sr == 0, "out_sample_rate should be 0");
    PASS();
}

static void test_write_read_silence(void) {
    TEST("write/read silence");

    int num_samples = 500;
    float* silence = (float*)calloc((size_t)num_samples, sizeof(float));
    ASSERT(silence != NULL, "OOM");

    int ret = audio_write_wav(silence, num_samples, 22050, TEST_WAV);
    ASSERT(ret == 0, "write_wav failed");

    int read_samples = 0, read_sr = 0;
    float* read_data = audio_read_wav(TEST_WAV, &read_samples, &read_sr);
    ASSERT(read_data != NULL, "read_wav returned NULL");
    ASSERT(read_samples == num_samples, "sample count mismatch");
    ASSERT(read_sr == 22050, "sample rate mismatch");

    /* All samples should be near zero */
    for (int i = 0; i < read_samples; i++) {
        ASSERT(fabsf(read_data[i]) < 1e-6f, "silence sample not zero");
    }

    free(silence);
    free(read_data);
    remove(TEST_WAV);
    PASS();
}

static void test_normalize(void) {
    TEST("normalize");

    int shape[] = { 6 };
    Tensor* t = tensor_create(1, shape);
    ASSERT(t != NULL, "tensor_create failed");

    /* Values: [0.0, 0.5, -0.3, 0.0, 1.0, -2.0] — max abs = 2.0 */
    t->data[0] = 0.0f;
    t->data[1] = 0.5f;
    t->data[2] = -0.3f;
    t->data[3] = 0.0f;
    t->data[4] = 1.0f;
    t->data[5] = -2.0f;

    audio_normalize(t);

    /* After normalization: scale = 1/2 = 0.5 */
    float expected[] = {0.0f, 0.25f, -0.15f, 0.0f, 0.5f, -1.0f};
    ASSERT(float_arrays_equal(t->data, expected, 6, TOLERANCE),
           "normalize values mismatch");

    tensor_free(t);
    PASS();
}

static void test_normalize_silent(void) {
    TEST("normalize silent tensor (no-op)");

    int shape[] = { 4 };
    Tensor* t = tensor_create(1, shape);
    ASSERT(t != NULL, "tensor_create failed");

    /* All zeros — should stay zeros */
    t->data[0] = 0.0f; t->data[1] = 0.0f; t->data[2] = 0.0f; t->data[3] = 0.0f;
    audio_normalize(t);
    ASSERT(float_arrays_equal(t->data, (float[4]){0,0,0,0}, 4, TOLERANCE),
           "silent normalize should not change values");

    tensor_free(t);
    PASS();
}

static void test_normalize_null(void) {
    TEST("normalize NULL tensor (no crash)");
    audio_normalize(NULL);
    PASS();
}

static void test_noise_gate(void) {
    TEST("noise gate");

    int shape[] = { 6 };
    Tensor* t = tensor_create(1, shape);
    ASSERT(t != NULL, "tensor_create failed");

    t->data[0] =  0.5f;
    t->data[1] = -0.1f;
    t->data[2] =  0.05f;
    t->data[3] = -0.5f;
    t->data[4] =  0.0f;
    t->data[5] =  0.3f;

    /* Threshold = 0.2, samples with |v| < 0.2 get zeroed */
    audio_noise_gate(t, 0.2f);

    float expected[] = {0.5f, 0.0f, 0.0f, -0.5f, 0.0f, 0.3f};
    ASSERT(float_arrays_equal(t->data, expected, 6, TOLERANCE),
           "noise gate values mismatch");

    tensor_free(t);
    PASS();
}

static void test_noise_gate_null(void) {
    TEST("noise gate NULL tensor (no crash)");
    audio_noise_gate(NULL, 0.5f);
    PASS();
}

static void test_pre_emphasis(void) {
    TEST("pre-emphasis");

    int shape[] = { 5 };
    Tensor* t = tensor_create(1, shape);
    ASSERT(t != NULL, "tensor_create failed");

    /* x = [1, 2, 3, 4, 5], coeff = 0.5 */
    t->data[0] = 1.0f;
    t->data[1] = 2.0f;
    t->data[2] = 3.0f;
    t->data[3] = 4.0f;
    t->data[4] = 5.0f;

    audio_pre_emphasis(t, 0.5f);

    /* y[0] = 1
     * y[1] = 2 - 0.5*1 = 1.5
     * y[2] = 3 - 0.5*2 = 2.0
     * y[3] = 4 - 0.5*3 = 2.5
     * y[4] = 5 - 0.5*4 = 3.0 */
    float expected[] = {1.0f, 1.5f, 2.0f, 2.5f, 3.0f};
    ASSERT(float_arrays_equal(t->data, expected, 5, TOLERANCE),
           "pre-emphasis values mismatch");

    tensor_free(t);
    PASS();
}

static void test_pre_emphasis_null(void) {
    TEST("pre-emphasis NULL tensor (no crash)");
    audio_pre_emphasis(NULL, 0.9f);
    PASS();
}

static void test_de_emphasis(void) {
    TEST("de-emphasis");

    int shape[] = { 5 };
    Tensor* t = tensor_create(1, shape);
    ASSERT(t != NULL, "tensor_create failed");

    /* x = [1, 2, 3, 4, 5], coeff = 0.5 */
    t->data[0] = 1.0f;
    t->data[1] = 2.0f;
    t->data[2] = 3.0f;
    t->data[3] = 4.0f;
    t->data[4] = 5.0f;

    audio_de_emphasis(t, 0.5f);

    /* y[0] = 1
     * y[1] = 2 + 0.5*1 = 2.5
     * y[2] = 3 + 0.5*2.5 = 4.25
     * y[3] = 4 + 0.5*4.25 = 6.125
     * y[4] = 5 + 0.5*6.125 = 8.0625 */
    float expected[] = {1.0f, 2.5f, 4.25f, 6.125f, 8.0625f};
    ASSERT(float_arrays_equal(t->data, expected, 5, TOLERANCE),
           "de-emphasis values mismatch");

    tensor_free(t);
    PASS();
}

static void test_de_emphasis_null(void) {
    TEST("de-emphasis NULL tensor (no crash)");
    audio_de_emphasis(NULL, 0.9f);
    PASS();
}

static void test_pre_de_emphasis_roundtrip(void) {
    TEST("pre-emphasis + de-emphasis roundtrip");

    int shape[] = { 8 };
    Tensor* t = tensor_create(1, shape);
    ASSERT(t != NULL, "create failed");

    float orig[] = {0.1f, 0.3f, -0.2f, 0.5f, -0.1f, 0.0f, 0.8f, -0.4f};
    memcpy(t->data, orig, 8 * sizeof(float));

    Tensor* copy = tensor_clone(t);
    ASSERT(copy != NULL, "clone failed");

    float coeff = 0.95f;
    audio_pre_emphasis(copy, coeff);
    audio_de_emphasis(copy, coeff);

    /* After pre then de, values should be close to original
     * (subject to floating-point drift at the first sample). */
    ASSERT(float_arrays_equal(t->data, copy->data, 8, 1e-4f),
           "pre+de roundtrip differs");

    tensor_free(t);
    tensor_free(copy);
    PASS();
}

static void test_resample_same_rate(void) {
    TEST("resample same rate");

    int shape[] = { 10 };
    Tensor* input = tensor_create(1, shape);
    ASSERT(input != NULL, "create failed");

    for (int i = 0; i < 10; i++) input->data[i] = (float)(i + 1);

    Tensor* output = audio_resample(input, 44100, 44100);
    ASSERT(output != NULL, "resample returned NULL");
    ASSERT(output->size == input->size, "size should match for same rate");
    ASSERT(float_arrays_equal(input->data, output->data, 10, TOLERANCE),
           "values should match for same rate");

    tensor_free(input);
    tensor_free(output);
    PASS();
}

static void test_resample_downsample(void) {
    TEST("resample downsample 2x");

    int shape[] = { 6 };
    Tensor* input = tensor_create(1, shape);
    ASSERT(input != NULL, "create failed");

    /* Linear signal: [0, 1, 2, 3, 4, 5] at 44100 Hz -> 22050 Hz
     * Output length: ceil(6 * 22050/44100) = 3
     * Positions: i*44100/22050 = 0, 2, 4
     * Output: [0, 2, 4] */
    for (int i = 0; i < 6; i++) input->data[i] = (float)i;

    Tensor* output = audio_resample(input, 44100, 22050);
    ASSERT(output != NULL, "resample returned NULL");
    ASSERT(output->shape[0] == 3, "expected 3 output samples");

    float expected[] = {0.0f, 2.0f, 4.0f};
    ASSERT(float_arrays_equal(output->data, expected, 3, TOLERANCE),
           "downsample values mismatch");

    tensor_free(input);
    tensor_free(output);
    PASS();
}

static void test_resample_upsample(void) {
    TEST("resample upsample 2x");

    int shape[] = { 4 };
    Tensor* input = tensor_create(1, shape);
    ASSERT(input != NULL, "create failed");

    /* [0, 2, 4, 6] at 22050 Hz -> 44100 Hz
     * Output length: ceil(4 * 44100/22050) = 8
     * Positions: i*22050/44100 = 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5
     * Output: [0, 1, 2, 3, 4, 5, 6, 6] */
    input->data[0] = 0.0f;
    input->data[1] = 2.0f;
    input->data[2] = 4.0f;
    input->data[3] = 6.0f;

    Tensor* output = audio_resample(input, 22050, 44100);
    ASSERT(output != NULL, "resample returned NULL");
    ASSERT(output->shape[0] == 8, "expected 8 output samples");

    float expected[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f};
    ASSERT(float_arrays_equal(output->data, expected, 8, TOLERANCE),
           "upsample values mismatch");

    tensor_free(input);
    tensor_free(output);
    PASS();
}

static void test_resample_invalid_params(void) {
    TEST("resample invalid params");

    int shape[] = { 5 };
    Tensor* t = tensor_create(1, shape);
    ASSERT(t != NULL, "create failed");

    ASSERT(audio_resample(NULL, 44100, 22050) == NULL,
           "should return NULL for NULL input");
    ASSERT(audio_resample(t, 0, 22050) == NULL,
           "should return NULL for zero input_rate");
    ASSERT(audio_resample(t, 44100, 0) == NULL,
           "should return NULL for zero output_rate");

    tensor_free(t);
    PASS();
}

static void test_resample_batched(void) {
    TEST("resample batched [batch, samples]");

    int shape[] = { 2, 4 };
    Tensor* input = tensor_create(2, shape);
    ASSERT(input != NULL, "create failed");

    /* Batch 0: [0, 1, 2, 3], Batch 1: [10, 11, 12, 13] */
    for (int i = 0; i < 8; i++) input->data[i] = (float)(i < 4 ? i : i + 6);

    Tensor* output = audio_resample(input, 44100, 22050);
    ASSERT(output != NULL, "resample returned NULL");
    ASSERT(output->ndim == 2, "output should be 2D");
    ASSERT(output->shape[0] == 2, "batch dim should be 2");
    ASSERT(output->shape[1] == 2, "expected 2 output samples per batch");

    /* Batch 0: [0, 2], Batch 1: [10, 12] */
    ASSERT(fabsf(output->data[0] - 0.0f)  < TOLERANCE, "batch0[0] mismatch");
    ASSERT(fabsf(output->data[1] - 2.0f)  < TOLERANCE, "batch0[1] mismatch");
    ASSERT(fabsf(output->data[2] - 10.0f) < TOLERANCE, "batch1[0] mismatch");
    ASSERT(fabsf(output->data[3] - 12.0f) < TOLERANCE, "batch1[1] mismatch");

    tensor_free(input);
    tensor_free(output);
    PASS();
}

static void test_write_clamp(void) {
    TEST("write_wav clamps out-of-range samples");

    /* Samples outside [-1, 1] should be clamped */
    float samples[] = { -2.0f, 0.5f, 2.0f, -0.5f, 0.0f };
    int ret = audio_write_wav(samples, 5, 44100, TEST_WAV);
    ASSERT(ret == 0, "write_wav failed");

    int ns = 0, sr = 0;
    float* read_data = audio_read_wav(TEST_WAV, &ns, &sr);
    ASSERT(read_data != NULL, "read_wav returned NULL");
    ASSERT(ns == 5, "sample count mismatch");

    /* Clamped to [-1, 1] before 16-bit quantization */
    ASSERT(fabsf(read_data[0] - (-1.0f)) < 5e-4f, "sample 0 should be -1.0");
    ASSERT(fabsf(read_data[1] - 0.5f)     < 5e-4f, "sample 1 should be 0.5");
    ASSERT(fabsf(read_data[2] - 1.0f)     < 5e-4f, "sample 2 should be 1.0");

    free(read_data);
    remove(TEST_WAV);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("=== VoxCPM2-C Audio Unit Tests ===\n\n");

    /* WAV I/O */
    test_write_read_roundtrip();
    test_write_read_silence();
    test_write_read_null_path();
    test_write_read_invalid_params();
    test_read_wav_nonexistent();
    test_read_wav_null_path();
    test_write_clamp();

    /* Normalize */
    test_normalize();
    test_normalize_silent();
    test_normalize_null();

    /* Noise gate */
    test_noise_gate();
    test_noise_gate_null();

    /* Pre/de-emphasis */
    test_pre_emphasis();
    test_pre_emphasis_null();
    test_de_emphasis();
    test_de_emphasis_null();
    test_pre_de_emphasis_roundtrip();

    /* Resample */
    test_resample_same_rate();
    test_resample_downsample();
    test_resample_upsample();
    test_resample_invalid_params();
    test_resample_batched();

    printf("\n=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}

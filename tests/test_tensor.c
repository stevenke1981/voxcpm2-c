// test_tensor.c — Tensor unit tests (Phase 0)
// VoxCPM2-C Project
// License: Apache-2.0
//
// Tests for core tensor operations.
// Compile with: gcc -I include -o test_tensor tests/test_tensor.c src/tensor.c -lm

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
 * P0-03: Lifecycle tests
 * ═══════════════════════════════════════════════════════════════ */

static void test_create_free(void) {
    TEST("create/free 2D tensor");
    int shape[] = {3, 4};
    Tensor* t = tensor_create(2, shape);
    ASSERT(t != NULL, "tensor_create returned NULL");
    ASSERT(t->ndim == 2, "ndim != 2");
    ASSERT(t->shape[0] == 3, "shape[0] != 3");
    ASSERT(t->shape[1] == 4, "shape[1] != 4");
    ASSERT(t->size == 12, "size != 12");
    ASSERT(t->data != NULL, "data is NULL");
    ASSERT(t->is_owned == true, "is_owned != true");
    ASSERT(t->is_cuda == false, "is_cuda != false");
    tensor_free(t);
    PASS();
}

static void test_create_from_buffer(void) {
    TEST("create from buffer");
    float buf[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int shape[] = {2, 2};
    Tensor* t = tensor_create_from_buffer(2, shape, buf);
    ASSERT(t != NULL, "returned NULL");
    ASSERT(t->data == buf, "data pointer mismatch");
    ASSERT(t->is_owned == false, "is_owned should be false");
    ASSERT(t->data[0] == 1.0f, "data[0] mismatch");
    tensor_free(t); // Should not free buf
    ASSERT(buf[0] == 1.0f, "buffer was freed");
    PASS();
}

static void test_scalar(void) {
    TEST("scalar tensor");
    Tensor* s = tensor_scalar(3.14f);
    ASSERT(s != NULL, "returned NULL");
    ASSERT(s->ndim == 1, "ndim != 1");
    ASSERT(s->size == 1, "size != 1");
    ASSERT(fabsf(s->data[0] - 3.14f) < 1e-6f, "value mismatch");
    tensor_free(s);
    PASS();
}

static void test_clone(void) {
    TEST("clone tensor");
    int shape[] = {2, 3};
    Tensor* a = tensor_create(2, shape);
    ASSERT(a != NULL, "create returned NULL");
    for (int i = 0; i < 6; i++) a->data[i] = (float)(i + 1);

    Tensor* b = tensor_clone(a);
    ASSERT(b != NULL, "clone returned NULL");
    ASSERT(tensor_shape_eq(a, b), "shape mismatch");
    for (int i = 0; i < 6; i++) {
        ASSERT(b->data[i] == (float)(i + 1), "data mismatch");
    }

    // Verify independence
    a->data[0] = 99.0f;
    ASSERT(b->data[0] == 1.0f, "not independent");

    tensor_free(a);
    tensor_free(b);
    PASS();
}

static void test_copy(void) {
    TEST("copy tensor");
    int shape_a[] = {2, 3};
    int shape_b[] = {2, 3};
    Tensor* a = tensor_create(2, shape_a);
    Tensor* b = tensor_create(2, shape_b);
    ASSERT(a && b, "create failed");
    for (int i = 0; i < 6; i++) a->data[i] = (float)i;

    ASSERT(tensor_copy(b, a) == VOXCPM_SUCCESS, "copy failed");
    for (int i = 0; i < 6; i++) {
        ASSERT(b->data[i] == (float)i, "data mismatch after copy");
    }

    tensor_free(a);
    tensor_free(b);
    PASS();
}

static void test_reshape(void) {
    TEST("reshape tensor");
    int shape[] = {2, 6};
    Tensor* t = tensor_create(2, shape);
    ASSERT(t != NULL, "create failed");
    for (int i = 0; i < 12; i++) t->data[i] = (float)i;

    int new_shape[] = {3, 4};
    ASSERT(tensor_reshape(t, 2, new_shape) == VOXCPM_SUCCESS, "reshape failed");
    ASSERT(t->shape[0] == 3, "shape[0] != 3");
    ASSERT(t->shape[1] == 4, "shape[1] != 4");
    ASSERT(t->size == 12, "size changed");

    tensor_free(t);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════
 * P0-04: Matmul tests
 * ═══════════════════════════════════════════════════════════════ */

static void test_matmul(void) {
    TEST("matrix multiply 2x3 @ 3x2");
    int shape_a[] = {2, 3};
    int shape_b[] = {3, 2};
    int shape_c[] = {2, 2};
    Tensor* a = tensor_create(2, shape_a);
    Tensor* b = tensor_create(2, shape_b);
    Tensor* c = tensor_create(2, shape_c);
    ASSERT(a && b && c, "create failed");

    // A = [[1, 2, 3], [4, 5, 6]]
    a->data[0] = 1; a->data[1] = 2; a->data[2] = 3;
    a->data[3] = 4; a->data[4] = 5; a->data[5] = 6;

    // B = [[7, 8], [9, 10], [11, 12]]
    b->data[0] = 7;  b->data[1] = 8;
    b->data[2] = 9;  b->data[3] = 10;
    b->data[4] = 11; b->data[5] = 12;

    ASSERT(tensor_matmul(a, b, c) == VOXCPM_SUCCESS, "matmul failed");

    // C = [[58, 64], [139, 154]]
    ASSERT(fabsf(c->data[0] - 58.0f) < 1e-5f, "C[0] != 58");
    ASSERT(fabsf(c->data[1] - 64.0f) < 1e-5f, "C[1] != 64");
    ASSERT(fabsf(c->data[2] - 139.0f) < 1e-5f, "C[2] != 139");
    ASSERT(fabsf(c->data[3] - 154.0f) < 1e-5f, "C[3] != 154");

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    PASS();
}

static void test_matmul_shape_mismatch(void) {
    TEST("matmul shape mismatch detection");
    int shape_a[] = {2, 3};
    int shape_b[] = {4, 2}; // K mismatch: 3 vs 4
    int shape_c[] = {2, 2};
    Tensor* a = tensor_create(2, shape_a);
    Tensor* b = tensor_create(2, shape_b);
    Tensor* c = tensor_create(2, shape_c);
    ASSERT(a && b && c, "create failed");

    ASSERT(tensor_matmul(a, b, c) != VOXCPM_SUCCESS,
           "should return error on shape mismatch");

    tensor_free(a); tensor_free(b); tensor_free(c);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════
 * P0-05: Element-wise tests
 * ═══════════════════════════════════════════════════════════════ */

static void test_element_wise(void) {
    TEST("element-wise add");
    int shape[] = {3};
    Tensor* a = tensor_create(1, shape);
    Tensor* b = tensor_create(1, shape);
    Tensor* c = tensor_create(1, shape);
    ASSERT(a && b && c, "create failed");

    a->data[0] = 1; a->data[1] = 2; a->data[2] = 3;
    b->data[0] = 4; b->data[1] = 5; b->data[2] = 6;

    ASSERT(tensor_add(a, b, c) == VOXCPM_SUCCESS, "add failed");
    ASSERT(fabsf(c->data[0] - 5.0f) < 1e-5f, "c[0] != 5");
    ASSERT(fabsf(c->data[1] - 7.0f) < 1e-5f, "c[1] != 7");
    ASSERT(fabsf(c->data[2] - 9.0f) < 1e-5f, "c[2] != 9");

    tensor_free(a); tensor_free(b); tensor_free(c);
    PASS();
}

static void test_scale(void) {
    TEST("scale tensor");
    int shape[] = {4};
    Tensor* t = tensor_create(1, shape);
    ASSERT(t, "create failed");
    t->data[0] = 1; t->data[1] = 2; t->data[2] = 3; t->data[3] = 4;

    ASSERT(tensor_scale(t, 2.0f) == VOXCPM_SUCCESS, "scale failed");
    ASSERT(fabsf(t->data[0] - 2.0f) < 1e-5f, "t[0] != 2");
    ASSERT(fabsf(t->data[3] - 8.0f) < 1e-5f, "t[3] != 8");

    tensor_free(t);
    PASS();
}

static void test_silu(void) {
    TEST("silu activation");
    int shape[] = {3};
    Tensor* t = tensor_create(1, shape);
    ASSERT(t, "create failed");
    t->data[0] = 0.0f;
    t->data[1] = 1.0f;
    t->data[2] = -1.0f;

    ASSERT(tensor_silu(t) == VOXCPM_SUCCESS, "silu failed");
    // silu(0) = 0
    ASSERT(fabsf(t->data[0]) < 1e-5f, "silu(0) != 0");
    // silu(1) should be > 0
    ASSERT(t->data[1] > 0.5f, "silu(1) < 0.5");
    // silu(-1) should be negative but > -1
    ASSERT(t->data[2] < 0.0f, "silu(-1) > 0");
    ASSERT(t->data[2] > -1.0f, "silu(-1) < -1");

    tensor_free(t);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════
 * P0-06: Normalization tests
 * ═══════════════════════════════════════════════════════════════ */

static void test_softmax(void) {
    TEST("softmax");
    int shape[] = {1, 4};
    Tensor* t = tensor_create(2, shape);
    ASSERT(t, "create failed");
    t->data[0] = 1.0f; t->data[1] = 2.0f;
    t->data[2] = 3.0f; t->data[3] = 4.0f;

    ASSERT(tensor_softmax(t, 1) == VOXCPM_SUCCESS, "softmax failed");

    // Sum should be 1
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) sum += t->data[i];
    ASSERT(fabsf(sum - 1.0f) < 1e-5f, "sum != 1");

    // Values should be in (0, 1)
    for (int i = 0; i < 4; i++) {
        ASSERT(t->data[i] > 0.0f, "value <= 0");
        ASSERT(t->data[i] < 1.0f, "value >= 1");
    }

    tensor_free(t);
    PASS();
}

static void test_rms_norm(void) {
    TEST("rms_norm");
    int shape_x[] = {1, 4};
    int shape_w[] = {4};
    Tensor* x = tensor_create(2, shape_x);
    Tensor* w = tensor_create(1, shape_w);
    Tensor* out = tensor_create(2, shape_x);
    ASSERT(x && w && out, "create failed");

    x->data[0] = 1; x->data[1] = 2; x->data[2] = 3; x->data[3] = 4;
    for (int i = 0; i < 4; i++) w->data[i] = 1.0f; // Unit weight

    ASSERT(tensor_rms_norm(x, w, 1e-5f, out) == VOXCPM_SUCCESS, "rms_norm failed");

    // RMS: sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    // out = x / rms = [0.365, 0.730, 1.095, 1.461]
    float rms = sqrtf(7.5f);
    for (int i = 0; i < 4; i++) {
        float expected = (float)(i + 1) / rms;
        ASSERT(fabsf(out->data[i] - expected) < 1e-4f, "rms_norm value mismatch");
    }

    tensor_free(x); tensor_free(w); tensor_free(out);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════
 * P0-07: Rotary embedding tests
 * ═══════════════════════════════════════════════════════════════ */

static void test_rotary_precompute(void) {
    TEST("precompute_freqs_cis");
    int max_seq = 10;
    int dim = 4;
    float theta = 10000.0f;
    int shape[] = {max_seq, dim/2, 2};
    Tensor* freqs = tensor_create(3, shape);
    ASSERT(freqs, "create failed");

    ASSERT(tensor_precompute_freqs_cis(max_seq, dim, theta, freqs) == VOXCPM_SUCCESS,
           "precompute failed");

    // At position 0, cos should be 1, sin should be 0
    ASSERT(fabsf(freqs->data[0] - 1.0f) < 1e-5f, "pos0 cos != 1");
    ASSERT(fabsf(freqs->data[1]) < 1e-5f, "pos0 sin != 0");

    tensor_free(freqs);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════
 * Debug/utility tests
 * ═══════════════════════════════════════════════════════════════ */

static void test_fill_zero(void) {
    TEST("fill and zero");
    int shape[] = {2, 5};
    Tensor* t = tensor_create(2, shape);
    ASSERT(t, "create failed");

    tensor_fill(t, 3.14f);
    for (size_t i = 0; i < t->size; i++) {
        ASSERT(fabsf(t->data[i] - 3.14f) < 1e-5f, "fill value mismatch");
    }

    tensor_zero(t);
    for (size_t i = 0; i < t->size; i++) {
        ASSERT(fabsf(t->data[i]) < 1e-5f, "zero value mismatch");
    }

    tensor_free(t);
    PASS();
}

static void test_minmax(void) {
    TEST("minmax");
    int shape[] = {5};
    Tensor* t = tensor_create(1, shape);
    ASSERT(t, "create failed");
    t->data[0] = 3; t->data[1] = -1; t->data[2] = 7; t->data[3] = 0; t->data[4] = -5;

    float min_val, max_val;
    int min_idx, max_idx;
    tensor_minmax(t, &min_val, &max_val, &min_idx, &max_idx);

    ASSERT(fabsf(min_val + 5.0f) < 1e-5f, "min != -5");
    ASSERT(fabsf(max_val - 7.0f) < 1e-5f, "max != 7");
    ASSERT(min_idx == 4, "min_idx != 4");
    ASSERT(max_idx == 2, "max_idx != 2");

    tensor_free(t);
    PASS();
}

static void test_check_nan(void) {
    TEST("check nan/inf");
    int shape[] = {4};
    Tensor* t = tensor_create(1, shape);
    ASSERT(t, "create failed");
    t->data[0] = 1; t->data[1] = 2; t->data[2] = 3; t->data[3] = 4;

    int bad_idx = -1;
    ASSERT(tensor_check_nan_inf(t, &bad_idx) == false, "false positive");

    t->data[2] = NAN;
    ASSERT(tensor_check_nan_inf(t, &bad_idx) == true, "false negative for NaN");
    ASSERT(bad_idx == 2, "bad_idx should be 2");

    tensor_free(t);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════
 * Shape utility tests
 * ═══════════════════════════════════════════════════════════════ */

static void test_shape_equality(void) {
    TEST("shape_eq");
    int shape1[] = {2, 3, 4};
    int shape2[] = {2, 3, 4};
    int shape3[] = {2, 3, 5};
    Tensor* a = tensor_create(3, shape1);
    Tensor* b = tensor_create(3, shape2);
    Tensor* c = tensor_create(3, shape3);

    ASSERT(tensor_shape_eq(a, b), "identical shapes should be equal");
    ASSERT(!tensor_shape_eq(a, c), "different shapes should not be equal");

    tensor_free(a); tensor_free(b); tensor_free(c);
    PASS();
}

static void test_get_set(void) {
    TEST("get/set accessors");
    int shape[] = {2, 3};
    Tensor* t = tensor_create(2, shape);
    ASSERT(t, "create failed");

    tensor_set(t, 0, 1, 0, 0, 42.0f);
    float v = tensor_get(t, 0, 1, 0, 0);
    ASSERT(fabsf(v - 42.0f) < 1e-5f, "get/set mismatch");

    tensor_set_flat(t, 0, 10.0f);
    ASSERT(fabsf(tensor_get_flat(t, 0) - 10.0f) < 1e-5f, "flat get/set mismatch");

    tensor_free(t);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("=== VoxCPM2-C Tensor Unit Tests ===\n\n");

    // P0-03: Lifecycle
    test_create_free();
    test_create_from_buffer();
    test_scalar();
    test_clone();
    test_copy();
    test_reshape();

    // P0-04: Matmul
    test_matmul();
    test_matmul_shape_mismatch();

    // P0-05: Element-wise
    test_element_wise();
    test_scale();
    test_silu();

    // P0-06: Normalization
    test_softmax();
    test_rms_norm();

    // P0-07: Rotary
    test_rotary_precompute();

    // Utilities
    test_fill_zero();
    test_minmax();
    test_check_nan();
    test_shape_equality();
    test_get_set();

    printf("\n=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}

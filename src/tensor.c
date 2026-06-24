// tensor.c — Tensor operations implementation (CPU reference)
// VoxCPM2-C Project
// License: Apache-2.0
//
// Naive CPU implementation with correctness-first design.
// Optimization phases (later): restrict, loop interchange, OpenMP, SIMD.

#include "tensor.h"

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * Internal Helpers
 * ═══════════════════════════════════════════════════════════════ */

// Compute strides from shape (row-major, in elements)
static void compute_strides(Tensor* t) {
    int64_t stride = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        t->strides[i] = stride;
        stride *= t->shape[i];
    }
    t->size = (size_t)stride;
}

// Validate shape dimensions are positive
static bool validate_shape(int ndim, const int* shape) {
    if (ndim < 0 || ndim > TENSOR_MAX_DIMS) return false;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) return false;
    }
    return true;
}

// Compute total size from shape
static size_t size_from_shape(int ndim, const int* shape) {
    if (ndim <= 0) return 0;
    size_t s = 1;
    for (int i = 0; i < ndim; i++) {
        s *= (size_t)shape[i];
    }
    return s;
}

/* ═══════════════════════════════════════════════════════════════
 * P0-03: Lifecycle
 * ═══════════════════════════════════════════════════════════════ */

Tensor* tensor_create(int ndim, const int* shape) {
    if (!validate_shape(ndim, shape)) return NULL;

    Tensor* t = (Tensor*)calloc(1, sizeof(Tensor));
    if (!t) return NULL;

    t->shape = (int*)calloc((size_t)ndim, sizeof(int));
    if (!t->shape) {
        free(t);
        return NULL;
    }

    memcpy(t->shape, shape, (size_t)ndim * sizeof(int));
    t->ndim = ndim;
    compute_strides(t);

    size_t alloc_bytes = (size_t)t->size * sizeof(float);
    t->data = (float*)calloc(t->size, sizeof(float));
    if (!t->data && t->size > 0) {
        fprintf(stderr, "[FATAL] tensor_create OOM: ndim=%d size=%zu bytes=%zu shape=[%d %d %d %d %d]\n",
                ndim, t->size, alloc_bytes,
                ndim>0?shape[0]:0, ndim>1?shape[1]:0,
                ndim>2?shape[2]:0, ndim>3?shape[3]:0,
                ndim>4?shape[4]:0);
        free(t->shape);
        free(t);
        return NULL;
    }

    t->is_cuda = false;
    t->is_owned = true;
    t->is_fp16 = false;
    t->data_fp16 = NULL;
    t->name[0] = '\0';

    return t;
}

Tensor* tensor_create_from_buffer(int ndim, const int* shape, float* data) {
    if (!validate_shape(ndim, shape) || !data) return NULL;

    Tensor* t = (Tensor*)calloc(1, sizeof(Tensor));
    if (!t) return NULL;

    t->shape = (int*)calloc((size_t)ndim, sizeof(int));
    if (!t->shape) {
        free(t);
        return NULL;
    }

    memcpy(t->shape, shape, (size_t)ndim * sizeof(int));
    t->ndim = ndim;
    compute_strides(t);
    t->data = data;
    t->is_cuda = false;
    t->is_owned = false;
    t->is_fp16 = false;
    t->data_fp16 = NULL;
    t->name[0] = '\0';

    return t;
}

Tensor* tensor_scalar(float value) {
    const int shape[1] = {1};
    Tensor* t = tensor_create(1, shape);
    if (t) {
        t->data[0] = value;
    }
    return t;
}

void tensor_free(Tensor* t) {
    if (!t) return;
    if (t->is_owned && t->data) {
        free(t->data);
    }
    if (t->is_owned && t->data_fp16) {
        free(t->data_fp16);
    }
    free(t->shape);
    free(t);
}

/* Convert fp16 tensor to fp32 in-place */
VoxCPMError tensor_ensure_fp32(Tensor* t) {
    if (!t) return VOXCPM_ERR_INTERNAL;
    if (!t->is_fp16 || !t->data_fp16) return VOXCPM_SUCCESS;
    if (t->is_cuda) return VOXCPM_ERR_INTERNAL; /* cannot convert on GPU */

    size_t n = t->size;
    float* fp32_data = (float*)malloc(n * sizeof(float));
    if (!fp32_data) return VOXCPM_ERR_OOM;

    for (size_t i = 0; i < n; i++) {
        fp32_data[i] = fp16_to_fp32(t->data_fp16[i]);
    }

    if (t->is_owned) {
        free(t->data_fp16);
    }
    t->data_fp16 = NULL;
    t->data = fp32_data;
    t->is_fp16 = false;
    t->is_owned = true;
    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_copy(Tensor* dst, const Tensor* src) {
    if (!dst || !src) return VOXCPM_ERR_INTERNAL;
    if (!tensor_shape_eq(dst, src)) {
        // Reallocate dst to match src shape
        int* new_shape = (int*)calloc((size_t)src->ndim, sizeof(int));
        if (!new_shape) return VOXCPM_ERR_OOM;
        memcpy(new_shape, src->shape, (size_t)src->ndim * sizeof(int));

        // Free old data if owned
        if (dst->is_owned && dst->data) {
            free(dst->data);
        }
        free(dst->shape);

        dst->shape = new_shape;
        dst->ndim = src->ndim;
        compute_strides(dst);

        dst->data = (float*)calloc(dst->size, sizeof(float));
        if (!dst->data && dst->size > 0) return VOXCPM_ERR_OOM;
        dst->is_owned = true;
    }

    memcpy(dst->data, src->data, src->size * sizeof(float));
    return VOXCPM_SUCCESS;
}

Tensor* tensor_clone(const Tensor* src) {
    if (!src) return NULL;
    Tensor* t = tensor_create(src->ndim, src->shape);
    if (!t) return NULL;
    memcpy(t->data, src->data, src->size * sizeof(float));
    return t;
}

/* ═══════════════════════════════════════════════════════════════
 * P0-03: Shape / Metadata
 * ═══════════════════════════════════════════════════════════════ */

VoxCPMError tensor_reshape(Tensor* t, int ndim, const int* shape) {
    if (!t || !validate_shape(ndim, shape)) return VOXCPM_ERR_INTERNAL;

    size_t new_size = size_from_shape(ndim, shape);
    if (new_size != t->size) return VOXCPM_ERR_SHAPE_MISMATCH;

    int* new_shape = (int*)realloc(t->shape, (size_t)ndim * sizeof(int));
    if (!new_shape) return VOXCPM_ERR_OOM;

    t->shape = new_shape;
    memcpy(t->shape, shape, (size_t)ndim * sizeof(int));
    t->ndim = ndim;
    compute_strides(t);

    return VOXCPM_SUCCESS;
}

size_t tensor_size(const Tensor* t) {
    return t ? t->size : 0;
}

int tensor_dim_size(const Tensor* t, int dim) {
    if (!t || dim < 0 || dim >= t->ndim) return 0;
    return t->shape[dim];
}

bool tensor_shape_eq(const Tensor* a, const Tensor* b) {
    if (!a || !b) return false;
    if (a->ndim != b->ndim) return false;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    return true;
}

void tensor_print_shape(const Tensor* t) {
    if (!t) {
        fprintf(stderr, "[Tensor] NULL\n");
        return;
    }
    fprintf(stderr, "[Tensor] \"%s\" ndim=%d size=%zu shape=[",
            t->name, t->ndim, t->size);
    for (int i = 0; i < t->ndim; i++) {
        fprintf(stderr, "%d", t->shape[i]);
        if (i < t->ndim - 1) fprintf(stderr, ",");
    }
    fprintf(stderr, "] %s\n", t->is_cuda ? "CUDA" : "CPU");
}

void tensor_print_stats(const Tensor* t) {
    if (!t || !t->data || t->size == 0) {
        fprintf(stderr, "[Tensor] empty\n");
        return;
    }
    float min = t->data[0], max = t->data[0];
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < t->size; i++) {
        float v = t->data[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += (double)v;
        sum_sq += (double)v * (double)v;
    }
    double mean = sum / (double)t->size;
    double variance = sum_sq / (double)t->size - mean * mean;
    fprintf(stderr, "[Tensor] \"%s\" min=%.6f max=%.6f mean=%.6f std=%.6f\n",
            t->name, (double)min, (double)max, mean, sqrt(variance));
}

/* ═══════════════════════════════════════════════════════════════
 * P0-03: Accessors
 * ═══════════════════════════════════════════════════════════════ */

float* tensor_get_ptr(const Tensor* t, size_t flat_idx) {
    if (!t || !t->data || flat_idx >= t->size) return NULL;
    return &t->data[flat_idx];
}

float tensor_get_flat(const Tensor* t, size_t flat_idx) {
    if (!t || !t->data || flat_idx >= t->size) return 0.0f;
    return t->data[flat_idx];
}

void tensor_set_flat(Tensor* t, size_t flat_idx, float val) {
    if (!t || !t->data || flat_idx >= t->size) return;
    t->data[flat_idx] = val;
}

size_t tensor_flat_index(const Tensor* t, int i, int j, int k, int l) {
    if (!t) return 0;
    size_t idx = 0;
    if (t->ndim >= 1) idx += (size_t)i * (size_t)t->strides[0];
    if (t->ndim >= 2) idx += (size_t)j * (size_t)t->strides[1];
    if (t->ndim >= 3) idx += (size_t)k * (size_t)t->strides[2];
    if (t->ndim >= 4) idx += (size_t)l * (size_t)t->strides[3];
    return idx;
}

float tensor_get(const Tensor* t, int i, int j, int k, int l) {
    size_t idx = tensor_flat_index(t, i, j, k, l);
    return tensor_get_flat(t, idx);
}

void tensor_set(Tensor* t, int i, int j, int k, int l, float val) {
    size_t idx = tensor_flat_index(t, i, j, k, l);
    tensor_set_flat(t, idx, val);
}

/* ═══════════════════════════════════════════════════════════════
 * P0-03: Fill / Initialize
 * ═══════════════════════════════════════════════════════════════ */

void tensor_fill(Tensor* t, float val) {
    if (!t || !t->data) return;
    if (t->is_fp16) return; /* Cannot fill fp16 tensor with float value */
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = val;
    }
}

void tensor_zero(Tensor* t) {
    tensor_fill(t, 0.0f);
}

// Simple xorshift64 PRNG
static inline uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

void tensor_rand_uniform(Tensor* t, uint64_t* seed) {
    if (!t || !t->data || !seed) return;
    for (size_t i = 0; i < t->size; i++) {
        uint64_t r = xorshift64(seed);
        // Convert to float in [0, 1)
        t->data[i] = (float)(r >> 11) * 0x1.0p-53f;
    }
}

void tensor_rand_normal(Tensor* t, uint64_t* seed) {
    if (!t || !t->data || !seed) return;
    // Box-Muller transform
    for (size_t i = 0; i + 1 < t->size; i += 2) {
        double u1 = (double)xorshift64(seed) / (double)UINT64_MAX;
        double u2 = (double)xorshift64(seed) / (double)UINT64_MAX;
        if (u1 < 1e-15) u1 = 1e-15;
        double r = sqrt(-2.0 * log(u1));
        double theta = 2.0 * M_PI * u2;
        t->data[i] = (float)(r * cos(theta));
        if (i + 1 < t->size)
            t->data[i + 1] = (float)(r * sin(theta));
    }
}

VoxCPMError tensor_copy_from_array(Tensor* t, const float* src, size_t count) {
    if (!t || !src) return VOXCPM_ERR_INTERNAL;
    size_t copy_size = count < t->size ? count : t->size;
    memcpy(t->data, src, copy_size * sizeof(float));
    if (count > t->size) return VOXCPM_ERR_SHAPE_MISMATCH;
    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_copy_to_array(const Tensor* t, float* dst, size_t count) {
    if (!t || !dst) return VOXCPM_ERR_INTERNAL;
    size_t copy_size = count < t->size ? count : t->size;
    memcpy(dst, t->data, copy_size * sizeof(float));
    if (count > t->size) return VOXCPM_ERR_SHAPE_MISMATCH;
    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * P0-05: Element-wise Operations
 * ═══════════════════════════════════════════════════════════════ */

#define ELEMENT_WISE_BINARY_OP(name, op)                               \
    VoxCPMError tensor_##name(const Tensor* a, const Tensor* b, Tensor* out) { \
        if (!a || !b || !out) return VOXCPM_ERR_INTERNAL;              \
        if (!tensor_shape_eq(a, b)) return VOXCPM_ERR_SHAPE_MISMATCH;  \
        if (!tensor_shape_eq(a, out) && out->size > 0)                  \
            return VOXCPM_ERR_SHAPE_MISMATCH;                          \
        for (size_t i = 0; i < a->size; i++) {                         \
            out->data[i] = a->data[i] op b->data[i];                   \
        }                                                               \
        return VOXCPM_SUCCESS;                                         \
    }

ELEMENT_WISE_BINARY_OP(add, +)
ELEMENT_WISE_BINARY_OP(sub, -)
ELEMENT_WISE_BINARY_OP(mul, *)

#undef ELEMENT_WISE_BINARY_OP

VoxCPMError tensor_scale(Tensor* t, float scalar) {
    if (!t || !t->data) return VOXCPM_ERR_INTERNAL;
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] *= scalar;
    }
    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_add_scalar(Tensor* t, float scalar) {
    if (!t || !t->data) return VOXCPM_ERR_INTERNAL;
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] += scalar;
    }
    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_neg(Tensor* t) {
    return tensor_scale(t, -1.0f);
}

VoxCPMError tensor_sigmoid(Tensor* t) {
    if (!t || !t->data) return VOXCPM_ERR_INTERNAL;
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
    }
    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_silu(Tensor* t) {
    if (!t || !t->data) return VOXCPM_ERR_INTERNAL;
    // silu(x) = x * sigmoid(x)
    for (size_t i = 0; i < t->size; i++) {
        float x = t->data[i];
        t->data[i] = x / (1.0f + expf(-x));
    }
    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_relu(Tensor* t) {
    if (!t || !t->data) return VOXCPM_ERR_INTERNAL;
    for (size_t i = 0; i < t->size; i++) {
        if (t->data[i] < 0.0f) t->data[i] = 0.0f;
    }
    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_gelu(Tensor* t) {
    if (!t || !t->data) return VOXCPM_ERR_INTERNAL;
    // GELU: 0.5 * x * (1 + tanh(sqrt(2/PI) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrtf(2.0f / M_PI)
    for (size_t i = 0; i < t->size; i++) {
        float x = t->data[i];
        float x3 = x * x * x;
        t->data[i] = 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x3)));
    }
    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_tanh(Tensor* t) {
    if (!t || !t->data) return VOXCPM_ERR_INTERNAL;
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = tanhf(t->data[i]);
    }
    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_exp(Tensor* t) {
    if (!t || !t->data) return VOXCPM_ERR_INTERNAL;
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = expf(t->data[i]);
    }
    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_sqrt(Tensor* t, float eps) {
    if (!t || !t->data) return VOXCPM_ERR_INTERNAL;
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = sqrtf(t->data[i] + eps);
    }
    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_rsqrt(Tensor* t, float eps) {
    if (!t || !t->data) return VOXCPM_ERR_INTERNAL;
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = 1.0f / sqrtf(t->data[i] + eps);
    }
    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * P0-04: Dense Linear Operations
 * ═══════════════════════════════════════════════════════════════ */

VoxCPMError tensor_matmul(const Tensor* a, const Tensor* b, Tensor* out) {
    if (!a || !b || !out) return VOXCPM_ERR_INTERNAL;
    if (a->ndim != 2 || b->ndim != 2) return VOXCPM_ERR_SHAPE_MISMATCH;

    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];

    if (b->shape[0] != K) return VOXCPM_ERR_SHAPE_MISMATCH;
    if (out->shape[0] != M || out->shape[1] != N) return VOXCPM_ERR_SHAPE_MISMATCH;

    // Handle fp16 B (weight) tensor — convert each column on-the-fly
    if (b->is_fp16) {
        float* b_col = (float*)malloc((size_t)K * sizeof(float));
        if (!b_col) return VOXCPM_ERR_OOM;

        for (int j = 0; j < N; j++) {
            // Convert B[:,j] from fp16 to fp32 (one column at a time)
            for (int k = 0; k < K; k++) {
                b_col[k] = fp16_to_fp32(b->data_fp16[(size_t)k * N + j]);
            }

            // Compute C[:,j] = A @ B[:,j]
            for (int i = 0; i < M; i++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += a->data[(size_t)i * K + k] * b_col[k];
                }
                out->data[(size_t)i * N + j] = sum;
            }
        }
        free(b_col);
        return VOXCPM_SUCCESS;
    }

    // Handle fp16 A (weight) tensor — convert each row on-the-fly
    if (a->is_fp16) {
        float* a_row = (float*)malloc((size_t)K * sizeof(float));
        if (!a_row) return VOXCPM_ERR_OOM;

        for (int i = 0; i < M; i++) {
            // Convert A[i,:] from fp16 to fp32
            for (int k = 0; k < K; k++) {
                a_row[k] = fp16_to_fp32(a->data_fp16[(size_t)i * K + k]);
            }

            // Compute C[i,:] = A_row @ B
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += a_row[k] * b->data[(size_t)k * N + j];
                }
                out->data[(size_t)i * N + j] = sum;
            }
        }
        free(a_row);
        return VOXCPM_SUCCESS;
    }

    // Naive triple-loop matmul (CPU reference implementation, both FP32)
    // OPTME: Will be optimized with loop interchange, tiling, SIMD in later phases
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a->data[(size_t)i * K + k] *
                       b->data[(size_t)k * N + j];
            }
            out->data[(size_t)i * N + j] = sum;
        }
    }

    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_batched_matmul(const Tensor* a, const Tensor* b, Tensor* out) {
    if (!a || !b || !out) return VOXCPM_ERR_INTERNAL;
    if (a->ndim != 3 || b->ndim != 3) return VOXCPM_ERR_SHAPE_MISMATCH;

    int batch = a->shape[0];
    int M = a->shape[1];
    int K = a->shape[2];
    int N = b->shape[2];

    if (b->shape[0] != batch || b->shape[1] != K) return VOXCPM_ERR_SHAPE_MISMATCH;
    if (out->shape[0] != batch || out->shape[1] != M || out->shape[2] != N)
        return VOXCPM_ERR_SHAPE_MISMATCH;

    size_t a_stride = (size_t)M * K;
    size_t b_stride = (size_t)K * N;
    size_t out_stride = (size_t)M * N;

    for (int b_idx = 0; b_idx < batch; b_idx++) {
        const float* a_batch = a->data + b_idx * a_stride;
        const float* b_batch = b->data + b_idx * b_stride;
        float* out_batch = out->data + b_idx * out_stride;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += a_batch[(size_t)i * K + k] *
                           b_batch[(size_t)k * N + j];
                }
                out_batch[(size_t)i * N + j] = sum;
            }
        }
    }

    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_matmul_nt(const Tensor* a, const Tensor* b, Tensor* out) {
    // out = A @ B^T : A[M,K], B[N,K] => out[M,N]
    if (!a || !b || !out) return VOXCPM_ERR_INTERNAL;
    if (a->ndim != 2 || b->ndim != 2) return VOXCPM_ERR_SHAPE_MISMATCH;

    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[0];

    if (b->shape[1] != K) return VOXCPM_ERR_SHAPE_MISMATCH;
    if (out->shape[0] != M || out->shape[1] != N) return VOXCPM_ERR_SHAPE_MISMATCH;

#ifdef VOXCPM_CUDA
    // Try CUDA-accelerated matmul first.
    // tensor_matmul_nt_cuda handles auto-upload of CPU inputs and auto-download
    // if output is CPU-resident. Falls back to CPU if CUDA is not initialized.
    {
        VoxCPMError cuda_err = tensor_matmul_nt_cuda(a, b, out);
        if (cuda_err != VOXCPM_ERR_CUDA_NOT_FOUND) {
            return cuda_err;  // Success or real error
        }
        // CUDA not available — fall through to CPU path
    }
#endif

    // Handle FP16 B (weight) tensor: convert each row on-the-fly
    if (b->is_fp16) {
        float* b_row = (float*)malloc((size_t)K * sizeof(float));
        if (!b_row) return VOXCPM_ERR_OOM;

        for (int j = 0; j < N; j++) {
            // Convert B[j,:] from fp16 to fp32 (one row at a time)
            for (int k = 0; k < K; k++) {
                b_row[k] = fp16_to_fp32(b->data_fp16[(size_t)j * K + k]);
            }

            // Compute C[:,j] = A @ B_row^T
            for (int i = 0; i < M; i++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += a->data[(size_t)i * K + k] * b_row[k];
                }
                out->data[(size_t)i * N + j] = sum;
            }
        }
        free(b_row);
        return VOXCPM_SUCCESS;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a->data[(size_t)i * K + k] *
                       b->data[(size_t)j * K + k];
            }
            out->data[(size_t)i * N + j] = sum;
        }
    }

    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_matmul_tn(const Tensor* a, const Tensor* b, Tensor* out) {
    // out = A^T @ B : A[K,M], B[K,N] => out[M,N]
    if (!a || !b || !out) return VOXCPM_ERR_INTERNAL;
    if (a->ndim != 2 || b->ndim != 2) return VOXCPM_ERR_SHAPE_MISMATCH;

    int K = a->shape[0];
    int M = a->shape[1];
    int N = b->shape[1];

    if (b->shape[0] != K) return VOXCPM_ERR_SHAPE_MISMATCH;
    if (out->shape[0] != M || out->shape[1] != N) return VOXCPM_ERR_SHAPE_MISMATCH;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a->data[(size_t)k * M + i] *
                       b->data[(size_t)k * N + j];
            }
            out->data[(size_t)i * N + j] = sum;
        }
    }

    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * P0-06: Normalization
 * ═══════════════════════════════════════════════════════════════ */

VoxCPMError tensor_softmax(Tensor* t, int axis) {
    if (!t || !t->data) return VOXCPM_ERR_INTERNAL;
    if (axis < 0) axis = t->ndim - 1;
    if (axis >= t->ndim) return VOXCPM_ERR_SHAPE_MISMATCH;

    int outer = 1;
    int dim_size = t->shape[axis];
    int inner = 1;

    for (int i = 0; i < axis; i++) outer *= t->shape[i];
    for (int i = axis + 1; i < t->ndim; i++) inner *= t->shape[i];

    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            // Find max for numerical stability
            float max_val = -INFINITY;
            for (int d = 0; d < dim_size; d++) {
                size_t idx = (size_t)o * dim_size * inner +
                             (size_t)d * inner + (size_t)i;
                float v = t->data[idx];
                if (v > max_val) max_val = v;
            }

            // Compute exp(x - max) and sum
            float sum = 0.0f;
            for (int d = 0; d < dim_size; d++) {
                size_t idx = (size_t)o * dim_size * inner +
                             (size_t)d * inner + (size_t)i;
                t->data[idx] = expf(t->data[idx] - max_val);
                sum += t->data[idx];
            }

            // Normalize
            float inv_sum = 1.0f / (sum + 1e-10f);
            for (int d = 0; d < dim_size; d++) {
                size_t idx = (size_t)o * dim_size * inner +
                             (size_t)d * inner + (size_t)i;
                t->data[idx] *= inv_sum;
            }
        }
    }

    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_layer_norm(
    const Tensor* x, const Tensor* weight, const Tensor* bias, float eps, Tensor* out)
{
    if (!x || !weight || !out) return VOXCPM_ERR_INTERNAL;
    // x shape: [batch, seq, d_model], weight/bias shape: [d_model]
    int ndim = x->ndim;
    if (ndim < 2) return VOXCPM_ERR_SHAPE_MISMATCH;

    int d_model = x->shape[ndim - 1];
    if (weight && weight->size != (size_t)d_model) return VOXCPM_ERR_SHAPE_MISMATCH;
    if (bias && bias->size != (size_t)d_model) return VOXCPM_ERR_SHAPE_MISMATCH;

    // Compute prefix dimensions
    int prefix = 1;
    for (int i = 0; i < ndim - 1; i++) prefix *= x->shape[i];

    for (int p = 0; p < prefix; p++) {
        // Compute mean
        double mean = 0.0;
        for (int d = 0; d < d_model; d++) {
            size_t idx = (size_t)p * d_model + (size_t)d;
            mean += (double)x->data[idx];
        }
        mean /= (double)d_model;

        // Compute variance
        double var = 0.0;
        for (int d = 0; d < d_model; d++) {
            size_t idx = (size_t)p * d_model + (size_t)d;
            double diff = (double)x->data[idx] - mean;
            var += diff * diff;
        }
        var /= (double)d_model;

        // Normalize and apply weight/bias
        double inv_std = 1.0 / sqrt(var + (double)eps);
        for (int d = 0; d < d_model; d++) {
            size_t idx = (size_t)p * d_model + (size_t)d;
            float normalized = (float)(((double)x->data[idx] - mean) * inv_std);
            if (weight) {
                normalized *= weight->is_fp16
                    ? fp16_to_fp32(weight->data_fp16[d])
                    : weight->data[d];
            }
            if (bias) {
                normalized += bias->is_fp16
                    ? fp16_to_fp32(bias->data_fp16[d])
                    : bias->data[d];
            }
            out->data[idx] = normalized;
        }
    }

    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_rms_norm(
    const Tensor* x, const Tensor* weight, float eps, Tensor* out)
{
    if (!x || !out) return VOXCPM_ERR_INTERNAL;
    int ndim = x->ndim;
    if (ndim < 2) return VOXCPM_ERR_SHAPE_MISMATCH;

    int d_model = x->shape[ndim - 1];
    if (weight && weight->size != (size_t)d_model) return VOXCPM_ERR_SHAPE_MISMATCH;

    int prefix = 1;
    for (int i = 0; i < ndim - 1; i++) prefix *= x->shape[i];

    for (int p = 0; p < prefix; p++) {
        // Compute RMS: sqrt(mean(x^2))
        double sum_sq = 0.0;
        for (int d = 0; d < d_model; d++) {
            size_t idx = (size_t)p * d_model + (size_t)d;
            float v = x->data[idx];
            if (isnan(v)) v = 0.0f; /* safety: treat NaN as zero */
            sum_sq += (double)v * (double)v;
        }
        double rms = sqrt(sum_sq / (double)d_model + (double)eps);
        double inv_rms = 1.0 / rms;

        for (int d = 0; d < d_model; d++) {
            size_t idx = (size_t)p * d_model + (size_t)d;
            float normalized = (float)((double)x->data[idx] * inv_rms);
            if (weight) {
                normalized *= weight->is_fp16
                    ? fp16_to_fp32(weight->data_fp16[d])
                    : weight->data[d];
            }
            out->data[idx] = normalized;
        }
    }

    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * P0-07: Positional Encoding (RoPE)
 * ═══════════════════════════════════════════════════════════════ */

VoxCPMError tensor_precompute_freqs_cis(
    int max_seq_len, int dim, float theta, Tensor* freqs_out)
{
    if (!freqs_out) return VOXCPM_ERR_INTERNAL;
    if (dim % 2 != 0) return VOXCPM_ERR_SHAPE_MISMATCH;

    int half_dim = dim / 2;
    if (freqs_out->shape[0] != max_seq_len ||
        freqs_out->shape[1] != half_dim ||
        freqs_out->shape[2] != 2) {
        return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = (float)pos / powf(theta, (float)(2 * i) / (float)dim);
            float cos_val = cosf(freq);
            float sin_val = sinf(freq);

            size_t idx = ((size_t)pos * half_dim + (size_t)i) * 2;
            freqs_out->data[idx]     = cos_val;
            freqs_out->data[idx + 1] = sin_val;
        }
    }

    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_rotary_emb(
    Tensor* q, Tensor* k, const Tensor* freqs_cis, int pos)
{
    if (!q || !k || !freqs_cis) return VOXCPM_ERR_INTERNAL;
    // q shape: [batch, n_heads, seq, head_dim]
    // k shape: [batch, n_kv_heads, seq, head_dim]
    if (q->ndim != 4 || k->ndim != 4) return VOXCPM_ERR_SHAPE_MISMATCH;

    int head_dim = q->shape[3];
    int half_dim = head_dim / 2;
    int seq_len = q->shape[2];
    int n_heads = q->shape[1];
    int n_kv_heads = k->shape[1];
    int batch = q->shape[0];

    // Apply rotary embeddings
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            int freqs_pos = pos + s;

            // Get cos/sin for this position
            size_t freqs_idx = ((size_t)freqs_pos * (size_t)half_dim) * 2;

            for (int h = 0; h < n_heads; h++) {
                size_t q_base = (size_t)b * n_heads * seq_len * head_dim +
                                (size_t)h * seq_len * head_dim +
                                (size_t)s * head_dim;
                for (int i = 0; i < half_dim; i++) {
                    float cos_val = freqs_cis->data[freqs_idx + (size_t)i * 2];
                    float sin_val = freqs_cis->data[freqs_idx + (size_t)i * 2 + 1];

                    float x0 = q->data[q_base + (size_t)i];
                    float x1 = q->data[q_base + (size_t)(i + half_dim)];

                    q->data[q_base + (size_t)i] =
                        x0 * cos_val - x1 * sin_val;
                    q->data[q_base + (size_t)(i + half_dim)] =
                        x0 * sin_val + x1 * cos_val;
                }
            }

            // Apply to K (if n_kv_heads matches or group-query)
            for (int h = 0; h < n_kv_heads; h++) {
                size_t k_base = (size_t)b * n_kv_heads * seq_len * head_dim +
                                (size_t)h * seq_len * head_dim +
                                (size_t)s * head_dim;
                for (int i = 0; i < half_dim; i++) {
                    float cos_val = freqs_cis->data[freqs_idx + (size_t)i * 2];
                    float sin_val = freqs_cis->data[freqs_idx + (size_t)i * 2 + 1];

                    float x0 = k->data[k_base + (size_t)i];
                    float x1 = k->data[k_base + (size_t)(i + half_dim)];

                    k->data[k_base + (size_t)i] =
                        x0 * cos_val - x1 * sin_val;
                    k->data[k_base + (size_t)(i + half_dim)] =
                        x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }

    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * View / Reshape Utilities
 * ═══════════════════════════════════════════════════════════════ */

VoxCPMError tensor_transpose(const Tensor* t, Tensor* out) {
    if (!t || !out) return VOXCPM_ERR_INTERNAL;
    if (t->ndim != 2 || out->ndim != 2) return VOXCPM_ERR_SHAPE_MISMATCH;
    if (t->shape[0] != out->shape[1] || t->shape[1] != out->shape[0])
        return VOXCPM_ERR_SHAPE_MISMATCH;

    int M = t->shape[0];
    int N = t->shape[1];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out->data[(size_t)j * M + (size_t)i] =
                t->data[(size_t)i * N + (size_t)j];
        }
    }

    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_permute(const Tensor* t, Tensor* out, const int* axes) {
    if (!t || !out || !axes) return VOXCPM_ERR_INTERNAL;
    if (t->ndim != out->ndim) return VOXCPM_ERR_SHAPE_MISMATCH;

    int ndim = t->ndim;

    // Verify axes permutation
    int seen[TENSOR_MAX_DIMS] = {0};
    for (int i = 0; i < ndim; i++) {
        if (axes[i] < 0 || axes[i] >= ndim || seen[axes[i]])
            return VOXCPM_ERR_SHAPE_MISMATCH;
        seen[axes[i]] = 1;
    }

    // Precompute strides for iteration
    int64_t in_strides[TENSOR_MAX_DIMS];
    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        in_strides[i] = stride;
        stride *= t->shape[i];
    }

    // Iterate over all elements using recursion-like nested looping
    // For simplicity, use flat iteration with index decomposition
    for (size_t flat = 0; flat < t->size; flat++) {
        // Decompose flat index into multi-dimensional indices
        int indices[TENSOR_MAX_DIMS];
        size_t remaining = flat;
        for (int i = 0; i < ndim; i++) {
            indices[i] = (int)(remaining / (size_t)in_strides[i]);
            remaining %= (size_t)in_strides[i];
        }

        // Compute output flat index
        size_t out_flat = 0;
        for (int i = 0; i < ndim; i++) {
            out_flat += (size_t)indices[axes[i]] * (size_t)out->strides[i];
        }

        out->data[out_flat] = t->data[flat];
    }

    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_cat(const Tensor* a, const Tensor* b, int axis, Tensor* out) {
    if (!a || !b || !out) return VOXCPM_ERR_INTERNAL;
    if (a->ndim != b->ndim || a->ndim != out->ndim)
        return VOXCPM_ERR_SHAPE_MISMATCH;

    // Check all non-concat dimensions match
    for (int i = 0; i < a->ndim; i++) {
        if (i == axis) continue;
        if (a->shape[i] != b->shape[i] || a->shape[i] != out->shape[i])
            return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    // Copy data: first a, then b along the concat axis
    // Fast path: axis == ndim-1 (contiguous in memory)
    if (axis == a->ndim - 1 || a->ndim == 1) {
        size_t prefix = 1;
        for (int i = 0; i < a->ndim - 1; i++) prefix *= a->shape[i];

        size_t a_stride = (size_t)a->shape[axis];
        size_t b_stride = (size_t)b->shape[axis];

        for (size_t p = 0; p < prefix; p++) {
            size_t out_base = p * (a_stride + b_stride);
            memcpy(out->data + out_base,
                   a->data + p * a_stride,
                   a_stride * sizeof(float));
            memcpy(out->data + out_base + a_stride,
                   b->data + p * b_stride,
                   b_stride * sizeof(float));
        }
    } else {
        // General case: copy element by element
        int64_t in_strides_a[TENSOR_MAX_DIMS];
        int64_t in_strides_b[TENSOR_MAX_DIMS];
        int64_t out_strides[TENSOR_MAX_DIMS];
        int64_t stride = 1;
        for (int i = a->ndim - 1; i >= 0; i--) {
            in_strides_a[i] = stride; stride *= a->shape[i];
        }
        stride = 1;
        for (int i = b->ndim - 1; i >= 0; i--) {
            in_strides_b[i] = stride; stride *= b->shape[i];
        }
        stride = 1;
        for (int i = out->ndim - 1; i >= 0; i--) {
            out_strides[i] = stride; stride *= out->shape[i];
        }

        // Iterate over all positions in output
        for (size_t flat = 0; flat < out->size; flat++) {
            int indices[TENSOR_MAX_DIMS];
            size_t remaining = flat;
            for (int i = 0; i < out->ndim; i++) {
                indices[i] = (int)(remaining / (size_t)out_strides[i]);
                remaining %= (size_t)out_strides[i];
            }

            int src_idx = indices[axis];
            const float* src_data;
            int64_t* src_strides;
            if (src_idx < a->shape[axis]) {
                src_data = a->data;
                src_strides = in_strides_a;
            } else {
                src_data = b->data;
                src_strides = in_strides_b;
                indices[axis] -= a->shape[axis];
            }

            size_t src_flat = 0;
            for (int i = 0; i < a->ndim; i++) {
                src_flat += (size_t)indices[i] * (size_t)src_strides[i];
            }
            out->data[flat] = src_data[src_flat];
        }
    }

    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_slice(const Tensor* t, int axis, int start, int length, Tensor* out) {
    if (!t || !out) return VOXCPM_ERR_INTERNAL;
    if (axis < 0 || axis >= t->ndim) return VOXCPM_ERR_SHAPE_MISMATCH;
    if (start < 0 || start + length > t->shape[axis])
        return VOXCPM_ERR_SHAPE_MISMATCH;

    // Check out shape
    for (int i = 0; i < t->ndim; i++) {
        int expected = (i == axis) ? length : t->shape[i];
        if (out->shape[i] != expected) return VOXCPM_ERR_SHAPE_MISMATCH;
    }

    // Fast path: last axis (contiguous)
    if (axis == t->ndim - 1) {
        size_t prefix = t->size / (size_t)t->shape[axis];
        size_t src_stride = (size_t)t->shape[axis];
        size_t dst_stride = (size_t)length;

        for (size_t p = 0; p < prefix; p++) {
            memcpy(out->data + p * dst_stride,
                   t->data + p * src_stride + (size_t)start,
                   dst_stride * sizeof(float));
        }
    } else {
        // General case
        int64_t t_strides[TENSOR_MAX_DIMS];
        int64_t out_strides[TENSOR_MAX_DIMS];
        int64_t stride = 1;
        for (int i = t->ndim - 1; i >= 0; i--) {
            t_strides[i] = stride; stride *= t->shape[i];
        }
        stride = 1;
        for (int i = out->ndim - 1; i >= 0; i--) {
            out_strides[i] = stride; stride *= out->shape[i];
        }

        for (size_t flat = 0; flat < out->size; flat++) {
            int indices[TENSOR_MAX_DIMS];
            size_t remaining = flat;
            for (int i = 0; i < out->ndim; i++) {
                indices[i] = (int)(remaining / (size_t)out_strides[i]);
                remaining %= (size_t)out_strides[i];
            }
            indices[axis] += start;

            size_t src_flat = 0;
            for (int i = 0; i < t->ndim; i++) {
                src_flat += (size_t)indices[i] * (size_t)t_strides[i];
            }
            out->data[flat] = t->data[src_flat];
        }
    }

    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * Debug / Utilities
 * ═══════════════════════════════════════════════════════════════ */

void tensor_set_name(Tensor* t, const char* name) {
    if (!t || !name) return;
    strncpy(t->name, name, TENSOR_MAX_NAME - 1);
    t->name[TENSOR_MAX_NAME - 1] = '\0';
}

bool tensor_check_nan_inf(const Tensor* t, int* bad_idx) {
    if (!t || !t->data) return false;
    for (size_t i = 0; i < t->size; i++) {
        float v = t->data[i];
        if (isnan(v) || isinf(v)) {
            if (bad_idx) *bad_idx = (int)i;
            return true;
        }
    }
    return false;
}

float tensor_sum(const Tensor* t) {
    if (!t || !t->data) return 0.0f;
    double s = 0.0;
    for (size_t i = 0; i < t->size; i++) {
        s += (double)t->data[i];
    }
    return (float)s;
}

float tensor_mean(const Tensor* t) {
    if (!t || t->size == 0) return 0.0f;
    return tensor_sum(t) / (float)t->size;
}

void tensor_minmax(const Tensor* t, float* min_val, float* max_val,
                    int* min_idx, int* max_idx)
{
    if (!t || !t->data || t->size == 0) {
        if (min_val) *min_val = 0.0f;
        if (max_val) *max_val = 0.0f;
        if (min_idx) *min_idx = -1;
        if (max_idx) *max_idx = -1;
        return;
    }

    float mn = t->data[0];
    float mx = t->data[0];
    int mn_i = 0, mx_i = 0;

    for (size_t i = 1; i < t->size; i++) {
        if (t->data[i] < mn) { mn = t->data[i]; mn_i = (int)i; }
        if (t->data[i] > mx) { mx = t->data[i]; mx_i = (int)i; }
    }

    if (min_val) *min_val = mn;
    if (max_val) *max_val = mx;
    if (min_idx) *min_idx = mn_i;
    if (max_idx) *max_idx = mx_i;
}

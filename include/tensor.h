#ifndef TENSOR_H
#define TENSOR_H

/*
 * tensor.h — Tensor data structure and operations
 * VoxCPM2-C Project
 * License: Apache-2.0
 *
 * Lightweight n-dimensional tensor library for CPU inference.
 * Reference implementation with correctness-first design.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "voxcpm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════ */
#define TENSOR_MAX_DIMS 5
#define TENSOR_MAX_NAME 64

/* ═══════════════════════════════════════════════════════════════
 * Backend type
 * ═══════════════════════════════════════════════════════════════ */
typedef enum {
    TENSOR_BACKEND_CPU = 0,
    TENSOR_BACKEND_CUDA = 1,
    TENSOR_BACKEND_GGML = 2,
} TensorBackend;

/* ═══════════════════════════════════════════════════════════════
 * Tensor structure
 *
 * Memory layout: row-major (C-style) contiguous.
 * shape[0] = outermost dimension (batch), shape[ndim-1] = innermost.
 * stride[i] = bytes to advance along dimension i.
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    float*  data;           /* data pointer (CPU: heap/arena, CUDA: device) */
    int*    shape;          /* shape array [ndim] (owned, heap-allocated) */
    int     ndim;           /* number of dimensions (0 = scalar) */
    int64_t strides[5];     /* precomputed strides in elements */
    size_t  size;           /* total number of elements */
    bool    is_cuda;        /* true if data lives on GPU */
    bool    is_owned;       /* true if data was allocated internally */
    uint16_t* data_fp16;    /* FP16 raw data pointer (for weights stored as FP16). Non-NULL if is_fp16=true */
    bool      is_fp16;      /* true if this tensor stores FP16 data in data_fp16 */
    char    name[TENSOR_MAX_NAME]; /* optional debug name */
} Tensor;

/* ═══════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════ */

/* Create a new tensor with given dimensions.
 * Shape array is copied internally. Data buffer is allocated.
 * Returns NULL on OOM. */
Tensor* tensor_create(int ndim, const int* shape);

/* Create a tensor that wraps an external buffer (no ownership). */
Tensor* tensor_create_from_buffer(int ndim, const int* shape, float* data);

/* Create a scalar tensor from a single float value. */
Tensor* tensor_scalar(float value);

/* Free all resources. Safe to call with NULL. */
void tensor_free(Tensor* t);

/* Convert fp16 tensor to fp32 in-place. If t is already fp32 or is_fp16=false, no-op.
 * Allocates t->data and converts all elements, then frees t->data_fp16. */
VoxCPMError tensor_ensure_fp32(Tensor* t);

/* Deep copy: dst gets its own data copy.
 * dst is re-allocated if shape mismatch. */
VoxCPMError tensor_copy(Tensor* dst, const Tensor* src);

/* Clone: create a new independent copy. */
Tensor* tensor_clone(const Tensor* src);

/* ═══════════════════════════════════════════════════════════════
 * Shape / Metadata
 * ═══════════════════════════════════════════════════════════════ */

/* Reshape (total size must match). Returns error on size mismatch. */
VoxCPMError tensor_reshape(Tensor* t, int ndim, const int* shape);

/* Get total number of elements. */
size_t tensor_size(const Tensor* t);

/* Get number of elements in a specific dimension. */
int tensor_dim_size(const Tensor* t, int dim);

/* Check shape equality between two tensors. */
bool tensor_shape_eq(const Tensor* a, const Tensor* b);

/* Print shape to stderr (debug helper). */
void tensor_print_shape(const Tensor* t);

/* Print min/max/mean/std to stderr (debug helper). */
void tensor_print_stats(const Tensor* t);

/* ═══════════════════════════════════════════════════════════════
 * Accessors
 * ═══════════════════════════════════════════════════════════════ */

/* Get pointer to element at flat index (row-major). */
float* tensor_get_ptr(const Tensor* t, size_t flat_idx);

/* Get value at flat index. */
float tensor_get_flat(const Tensor* t, size_t flat_idx);

/* Set value at flat index. */
void tensor_set_flat(Tensor* t, size_t flat_idx, float val);

/* Get value at multi-dimensional indices (variadic, up to 4 dims). */
float tensor_get(const Tensor* t, int i, int j, int k, int l);

/* Set value at multi-dimensional indices (variadic, up to 4 dims). */
void tensor_set(Tensor* t, int i, int j, int k, int l, float val);

/* Compute flat index from multi-dimensional indices. */
size_t tensor_flat_index(const Tensor* t, int i, int j, int k, int l);

/* ═══════════════════════════════════════════════════════════════
 * Fill / Initialize
 * ═══════════════════════════════════════════════════════════════ */

/* Fill all elements with a constant value. */
void tensor_fill(Tensor* t, float val);

/* Fill with zeros. */
void tensor_zero(Tensor* t);

/* Fill with random uniform [0, 1). Uses a simple xorshift generator. */
void tensor_rand_uniform(Tensor* t, uint64_t* seed);

/* Fill with random normal (Box-Muller transform). */
void tensor_rand_normal(Tensor* t, uint64_t* seed);

/* Copy data from a float array. */
VoxCPMError tensor_copy_from_array(Tensor* t, const float* src, size_t count);

/* Copy data to a float array. */
VoxCPMError tensor_copy_to_array(const Tensor* t, float* dst, size_t count);

/* ═══════════════════════════════════════════════════════════════
 * Element-wise Operations
 * ═══════════════════════════════════════════════════════════════ */

/* C = A + B  (broadcast not yet supported, shapes must match) */
VoxCPMError tensor_add(const Tensor* a, const Tensor* b, Tensor* out);

/* C = A - B */
VoxCPMError tensor_sub(const Tensor* a, const Tensor* b, Tensor* out);

/* C = A * B (element-wise multiply, NOT matmul) */
VoxCPMError tensor_mul(const Tensor* a, const Tensor* b, Tensor* out);

/* out = t * scalar */
VoxCPMError tensor_scale(Tensor* t, float scalar);

/* out = t + scalar */
VoxCPMError tensor_add_scalar(Tensor* t, float scalar);

/* out = -t (negate) */
VoxCPMError tensor_neg(Tensor* t);

/* out = 1.0 / (1.0 + exp(-x)) */
VoxCPMError tensor_sigmoid(Tensor* t);

/* out = x * sigmoid(x) */
VoxCPMError tensor_silu(Tensor* t);

/* out = max(0, x) */
VoxCPMError tensor_relu(Tensor* t);

/* out = 0.5 * x * (1.0 + tanh(sqrt(2.0/PI) * (x + 0.044715 * x^3))) */
VoxCPMError tensor_gelu(Tensor* t);

/* out = tanh(x) */
VoxCPMError tensor_tanh(Tensor* t);

/* out = exp(x) */
VoxCPMError tensor_exp(Tensor* t);

/* out = sqrt(x + eps) */
VoxCPMError tensor_sqrt(Tensor* t, float eps);

/* out = 1.0 / sqrt(x + eps) */
VoxCPMError tensor_rsqrt(Tensor* t, float eps);

/* ═══════════════════════════════════════════════════════════════
 * Dense Linear Operations
 * ═══════════════════════════════════════════════════════════════ */

/* Matrix multiply: C[M,N] = A[M,K] @ B[K,N]
 * Both A and B must be 2D. Supports in-place if C is pre-allocated. */
VoxCPMError tensor_matmul(const Tensor* a, const Tensor* b, Tensor* out);

/* Batched matrix multiply: supports leading batch dimension.
 * A shape: [batch, M, K], B shape: [batch, K, N], C shape: [batch, M, N] */
VoxCPMError tensor_batched_matmul(const Tensor* a, const Tensor* b, Tensor* out);

/* out = A @ B^T  (A[M,K], B[N,K], out[M,N]) */
VoxCPMError tensor_matmul_nt(const Tensor* a, const Tensor* b, Tensor* out);

/* out = A^T @ B  (A[K,M], B[K,N], out[M,N]) */
VoxCPMError tensor_matmul_tn(const Tensor* a, const Tensor* b, Tensor* out);

/* ═══════════════════════════════════════════════════════════════
 * Normalization
 * ═══════════════════════════════════════════════════════════════ */

/* Softmax along specified axis (typically axis=-1 for last dim).
 * Uses numerically stable implementation: max subtraction. */
VoxCPMError tensor_softmax(Tensor* t, int axis);

/* Layer Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
 * x shape: [batch, seq, d_model], weight/bias shape: [d_model] */
VoxCPMError tensor_layer_norm(
    const Tensor* x,
    const Tensor* weight,
    const Tensor* bias,
    float eps,
    Tensor* out
);

/* RMS Normalization: y = x * weight / sqrt(mean(x^2) + eps)
 * x shape: [batch, seq, d_model], weight shape: [d_model] */
VoxCPMError tensor_rms_norm(
    const Tensor* x,
    const Tensor* weight,
    float eps,
    Tensor* out
);

/* ═══════════════════════════════════════════════════════════════
 * Positional Encoding
 * ═══════════════════════════════════════════════════════════════ */

/* Precompute RoPE frequencies (cos/sin pairs).
 * freqs_out shape: [max_seq_len, dim/2, 2]  (2 = {cos, sin}) */
VoxCPMError tensor_precompute_freqs_cis(
    int max_seq_len,
    int dim,
    float theta,
    Tensor* freqs_out
);

/* Apply rotary position embeddings to Q and K tensors.
 * q, k shape: [batch, n_heads, seq, head_dim]
 * freqs_cis shape: [max_seq_len, head_dim/2, 2]
 * pos: starting position offset */
VoxCPMError tensor_rotary_emb(
    Tensor* q,
    Tensor* k,
    const Tensor* freqs_cis,
    int pos
);

/* ═══════════════════════════════════════════════════════════════
 * View / Reshape Utilities
 * ═══════════════════════════════════════════════════════════════ */

/* Transpose 2D tensor: out[M,N] = in[N,M].
 * Supports in-place when t == out. */
VoxCPMError tensor_transpose(const Tensor* t, Tensor* out);

/* Permute dimensions (general transpose up to 4D) */
VoxCPMError tensor_permute(const Tensor* t, Tensor* out, const int* axes);

/* Concatenate two tensors along a given axis.
 * All other dimensions must match. */
VoxCPMError tensor_cat(const Tensor* a, const Tensor* b, int axis, Tensor* out);

/* Slice a tensor along a dimension. */
VoxCPMError tensor_slice(
    const Tensor* t,
    int axis,
    int start,
    int length,
    Tensor* out
);

/* ═══════════════════════════════════════════════════════════════
 * Debug / Utilities
 * ═══════════════════════════════════════════════════════════════ */

/* Set debug name (copied internally, max TENSOR_MAX_NAME chars). */
void tensor_set_name(Tensor* t, const char* name);

/* Check if tensor data contains NaN or Inf. */
bool tensor_check_nan_inf(const Tensor* t, int* bad_idx);

/* Compute sum of all elements. */
float tensor_sum(const Tensor* t);

/* Compute mean of all elements. */
float tensor_mean(const Tensor* t);

/* Compute min/max values and their indices. */
void tensor_minmax(const Tensor* t, float* min_val, float* max_val,
                    int* min_idx, int* max_idx);

/* ═══════════════════════════════════════════════════════════════
 * CUDA / GPU Acceleration
 * ═══════════════════════════════════════════════════════════════ */

#ifdef VOXCPM_CUDA

/* Initialize CUDA + cuBLAS (call once at startup).
 * Returns VOXCPM_ERR_CUDA_NOT_FOUND if no GPU available. */
VoxCPMError tensor_cuda_init(void);

/* Shutdown CUDA + cuBLAS. */
void tensor_cuda_shutdown(void);

/* Transfer tensor data between CPU and GPU.
 * Sets t->is_cuda accordingly. */
VoxCPMError tensor_to_cuda(Tensor* t);
VoxCPMError tensor_to_cpu(Tensor* t);

/* Free GPU memory for this tensor (does NOT free CPU data). */
VoxCPMError tensor_cuda_free(Tensor* t);

/* cuBLAS-accelerated matmul: C = A @ B^T
 * A [M,K], B [N,K], out [M,N] — all row-major.
 * Inputs can be CPU (auto-uploaded); output must be GPU-resident. */
VoxCPMError tensor_matmul_nt_cuda(
    const Tensor* a,
    const Tensor* b,
    Tensor* out
);

#endif /* VOXCPM_CUDA */

/* ═══════════════════════════════════════════════════════════════
 * FP16 conversion helpers
 * ═══════════════════════════════════════════════════════════════ */

/* Convert IEEE 754 FP16 (uint16_t) to FP32 (float).
 * Uses type-punning via union (C99+ well-defined). */
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1F);
    uint32_t mant = (uint32_t)(h & 0x3FF);
    union { uint32_t u; float f; } conv;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        /* Subnormal: normalize */
        exp = 1;
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        mant &= 0x3FF;
        conv.u = sign | ((exp + 112) << 23) | (mant << 13);
        return conv.f;
    } else if (exp == 31) {
        /* Infinity or NaN */
        conv.u = sign | 0x7F800000 | (mant << 13);
        return conv.f;
    }
    /* Normal number */
    conv.u = sign | ((exp + 112) << 23) | (mant << 13);
    return conv.f;
}

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_H */

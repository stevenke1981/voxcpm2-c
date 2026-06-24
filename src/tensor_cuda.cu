// tensor_cuda.cu — CUDA/cuBLAS accelerated tensor operations
// VoxCPM2-C Project
//
// Provides GPU-offloaded matmul and tensor transfer helpers
// using cuBLAS for high-performance matrix multiplication.

#include "tensor.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─────────────────────────────────────────────────────────────
 * Internal state
 * ───────────────────────────────────────────────────────────── */
static cublasHandle_t g_cublas = NULL;
static bool g_cuda_init = false;

#define CUDA_CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERR] %s:%d: %s (%d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(e), (int)e); \
        return VOXCPM_ERR_INTERNAL; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[cuBLAS ERR] %s:%d: status=%d\n", \
                __FILE__, __LINE__, (int)s); \
        return VOXCPM_ERR_INTERNAL; \
    } \
} while(0)

/* ─────────────────────────────────────────────────────────────
 * FP16 ↔ FP32 conversion kernels
 * ───────────────────────────────────────────────────────────── */
static __global__ void fp16_to_fp32_kernel(const uint16_t* src, float* dst, int n);

/* ═════════════════════════════════════════════════════════════
 * Initialization
 * ═════════════════════════════════════════════════════════════ */

VoxCPMError tensor_cuda_init(void) {
    if (g_cuda_init) return VOXCPM_SUCCESS;

    cudaError_t ce = cudaSetDevice(0);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[CUDA] No CUDA device available: %s\n",
                cudaGetErrorString(ce));
        return VOXCPM_ERR_CUDA_NOT_FOUND;
    }

    cudaDeviceProp prop;
    ce = cudaGetDeviceProperties(&prop, 0);
    if (ce == cudaSuccess) {
        fprintf(stderr, "[CUDA] Device: %s (CC %d.%d, %zu MB VRAM, PCIe %s)\n",
                prop.name, prop.major, prop.minor,
                (size_t)prop.totalGlobalMem / (1024 * 1024),
                prop.pciBusID == prop.pciDomainID ? "?" : "GEN?");
    }

    cublasStatus_t cs = cublasCreate(&g_cublas);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[CUDA] cuBLAS create failed: %d\n", (int)cs);
        return VOXCPM_ERR_CUDA_NOT_FOUND;
    }

    // Use cuBLAS DEFAULT math mode for bit-exact FP32 computation.
    // TF32 tensor cores reduce mantissa to 10 bits, causing accumulated
    // precision errors that destroy final audio quality (~87% identical
    // then diverges to complete DC offset).
    // We use standard CUDA cores for correctness; tensor core perf
    // gain is not worth the quality loss for TTS inference.

    g_cuda_init = true;
    fprintf(stderr, "[CUDA] Initialized (cuBLAS DEFAULT math mode, no TF32 tensor cores)\n");
    return VOXCPM_SUCCESS;
}

void tensor_cuda_shutdown(void) {
    if (g_cublas) {
        cublasDestroy(g_cublas);
        g_cublas = NULL;
    }
    g_cuda_init = false;
}

/* ═════════════════════════════════════════════════════════════
 * Tensor transfer helpers
 * ═════════════════════════════════════════════════════════════ */

VoxCPMError tensor_to_cuda(Tensor* t) {
    if (!t) return VOXCPM_ERR_INTERNAL;
    if (t->is_cuda) return VOXCPM_SUCCESS;
    if (t->size == 0) return VOXCPM_SUCCESS;

    /* Handle fp16 tensor — upload raw fp16 data */
    if (t->is_fp16) {
        if (!t->data_fp16) return VOXCPM_ERR_INTERNAL;
        size_t bytes = t->size * sizeof(uint16_t);
        uint16_t* d_data = NULL;
        CUDA_CHECK(cudaMalloc(&d_data, bytes));
        CUDA_CHECK(cudaMemcpy(d_data, t->data_fp16, bytes,
                              cudaMemcpyHostToDevice));
        free(t->data_fp16);
        t->data_fp16 = d_data;
        t->is_cuda = true;
        return VOXCPM_SUCCESS;
    }

    if (!t->data) return VOXCPM_SUCCESS;

    float* d_data = NULL;
    CUDA_CHECK(cudaMalloc(&d_data, t->size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, t->data, t->size * sizeof(float),
                          cudaMemcpyHostToDevice));

    free(t->data);
    t->data = d_data;
    t->is_cuda = true;
    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_to_cpu(Tensor* t) {
    if (!t) return VOXCPM_ERR_INTERNAL;
    if (!t->is_cuda) return VOXCPM_SUCCESS;
    if (t->size == 0) return VOXCPM_SUCCESS;

    /* Handle fp16 tensor — download raw fp16 data */
    if (t->is_fp16) {
        if (!t->data_fp16) return VOXCPM_ERR_INTERNAL;
        size_t bytes = t->size * sizeof(uint16_t);
        uint16_t* h_data = (uint16_t*)malloc(bytes);
        if (!h_data) return VOXCPM_ERR_OOM;
        CUDA_CHECK(cudaMemcpy(h_data, t->data_fp16, bytes,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(t->data_fp16));
        t->data_fp16 = h_data;
        t->is_cuda = false;
        return VOXCPM_SUCCESS;
    }

    if (!t->data) return VOXCPM_SUCCESS;

    float* h_data = (float*)malloc(t->size * sizeof(float));
    if (!h_data) return VOXCPM_ERR_OOM;

    CUDA_CHECK(cudaMemcpy(h_data, t->data, t->size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(t->data));

    t->data = h_data;
    t->is_cuda = false;
    return VOXCPM_SUCCESS;
}

VoxCPMError tensor_cuda_free(Tensor* t) {
    if (!t) return VOXCPM_ERR_INTERNAL;
    if (t->is_cuda) {
        if (t->is_fp16 && t->data_fp16) {
            CUDA_CHECK(cudaFree(t->data_fp16));
            t->data_fp16 = NULL;
        } else if (t->data) {
            CUDA_CHECK(cudaFree(t->data));
            t->data = NULL;
        }
        t->is_cuda = false;
    }
    return VOXCPM_SUCCESS;
}

/* ═════════════════════════════════════════════════════════════
 * cuBLAS matmul: C = A @ B^T
 *
 * All tensors are row-major.
 * A [M, K], B [N, K], C [M, N]
 *
 * cuBLAS is column-major internally.
 * We use the identity:
 *   C_row = A_row @ B_row^T
 *   → C_col (N×M) = B_col^T (N×K) @ A_col (K×M)
 *   → cublasSgemm(OP_T, OP_N, N, M, K, B, K, A, K, C, N)
 *
 * If input tensors are on CPU, they are auto-uploaded.
 * Output tensors must be on GPU (allocated by caller).
 * ═════════════════════════════════════════════════════════════ */

VoxCPMError tensor_matmul_nt_cuda(
    const Tensor* a,   /* [M, K] row-major */
    const Tensor* b,   /* [N, K] row-major */
    Tensor* out        /* [M, N] row-major, GPU-resident */
) {
    if (!a || !b || !out) return VOXCPM_ERR_INTERNAL;
    if (a->ndim != 2 || b->ndim != 2) return VOXCPM_ERR_SHAPE_MISMATCH;

    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[0];

    if (b->shape[1] != K) return VOXCPM_ERR_SHAPE_MISMATCH;
    if (out->shape[0] != M || out->shape[1] != N) return VOXCPM_ERR_SHAPE_MISMATCH;

    if (!g_cuda_init) return VOXCPM_ERR_CUDA_NOT_FOUND;

    // ── Upload activation X (a, [M, K], FP32) to GPU if needed ──
    const float* d_x = a->data;
    float* d_x_temp = NULL;
    if (!a->is_cuda) {
        CUDA_CHECK(cudaMalloc(&d_x_temp, (size_t)M * (size_t)K * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x_temp, a->data,
                              (size_t)M * (size_t)K * sizeof(float),
                              cudaMemcpyHostToDevice));
        d_x = d_x_temp;
    }

    // ── Upload weight W (b, [N, K], potentially FP16) to GPU if needed ──
    const void* d_w = NULL;   // can be float* or uint16_t*
    void* d_w_temp = NULL;
    size_t w_elem_size = b->is_fp16 ? sizeof(uint16_t) : sizeof(float);
    size_t w_bytes = (size_t)N * (size_t)K * w_elem_size;

    if (!b->is_cuda) {
        if (b->is_fp16) {
            CUDA_CHECK(cudaMalloc(&d_w_temp, w_bytes));
            CUDA_CHECK(cudaMemcpy(d_w_temp, b->data_fp16, w_bytes,
                                  cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMalloc(&d_w_temp, w_bytes));
            CUDA_CHECK(cudaMemcpy(d_w_temp, b->data, w_bytes,
                                  cudaMemcpyHostToDevice));
        }
        d_w = d_w_temp;
    } else {
        d_w = b->is_fp16 ? (const void*)b->data_fp16 : (const void*)b->data;
    }

    // ── Ensure output is on GPU ──
    float* d_out = out->data;
    float* d_out_temp = NULL;
    if (!out->is_cuda) {
        CUDA_CHECK(cudaMalloc(&d_out_temp,
                              (size_t)M * (size_t)N * sizeof(float)));
        d_out = d_out_temp;
    }

    // ── Prepare cuBLAS GEMM ──
    // Row-major: Y[M,N] = X[M,K] @ W^T[K,N]
    // Column-major (cuBLAS): Y_col[N,M] = W_col[N,K] @ X_col[K,M]
    //
    // cuBLAS: C = op(A) @ op(B)
    //   A = W_col (N×K),  op = OP_T  →  op(A) = W^T_col (K×N)
    //   B = X_col (K×M),  op = OP_N  →  op(B) = X_col   (K×M)
    //   C = op(A) @ op(B) = W^T_col @ X_col = (X @ W^T)^T = Y_col  [N×M]
    //
    // When weight is FP16, both inputs must be FP16 for cuBLAS mixed precision.
    // We convert the activation from FP32 to FP16 on GPU.
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // FP16 weights + FP32 activations → FP32 compute.
    // Convert FP16 weights to FP32 on GPU per matmul for full precision.
    // We keep FP16 on GPU (memory efficient) and convert on-the-fly.
    bool weight_is_fp16 = b->is_fp16;
    const void* d_w_gemm = d_w;
    const void* d_x_gemm = d_x;

    // FP32 weight buffer (temporary, freed after gemm)
    float* d_w_fp32 = NULL;

    cudaDataType_t w_type;   // type of A in cuBLAS (weight)
    cudaDataType_t x_type;   // type of B in cuBLAS (activation)
    cudaDataType_t c_type = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    if (weight_is_fp16) {
        // Convert FP16 weights to FP32 on GPU
        size_t w_count = (size_t)N * (size_t)K;
        CUDA_CHECK(cudaMalloc(&d_w_fp32, w_count * sizeof(float)));
        fp16_to_fp32_kernel<<<(int)((w_count+255)/256), 256>>>(
            (const uint16_t*)d_w, d_w_fp32, (int)w_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        d_w_gemm = d_w_fp32;
        w_type = CUDA_R_32F;
    } else {
        w_type = CUDA_R_32F;
    }
    x_type = CUDA_R_32F;  // Activations stay as FP32

    CUBLAS_CHECK(cublasGemmEx(
        g_cublas,
        CUBLAS_OP_T,          // transa: W_col^T (N×K → K×N)
        CUBLAS_OP_N,          // transb: X_col unchanged (K×M)
        N,                    // m: rows of op(A) = K
        M,                    // n: cols of op(B) = M
        K,                    // k: inner dimension
        &alpha,
        d_w_gemm,             // A: W_col, row-major equals N×K, col-major K×N
        w_type,               // CUDA_R_16F or CUDA_R_32F
        K,                    // lda: leading dim of A (col-major K×N → K)
        d_x_gemm,             // B: X_col, row-major K×M, col-major K×M
        x_type,               // CUDA_R_16F or CUDA_R_32F
        K,                    // ldb: leading dim of B (col-major K×M → K)
        &beta,
        d_out,                // C: Y_col = N×M
        c_type,               // CUDA_R_32F
        N,                    // ldc: leading dim of C (col-major N×M → N)
        compute_type,
        CUBLAS_GEMM_DEFAULT
    ));

    // Synchronize to surface any deferred CUDA errors before accessing data
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back if output was CPU
    if (!out->is_cuda && d_out_temp) {
        CUDA_CHECK(cudaMemcpy(out->data, d_out_temp,
                              (size_t)M * (size_t)N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_out_temp));
    }

    // Cleanup temp GPU buffers
    if (d_x_temp) CUDA_CHECK(cudaFree(d_x_temp));
    if (d_w_temp) CUDA_CHECK(cudaFree(d_w_temp));
    if (d_w_fp32) CUDA_CHECK(cudaFree(d_w_fp32));

    return VOXCPM_SUCCESS;
}

/* ─────────────────────────────────────────────────────────────
 * CUDA kernel: FP16 → FP32 conversion
 * ───────────────────────────────────────────────────────────── */
static __global__ void fp16_to_fp32_kernel(const uint16_t* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Explicit conversion: interpret uint16_t bits as __half
        __half_raw hr;
        hr.x = src[i];
        dst[i] = __half2float(hr);
    }
}

/* ═════════════════════════════════════════════════════════════
 * Upload all tensors of a model to GPU
 * ═════════════════════════════════════════════════════════════ */

VoxCPMError tensor_cuda_upload_weight(Tensor** t_ptr) {
    if (!t_ptr || !*t_ptr) return VOXCPM_SUCCESS;
    return tensor_to_cuda(*t_ptr);
}

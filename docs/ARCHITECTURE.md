# VoxCPM2-C — 深度架構設計文件

> 本文檔描述 VoxCPM2-C 的內部架構設計決策、資料流、記憶體佈局和關鍵演算法實作策略。

---

## 1. 整體架構

### 1.1 四階段推論管線

```
┌─────────────────────────────────────────────────────────────────┐
│                     VoxCPM2 推論管線                             │
│                                                                  │
│  階段1: 文字編碼                                                 │
│  ┌──────────┐    ┌──────────┐                                   │
│  │ BPE Token│───►│ LocEnc   │  local features                   │
│  │ izer     │    │ (1層)    │  [1, seq, d]                      │
│  └──────────┘    └────┬─────┘                                   │
│                        │                                         │
│  階段2: 語言模型       │                                         │
│                        ▼                                         │
│  ┌──────────────────────────────────────────────────┐           │
│  │  TSLM (Text Speech Language Model)               │           │
│  │  24× MiniCPM-4 Decoder Layer                     │           │
│  │  ┌──────────────────────────────────────────┐    │           │
│  │  │ Layer i:                                  │    │           │
│  │  │   x = RMSNorm(x)                         │    │           │
│  │  │   x = x + RoPE(MultiHeadAttn(x, kv_cache)) │    │           │
│  │  │   x = x + SwiGLU_FFN(RMSNorm(x))         │    │           │
│  │  └──────────────────────────────────────────┘    │           │
│  │  ... × 24                                       │           │
│  │  KV Cache: [24, 2, seq, n_kv_heads, head_dim]    │           │
│  └──────────────────────┬───────────────────────────┘           │
│                          │                                       │
│  階段3: 擴散去噪        │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────┐           │
│  │  LocDiT (Local Diffusion Transformer)            │           │
│  │  8層 DiT Block                                    │           │
│  │  擴散步數: 10 (DDIM)                              │           │
│  │  CFG: 2.0                                        │           │
│  │  latent: [1, T, d_latent]                       │           │
│  └──────────────────────┬───────────────────────────┘           │
│                          │                                       │
│  階段4: 音訊解碼        │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────┐           │
│  │  AudioVAE V2 Decoder                             │           │
│  │  16kHz latent → 48kHz waveform (3× upsampling)   │           │
│  │  1D Conv + Snake Activation + Residual          │           │
│  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐            │           │
│  │  │Conv1│→│Snake│→│Conv2│→│Snake│→ ... → │ConvN│→ 48kHz  │           │
│  │  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘            │           │
│  └──────────────────────┬───────────────────────────┘           │
│                          │                                       │
│                          ▼                                       │
│                   48kHz WAV Output                               │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 語音複製管線 (含 RALM)

```
Reference Audio ──► AudioVAE V2 ──► ref_latent [1, T_ref, d_latent]
                       Encoder              │
                                            ▼
Reference Text ──► Tokenizer ──► TSLM ──► ref_emb [1, T_ref, d]
                                            │
                                            ▼
Main Text ──► Tokenizer ──► TSLM ──► RALM ──► LocDiT ──► AudioVAE Dec ──► Output
                                  │         ▲
                                  │    Cross-Attention
                                  └────► Q from main, KV from ref
```

---

## 2. 記憶體佈局

### 2.1 權重 mmap 策略

```
mmap 配置:
┌────────────────────────────────────────────────────┐
│  權重檔案 (voxcpm2-q4.bin)                          │
│                                                    │
│  ┌────────────────────────────────────────────────┐│
│  │ Header (256 bytes)                             ││
│  │  - magic: "VXCPM2"                            ││
│  │  - version: uint32                            ││
│  │  - num_layers: uint32                         ││
│  │  - d_model, d_ff, n_heads, n_kv_heads        ││
│  │  - quant_type: uint32 (0=FP32,1=FP16,2=Q4)   ││
│  │  - checksum: uint64                           ││
│  │  - layer_offsets[num_layers+1]                ││
│  └────────────────────────────────────────────────┘│
│  ┌────────────────────────────────────────────────┐│
│  │ TSLM Layers (24 layers × ~150MB)              ││
│  │  Layer 0: wq, wk, wv, wo, w1, w2, w3, rms    ││
│  │  Layer 1: ...                                 ││
│  │  ...                                          ││
│  └────────────────────────────────────────────────┘│
│  ┌────────────────────────────────────────────────┐│
│  │ LocEnc Layer                                   ││
│  └────────────────────────────────────────────────┘│
│  ┌────────────────────────────────────────────────┐│
│  │ RALM Layers (4 layers)                         ││
│  └────────────────────────────────────────────────┘│
│  ┌────────────────────────────────────────────────┐│
│  │ LocDiT Layers (8 layers)                       ││
│  └────────────────────────────────────────────────┘│
│  ┌────────────────────────────────────────────────┐│
│  │ AudioVAE V2 Weights                            ││
│  └────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────┘
```

```c
// 權重指標結構（指向 mmap 區域）
typedef struct {
    // 所有權重都是 mmap 區域內的偏移指標，不佔用額外記憶體
    struct {
        Tensor wq, wk, wv, wo;
        Tensor w1, w2, w3;
        Tensor rms_att, rms_ffn;
    } tslm_layers[24];

    struct {
        Tensor wq, wk, wv, wo;
        Tensor w1, w2, w3;
        Tensor rms_att, rms_ffn;
    } loc_enc;

    struct {
        Tensor wq, wk, wv, wo;
        Tensor w1, w2, w3;
        Tensor rms_att, rms_ffn;
        Tensor wq_cross, wk_cross, wv_cross, wo_cross;  // cross-attn
    } ralm_layers[4];

    struct {
        // DiT block params
        Tensor wq, wk, wv, wo;
        Tensor w1, w2, w3;
        Tensor rms_att, rms_ffn;
        Tensor t_embed;  // time embedding
    } dit_layers[8];

    struct {
        // AudioVAE V2
        Tensor conv_in_weight, conv_in_bias;
        // ... 多層 conv
        Tensor conv_out_weight, conv_out_bias;
    } audio_vae;

    Tensor text_embed;    // 詞嵌入 [vocab_size, d_model]
} ModelWeights;
```

### 2.2 推理時記憶體佈局

```
推理時記憶體 (Arena Allocator):
┌──────────────────────────────────────────────┐
│  Persistent (模型生命週期)                    │
│  ┌────────────────────────────────────────┐  │
│  │ ModelWeights (mmap, 唯讀)             │  │
│  │ KV Cache (動態增長)                   │  │
│  │ Tokenizer 詞彙表                      │  │
│  └────────────────────────────────────────┘  │
├──────────────────────────────────────────────┤
│  Per-Generate (每次 generate 配置)           │
│  ┌────────────────────────────────────────┐  │
│  │ Layer input/output buffers             │  │
│  │ Attention scores                       │  │
│  │ FFN intermediate                       │  │
│  │ Diffusion latent                       │  │
│  │ Audio output buffer                    │  │
│  └────────────────────────────────────────┘  │
│  ✓ Arena reset O(1) 後可重複使用             │
└──────────────────────────────────────────────┘
```

### 2.3 KV Cache 設計

```c
// KV Cache 是推理效能的關鍵
typedef struct {
    Tensor* k_cache;  // [n_layers, 2, max_seq, n_kv_heads, head_dim]
    Tensor* v_cache;  // 同上
    int seq_len;      // 當前已 cache 的長度
    int max_seq;      // 最大支援長度 (2048)
    int n_layers;

    // 使用 circular buffer 管理
    int write_head;   // 當前寫入位置
} KVCache;

// 初始化
KVCache* kv_cache_create(int n_layers, int max_seq, int n_kv_heads, int head_dim) {
    KVCache* cache = (KVCache*)malloc(sizeof(KVCache));
    cache->n_layers = n_layers;
    cache->max_seq = max_seq;
    cache->seq_len = 0;

    // 預先配置所有 KV cache 空間
    int layer_shape[] = {2, max_seq, n_kv_heads, head_dim};
    for (int i = 0; i < n_layers; i++) {
        cache->k_cache[i] = tensor_create(4, layer_shape);
        cache->v_cache[i] = tensor_create(4, layer_shape);
    }
    return cache;
}

// 追加一個新 token 的 KV
void kv_cache_append(KVCache* cache, int layer_idx,
                     const Tensor* k, const Tensor* v) {
    // k shape: [1, n_kv_heads, head_dim]
    // 寫入到 cache->seq_len 位置
    int pos = cache->seq_len;
    int n_heads = k->shape[0];
    int head_dim = k->shape[1];

    for (int h = 0; h < n_heads; h++) {
        memcpy(
            TENSOR_AT(cache->k_cache[layer_idx], 0, pos, h, 0),
            TENSOR_AT(k, h, 0),
            head_dim * sizeof(float)
        );
        memcpy(
            TENSOR_AT(cache->v_cache[layer_idx], 0, pos, h, 0),
            TENSOR_AT(v, h, 0),
            head_dim * sizeof(float)
        );
    }
    cache->seq_len++;
}

// 獲取 KV cache 切片（給 attention 使用）
void kv_cache_get_slice(const KVCache* cache, int layer_idx,
                        Tensor* k_out, Tensor* v_out) {
    // 從 cache 中複製從 0 到 seq_len 的所有 KV
    int seq = cache->seq_len;
    // k_out: [1, seq, n_kv_heads, head_dim]
    memcpy(k_out->data, cache->k_cache[layer_idx]->data,
           seq * cache->k_cache[layer_idx]->shape[2] *
           cache->k_cache[layer_idx]->shape[3] * sizeof(float));
    memcpy(v_out->data, cache->v_cache[layer_idx]->data,
           seq * cache->v_cache[layer_idx]->shape[2] *
           cache->v_cache[layer_idx]->shape[3] * sizeof(float));
}
```

---

## 3. 張量運算實作策略

### 3.1 Matmul 效能層級

```c
// Level 0: 純 CPU 三重迴圈 (正確性驗證用)
void matmul_naive(const float* A, const float* B, float* C,
                  int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Level 1: 迴圈交換 + cache blocking
void matmul_blocked(const float* A, const float* B, float* C,
                    int M, int N, int K) {
    int BK = 64, BN = 64;  // tile sizes
    for (int i = 0; i < M; i++) {
        for (int jj = 0; jj < N; jj += BN) {
            int j_end = (jj + BN < N) ? jj + BN : N;
            for (int kk = 0; kk < K; kk += BK) {
                int k_end = (kk + BK < K) ? kk + BK : K;
                for (int j = jj; j < j_end; j++) {
                    float sum = 0.0f;
                    for (int k = kk; k < k_end; k++) {
                        sum += A[i * K + k] * B[k * N + j];
                    }
                    C[i * N + j] += sum;
                }
            }
        }
    }
}

// Level 2: AVX2 向量化
#ifdef __AVX2__
#include <immintrin.h>
void matmul_avx2(const float* A, const float* B, float* C,
                 int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 8) {  // 8 floats per AVX2 register
            __m256 c_vec = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 a_broadcast = _mm256_broadcast_ss(&A[i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
            }
            _mm256_storeu_ps(&C[i * N + j], c_vec);
        }
    }
}
#endif

// Level 3: OpenMP 多執行緒
void matmul_omp(const float* A, const float* B, float* C,
                int M, int N, int K) {
    #pragma omp parallel for collapse(2) if(M > 4 && N > 64)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Level 4: CUDA
// → 見 src/tensor_cuda.cu
```

### 3.2 量化線性層

```c
// Q4 量化權重格式
typedef struct {
    uint8_t q4_data[2];    // 2 個 4-bit value per byte
    float scale;           // 每個 group 共享 scale
    float min_val;         // min value for dequant
} Q4Block;  // 每個 block = 32 個 weights

// Q4 反量化
void dequant_q4(const Q4Block* blocks, float* output, int n_blocks) {
    for (int b = 0; b < n_blocks; b++) {
        float scale = blocks[b].scale;
        float min_val = blocks[b].min_val;

        for (int i = 0; i < BLOCK_SIZE; i++) {
            int byte_idx = i / 2;
            int nibble = (i % 2 == 0) ?
                (blocks[b].q4_data[byte_idx] & 0x0F) :
                ((blocks[b].q4_data[byte_idx] >> 4) & 0x0F);
            output[b * BLOCK_SIZE + i] = (float)nibble * scale + min_val;
        }
    }
}

// Q4 線性層: y = x @ W_q4^T
void linear_q4(const float* x, const Q4Block* w_q4, float* y,
               int M, int N, int K) {
    // x: [M, K], w_q4: Q4(K×N), y: [M, N]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k += BLOCK_SIZE) {
                float w[BLOCK_SIZE];
                int block_idx = (j * K + k) / BLOCK_SIZE;
                dequant_q4(&w_q4[block_idx], w, 1);

                for (int kk = 0; kk < BLOCK_SIZE; kk++) {
                    sum += x[i * K + k + kk] * w[kk];
                }
            }
            y[i * N + j] = sum;
        }
    }
}
```

---

## 4. Attention 實作

### 4.1 GQA (Grouped Query Attention)

VoxCPM2 使用 GQA 而非傳統 MHA，以減少 KV cache 大小：

```c
// GQA: n_heads = n_kv_heads * n_groups
// Q 有 n_heads, K/V 只有 n_kv_heads
// 每個 K/V head 被 n_groups 個 Q head 共享

void attention_gqa(const Tensor* Q, const Tensor* K, const Tensor* V,
                   Tensor* out, int n_heads, int n_kv_heads, bool causal) {
    int batch = Q->shape[0];
    int seq_q = Q->shape[1];
    int seq_k = K->shape[1];
    int head_dim = Q->shape[2] / n_heads;
    int n_groups = n_heads / n_kv_heads;

    // 暫存 attention scores: [batch, n_heads, seq_q, seq_k]
    int score_shape[] = {batch, n_heads, seq_q, seq_k};
    Tensor* scores = tensor_create(4, score_shape);

    // Q: [batch, seq_q, n_heads * head_dim] → [batch, n_heads, seq_q, head_dim]
    // K: [batch, seq_k, n_kv_heads * head_dim] → [batch, n_kv_heads, seq_k, head_dim]
    // 計算 score = Q @ K^T / sqrt(head_dim)
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / n_groups;  // 對應的 KV head
            for (int q_pos = 0; q_pos < seq_q; q_pos++) {
                for (int k_pos = 0; k_pos < seq_k; k_pos++) {
                    // dot product
                    float sum = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        float q_val = TENSOR_GET(Q, b, q_pos, h * head_dim + d);
                        float k_val = TENSOR_GET(K, b, k_pos, kv_h * head_dim + d);
                        sum += q_val * k_val;
                    }
                    sum *= scale;

                    // Causal mask
                    if (causal && k_pos > q_pos) {
                        sum = -INFINITY;
                    }

                    TENSOR_SET(scores, b, h, q_pos, k_pos, sum);
                }
            }
        }
    }

    // Softmax over k_pos
    tensor_softmax_4d(scores, 3);  // softmax on last dim

    // Score @ V
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / n_groups;
            for (int q_pos = 0; q_pos < seq_q; q_pos++) {
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
                        float s = TENSOR_GET(scores, b, h, q_pos, k_pos);
                        float v_val = TENSOR_GET(V, b, k_pos, kv_h * head_dim + d);
                        sum += s * v_val;
                    }
                    TENSOR_SET(out, b, q_pos, h * head_dim + d, sum);
                }
            }
        }
    }

    tensor_free(scores);
}
```

### 4.2 Flash Attention (使用 loop 模擬)

```c
// 不計算完整 attention matrix，以 tile 方式逐塊處理
// 節省記憶體（不必儲存完整 seq_q × seq_k 的 score matrix）

void flash_attention_tiled(const Tensor* Q, const Tensor* K, const Tensor* V,
                           Tensor* out, int n_heads, int n_kv_heads,
                           int tile_size) {
    // 每個 tile 計算一個 block 的 softmax
    // 跨 tile 使用 rescaling 技巧合併
    // 參考: FlashAttention: Fast and Memory-Efficient Exact Attention
    // 實作省略 (細節實現見 src/nn.c)
}
```

---

## 5. 擴散模型實作 (LocDiT)

### 5.1 噪聲調度

```c
// 使用 zero-SNR 噪聲調度 (cosine schedule 變體)
typedef struct {
    float* betas;           // [T] noise variance
    float* alphas;          // [T] 1 - betas
    float* alphas_cumprod;  // [T] cumprod of alphas
    float* sqrt_alphas_cumprod;
    float* sqrt_one_minus_alphas_cumprod;
    int num_timesteps;       // 擴散步數
} NoiseSchedule;

NoiseSchedule* noise_schedule_create(int num_timesteps) {
    NoiseSchedule* s = malloc(sizeof(NoiseSchedule));
    s->num_timesteps = num_timesteps;

    // Cosine schedule
    for (int t = 0; t < num_timesteps; t++) {
        float progress = (float)t / num_timesteps;
        float alpha_bar = cosf((progress + 0.008f) / 1.008f * M_PI_2);
        alpha_bar = alpha_bar * alpha_bar;  // cosine^2
        s->alphas_cumprod[t] = CLAMP(alpha_bar, 0.001f, 0.999f);
    }

    // 計算衍生值
    for (int t = 0; t < num_timesteps; t++) {
        s->sqrt_alphas_cumprod[t] = sqrtf(s->alphas_cumprod[t]);
        s->sqrt_one_minus_alphas_cumprod[t] = sqrtf(1.0f - s->alphas_cumprod[t]);
        s->alphas[t] = t > 0 ?
            s->alphas_cumprod[t] / s->alphas_cumprod[t-1] :
            s->alphas_cumprod[0];
        s->betas[t] = 1.0f - s->alphas[t];
    }

    return s;
}
```

### 5.2 DDIM 採樣

```c
// DDIM 採樣: 從噪聲逐步去噪
// 比 DDPM 快 (10 steps vs 1000 steps)，品質相近

void ddim_sample(const NoiseSchedule* noise, const LocDiT* model,
                 Tensor* latent, int num_steps, float eta, float cfg_scale) {
    // 初始 latent: 純高斯噪聲 [1, T, d_latent]
    int T = latent->shape[1];
    int d = latent->shape[2];

    // 從 num_timesteps-1 開始向後採樣
    int skip = noise->num_timesteps / num_steps;

    for (int step = num_steps - 1; step >= 0; step--) {
        int t = step * skip;

        // 模型預測噪聲
        Tensor* pred_noise = tensor_create(3, (int[]){1, T, d});
        loc_dit_forward(model, latent, t, cfg_scale, pred_noise);

        // DDIM 更新公式:
        // x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 +
        //           sqrt(1 - alpha_{t-1}) * noise_direction

        float alpha_t = noise->alphas_cumprod[t];
        float alpha_t_prev = (step > 0) ?
            noise->alphas_cumprod[(step - 1) * skip] : 1.0f;

        float pred_x0_coeff = sqrtf(alpha_t_prev / alpha_t);
        float noise_coeff = sqrtf(1.0f - alpha_t_prev);

        float eta_scaled = eta * sqrtf((1 - alpha_t_prev) / (1 - alpha_t));

        for (int i = 0; i < T * d; i++) {
            float x_t = latent->data[i];
            float eps = pred_noise->data[i];

            float pred_x0 = (x_t - sqrtf(1.0f - alpha_t) * eps) / sqrtf(alpha_t);

            // 添加隨機噪聲 (eta=0 時為確定性採樣)
            float noise_part = 0.0f;
            if (eta > 0 && step > 0) {
                noise_part = randn() * eta_scaled;
            }

            latent->data[i] = pred_x0_coeff * pred_x0 + noise_coeff * noise_part;
        }

        tensor_free(pred_noise);
    }
}
```

### 5.3 CFG (Classifier-Free Guidance)

```c
// CFG: 同時推理 conditioned 和 unconditioned，然後外插
void loc_dit_forward_with_cfg(const LocDiT* model, Tensor* latent, int t,
                              float cfg_scale, Tensor* output) {
    if (cfg_scale <= 1.0f) {
        // 無 CFG，直接 forward
        loc_dit_forward(model, latent, t, NULL, output);
        return;
    }

    // Conditioned forward (有 text condition)
    Tensor* cond_out = tensor_create(3, (int[]){1, latent->shape[1], latent->shape[2]});
    loc_dit_forward(model, latent, t, &model->condition, cond_out);

    // Unconditioned forward (使用 null condition)
    Tensor* uncond_out = tensor_create(3, (int[]){1, latent->shape[1], latent->shape[2]});
    loc_dit_forward(model, latent, t, NULL, uncond_out);

    // CFG外插: output = uncond + cfg_scale * (cond - uncond)
    for (int i = 0; i < latent->size; i++) {
        output->data[i] = uncond_out->data[i] +
                          cfg_scale * (cond_out->data[i] - uncond_out->data[i]);
    }

    tensor_free(cond_out);
    tensor_free(uncond_out);
}
```

---

## 6. AudioVAE V2 詳細設計

### 6.1 架構

```
AudioVAE V2 非對稱編解碼器:
┌────────────────────────────────────┐
│ Encoder (僅 clone 模式使用)        │
│  16kHz PCM                         │
│  │                                 │
│  ├── Conv1D (downsample 2×)       │
│  ├── SnakeAct                      │
│  ├── Conv1D (downsample 2×)       │
│  ├── SnakeAct                      │
│  └── ... (總共 downsample 到 ~6Hz) │
│  → latent                          │
└────────────────────────────────────┘
                                        ┌────────────────────────────────────┐
                                        │ Decoder (always used for output)  │
                                        │  latent (16kHz rate)              │
                                        │  │                                 │
                                        │  ├── ConvTranspose1D (upsample 2×)│
                                        │  ├── SnakeAct                      │
                                        │  ├── Conv1D (residual)            │
                                        │  ├── SnakeAct                      │
                                        │  ├── ConvTranspose1D (upsample 2×)│
                                        │  ├── SnakeAct                      │
                                        │  └── ... (3× upsampling to 48kHz) │
                                        │  → 48kHz PCM                       │
                                        └────────────────────────────────────┘
```

### 6.2 Snake 激活函數

```c
// Snake: x + sin^2(alpha * x) / alpha
// 比 SiLU/ReLU 更適合音訊生成（週期性歸納偏置）

float snake_activation(float x, float alpha) {
    return x + sinf(alpha * x) * sinf(alpha * x) / alpha;
}

void snake_forward(const Tensor* input, float alpha, Tensor* output) {
    for (int i = 0; i < input->size; i++) {
        output->data[i] = snake_activation(input->data[i], alpha);
    }
}
```

---

## 7. Tokenizer 設計

```c
// VoxCPM2 使用 BPE tokenizer (與 MiniCPM-4 相同)
// 詞彙表大小: ~50k

typedef struct {
    char** vocab;        // [vocab_size] token strings
    float* embeddings;   // [vocab_size, d_model] token embeddings
    int vocab_size;
    int d_model;

    // BPE merge table
    struct {
        int left;
        int right;
    } *merges;
    int num_merges;
} Tokenizer;

// 編碼: 文字 → token IDs
int* tokenizer_encode(const Tokenizer* tok, const char* text, int* out_len) {
    // 1. 文字預處理 (normalization)
    // 2. BPE分詞 (byte-level BPE)
    // 3. 轉為 token IDs
}

// 解碼: token IDs → 文字
char* tokenizer_decode(const Tokenizer* tok, const int* tokens, int len);
```

---

## 8. 多執行緒設計

```c
// VoxCPM2-C 支援三層並行：
//
// Layer 1: Operator-level 並行 (OpenMP)
//   matmul, attention 內的迴圈層級
//
// Layer 2: Batch-level 並行 (pthreads/Win32)
//   多段文字同時推理
//
// Layer 3: Pipeline 並行 (計畫中)
//   LocEnc → TSLM → LocDiT → AudioVAE 管線化

// OpenMP 範例 (src/tensor.c)
void tensor_matmul_parallel(Tensor* a, Tensor* b, Tensor* out) {
    int M = a->shape[0];
    int N = b->shape[1];
    int K = a->shape[1];

    #pragma omp parallel for collapse(2) num_threads(g_n_threads) \
        if(M * N > 1024) schedule(dynamic, 8)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < K; k++) {
                sum += TENSOR_GET(a, i, k) * TENSOR_GET(b, k, j);
            }
            TENSOR_SET(out, i, j, sum);
        }
    }
}
```

---

## 9. CUDA 後端設計

```cuda
// src/tensor_cuda.cu — CUDA kernel 範例

// Matmul: M×K × K×N → M×N
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    // 使用共享記憶體的 tile 化 matmul
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 協同載入 tile
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// GPU 記憶體管理
typedef struct {
    float* device_ptr;
    size_t size;
    bool is_pinned;  // 頁面鎖定記憶體 (faster H2D/D2H)
} GPUBuffer;

// Memory pool 減少 cudaMalloc 開銷
typedef struct {
    struct {
        float* ptr;
        size_t size;
        bool in_use;
    } blocks[MAX_POOL_BLOCKS];
    int n_blocks;
} GPUMemoryPool;
```

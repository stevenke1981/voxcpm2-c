# VoxCPM2-C — 演算法參考 (Algorithm Reference)

> 本文檔收錄 VoxCPM2 所需的所有演算法公式與對應的 C 實作參考。

---

## 1. RMSNorm

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

```c
void rmsnorm(float* x, float* weight, float* out, int n, float eps) {
    // 計算均方根
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    float rms = sqrtf(ss / n + eps);
    float inv_rms = 1.0f / rms;

    for (int i = 0; i < n; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

// 融合版本 (fused into attention/FFN):
// 不需額外 buffer，直接在 forward 中 inline 計算
```

---

## 2. RoPE (旋轉位置編碼)

```
θ_i = 10000^(-2i/d)   for i = 0, 1, ..., d/2-1

pos 位置, 維度對 (2i, 2i+1):
  RoPE(x_pos, 2i)   = x_pos[2i] * cos(pos*θ_i) - x_pos[2i+1] * sin(pos*θ_i)
  RoPE(x_pos, 2i+1) = x_pos[2i] * sin(pos*θ_i) + x_pos[2i+1] * cos(pos*θ_i)
```

```c
// 預計算 sin/cos 表 (一次初始化，多次使用)
float* precompute_freqs_cis(int dim, int max_seq_len, float theta) {
    float* freqs_cis = (float*)malloc(max_seq_len * dim * 2 * sizeof(float));
    // 每個位置儲存 [cos, sin] pair

    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / dim);
            float val = pos * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);

            freqs_cis[(pos * dim + i) * 2 + 0] = cos_val;
            freqs_cis[(pos * dim + i) * 2 + 1] = sin_val;
        }
    }
    return freqs_cis;
}

// 應用 RoPE 到 Q 和 K
void apply_rotary_emb(float* q, float* k, int seq_len, int dim,
                      const float* freqs_cis, int pos_offset) {
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < dim; i += 2) {
            int idx = (pos_offset + pos) * dim + i;
            float cos_val = freqs_cis[idx * 2 + 0];
            float sin_val = freqs_cis[idx * 2 + 1];

            float q0 = q[pos * dim + i];
            float q1 = q[pos * dim + i + 1];
            q[pos * dim + i]     = q0 * cos_val - q1 * sin_val;
            q[pos * dim + i + 1] = q0 * sin_val + q1 * cos_val;

            float k0 = k[pos * dim + i];
            float k1 = k[pos * dim + i + 1];
            k[pos * dim + i]     = k0 * cos_val - k1 * sin_val;
            k[pos * dim + i + 1] = k0 * sin_val + k1 * cos_val;
        }
    }
}
```

---

## 3. SwiGLU FFN

```
SwiGLU(x) = (SiLU(x @ W1) ⊙ (x @ W3)) @ W2

SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
```

```c
void swiglu_ffn(const float* x, int d_model, int d_ff,
                const float* w1, const float* w2, const float* w3,
                float* out) {
    // x: [d_model], w1: [d_model, d_ff], w2: [d_ff, d_model], w3: [d_model, d_ff]

    // hidden1 = x @ w1  [d_ff]
    // hidden3 = x @ w3  [d_ff]
    // gate = silu(hidden1) * hidden3  [d_ff]
    // out = gate @ w2  [d_model]

    float* hidden1 = (float*)malloc(d_ff * sizeof(float));
    float* hidden3 = (float*)malloc(d_ff * sizeof(float));

    // x @ w1
    for (int j = 0; j < d_ff; j++) {
        float sum = 0.0f;
        for (int i = 0; i < d_model; i++) {
            sum += x[i] * w1[i * d_ff + j];
        }
        hidden1[j] = sum;
    }

    // x @ w3
    for (int j = 0; j < d_ff; j++) {
        float sum = 0.0f;
        for (int i = 0; i < d_model; i++) {
            sum += x[i] * w3[i * d_ff + j];
        }
        hidden3[j] = sum;
    }

    // SiLU gate
    for (int j = 0; j < d_ff; j++) {
        hidden1[j] = hidden1[j] / (1.0f + expf(-hidden1[j]));  // SiLU
        hidden1[j] *= hidden3[j];  // element-wise multiply
    }

    // gate @ w2
    for (int i = 0; i < d_model; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d_ff; j++) {
            sum += hidden1[j] * w2[j * d_model + i];
        }
        out[i] = sum;
    }

    free(hidden1);
    free(hidden3);
}
```

---

## 4. Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

```c
// 單頭 attention
void attention_head(const float* q, const float* k, const float* v,
                    float* out, int seq_q, int seq_k, int head_dim,
                    bool causal, const bool* mask) {
    // scores = Q @ K^T, 形狀 [seq_q, seq_k]
    float* scores = (float*)malloc(seq_q * seq_k * sizeof(float));

    for (int i = 0; i < seq_q; i++) {
        float max_score = -INFINITY;

        for (int j = 0; j < seq_k; j++) {
            if (causal && j > i) {
                scores[i * seq_k + j] = -INFINITY;
            } else if (mask && !mask[j]) {
                scores[i * seq_k + j] = -INFINITY;
            } else {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    sum += q[i * head_dim + d] * k[j * head_dim + d];
                }
                sum /= sqrtf((float)head_dim);
                scores[i * seq_k + j] = sum;
                if (sum > max_score) max_score = sum;
            }
        }

        // Softmax (numerically stable)
        float exp_sum = 0.0f;
        for (int j = 0; j < seq_k; j++) {
            if (scores[i * seq_k + j] > -INFINITY / 2) {
                scores[i * seq_k + j] = expf(scores[i * seq_k + j] - max_score);
                exp_sum += scores[i * seq_k + j];
            }
        }
        float inv_exp_sum = 1.0f / exp_sum;
        for (int j = 0; j < seq_k; j++) {
            if (scores[i * seq_k + j] > -INFINITY / 2) {
                scores[i * seq_k + j] *= inv_exp_sum;
            }
        }

        // Weighted sum of V
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j < seq_k; j++) {
                sum += scores[i * seq_k + j] * v[j * head_dim + d];
            }
            out[i * head_dim + d] = sum;
        }
    }

    free(scores);
}
```

---

## 5. DDIM 擴散採樣

### 5.1 正向擴散

```
q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) * x_0, (1 - ᾱ_t) * I)
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε, ε ~ N(0, I)
```

### 5.2 逆向去噪 (DDIM)

```
給定 x_t 和模型預測噪聲 ε_θ(x_t, t, c):

x_0_pred = (x_t - sqrt(1 - ᾱ_t) * ε_θ) / sqrt(ᾱ_t)
x_{t-1} = sqrt(ᾱ_{t-1}) * x_0_pred + sqrt(1 - ᾱ_{t-1}) * ε_θ

（確定性 DDIM, η=0）
```

```c
void ddim_step(const float* x_t, const float* noise_pred,
               float* x_t_prev, int n_elements,
               float alpha_bar_t, float alpha_bar_t_prev) {
    float sqrt_alpha_t = sqrtf(alpha_bar_t);
    float sqrt_one_minus_alpha_t = sqrtf(1.0f - alpha_bar_t);
    float sqrt_alpha_t_prev = sqrtf(alpha_bar_t_prev);
    float sqrt_one_minus_alpha_t_prev = sqrtf(1.0f - alpha_bar_t_prev);

    for (int i = 0; i < n_elements; i++) {
        // Predict x_0
        float x_0_pred = (x_t[i] - sqrt_one_minus_alpha_t * noise_pred[i]) / sqrt_alpha_t;

        // DDIM step
        x_t_prev[i] = sqrt_alpha_t_prev * x_0_pred + sqrt_one_minus_alpha_t_prev * noise_pred[i];
    }
}
```

---

## 6. CFG (Classifier-Free Guidance)

```
ε_guided = ε_uncond + cfg_scale × (ε_cond - ε_uncond)

where:
  ε_cond = ε_θ(x_t, t, condition)     — conditioned prediction
  ε_uncond = ε_θ(x_t, t, NULL)        — unconditioned prediction
  cfg_scale = 2.0 (建議值)
```

```c
void apply_cfg(float* noise_out, const float* cond, const float* uncond,
               int n_elements, float cfg_scale) {
    for (int i = 0; i < n_elements; i++) {
        noise_out[i] = uncond[i] + cfg_scale * (cond[i] - uncond[i]);
    }
}
```

---

## 7. AudioVAE V2 1D Convolution

### 7.1 Conv1D 前向

```c
// 一維卷積: y[n] = Σ_k x[n × stride + k] × w[k] + bias
// 支援空洞卷積 (dilation)

void conv1d_forward(const float* input, int in_channels, int in_length,
                    const float* weight, const float* bias,
                    int out_channels, int kernel_size, int stride, int dilation,
                    float* output, int* out_length) {
    *out_length = (in_length - dilation * (kernel_size - 1) - 1) / stride + 1;

    for (int oc = 0; oc < out_channels; oc++) {
        for (int pos = 0; pos < *out_length; pos++) {
            float sum = bias ? bias[oc] : 0.0f;

            for (int ic = 0; ic < in_channels; ic++) {
                for (int k = 0; k < kernel_size; k++) {
                    int in_pos = pos * stride + k * dilation;
                    if (in_pos >= 0 && in_pos < in_length) {
                        int w_idx = oc * (in_channels * kernel_size) +
                                    ic * kernel_size + k;
                        sum += input[ic * in_length + in_pos] * weight[w_idx];
                    }
                }
            }

            output[oc * (*out_length) + pos] = sum;
        }
    }
}
```

### 7.2 ConvTranspose1D (轉置卷積，用於升頻)

```c
// 轉置卷積: 將 16kHz latent 升頻至 48kHz
// y[n] = Σ_k x[(n - k) / stride] × w[k]  (where index integer)

void convtranspose1d_forward(const float* input, int in_channels, int in_length,
                             const float* weight, const float* bias,
                             int out_channels, int kernel_size, int stride,
                             float* output, int* out_length) {
    *out_length = (in_length - 1) * stride + kernel_size;

    // 初始化為 bias
    for (int oc = 0; oc < out_channels; oc++) {
        for (int pos = 0; pos < *out_length; pos++) {
            output[oc * (*out_length) + pos] = bias ? bias[oc] : 0.0f;
        }
    }

    // 散佈相加 (scatter-add)
    for (int ic = 0; ic < in_channels; ic++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int i = 0; i < in_length; i++) {
                float val = input[ic * in_length + i];
                for (int k = 0; k < kernel_size; k++) {
                    int out_pos = i * stride + k;
                    int w_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k;
                    output[oc * (*out_length) + out_pos] += val * weight[w_idx];
                }
            }
        }
    }
}
```

---

## 8. 音訊後處理

### 8.1 音量正規化

```c
void normalize_audio(float* samples, int n, float target_peak) {
    // 找 peak
    float peak = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_val = fabsf(samples[i]);
        if (abs_val > peak) peak = abs_val;
    }

    if (peak > 0.001f) {
        float gain = target_peak / peak;
        for (int i = 0; i < n; i++) {
            samples[i] *= gain;
        }
    }
}

void normalize_loudness(float* samples, int n, int sample_rate, float target_lufs) {
    // 簡易 RMS 正規化
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += samples[i] * samples[i];
    }
    float rms = sqrtf(sum_sq / n);
    float target_rms = powf(10.0f, target_lufs / 20.0f);

    if (rms > 0.0001f) {
        float gain = target_rms / rms;
        for (int i = 0; i < n; i++) {
            samples[i] *= gain;
        }
    }
}
```

### 8.2 簡易降噪

```c
void simple_denoise(float* samples, int n, int sample_rate) {
    // 簡單的頻譜閘 (spectral gating) 簡化版
    // 背景噪聲估計: 使用前 100ms 的 RMS
    int noise_window = (int)(0.1f * sample_rate);
    if (noise_window > n) noise_window = n;

    float noise_rms = 0.0f;
    for (int i = 0; i < noise_window; i++) {
        noise_rms += samples[i] * samples[i];
    }
    noise_rms = sqrtf(noise_rms / noise_window);

    // Soft noise gate
    float threshold = noise_rms * 2.0f;
    for (int i = 0; i < n; i++) {
        float abs_val = fabsf(samples[i]);
        if (abs_val < threshold) {
            float gain = (abs_val / threshold);
            gain = gain * gain;  // quadratic knee
            samples[i] *= gain;
        }
    }
}
```

---

## 9. WAV 檔案 I/O

```c
// WAV 格式:
// ┌──────────┬──────────────────────────────────────┐
// │ RIFF     │ "RIFF" (4 bytes)                     │
// │ 大小     │ uint32 (檔案大小 - 8)                │
// │ WAVE     │ "WAVE" (4 bytes)                     │
// │ fmt      │ "fmt " (4 bytes)                     │
// │ fmt 大小  │ uint32 (16 for PCM)                 │
// │ 格式     │ uint16 (1 = PCM)                    │
// │ 聲道     │ uint16 (1 = mono)                   │
// │ 採樣率   │ uint32 (48000)                       │
// │ 位元率   │ uint32 (sample_rate * bps/8)         │
// │ block    │ uint16 (bps/8 * channels)            │
// │ bps      │ uint16 (16)                          │
// │ data     │ "data" (4 bytes)                     │
// │ data大小 │ uint32                              │
// │ 音訊資料 │ PCM samples                          │
// └──────────┴──────────────────────────────────────┘

VoxCPMError audio_write_wav(const char* path, const float* samples,
                             int num_samples, int sample_rate) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return VOXCPM_ERR_FILE_NOT_FOUND;

    int bits_per_sample = 16;
    int num_channels = 1;
    int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    int block_align = num_channels * bits_per_sample / 8;
    int data_size = num_samples * block_align;
    int file_size = 36 + data_size;

    // RIFF header
    fwrite("RIFF", 1, 4, fp);
    fwrite(&file_size, 4, 1, fp);
    fwrite("WAVE", 1, 4, fp);

    // fmt chunk
    int fmt_size = 16;
    short audio_fmt = 1;  // PCM
    short num_chan = num_channels;
    int sample_rate_le = sample_rate;
    short bps = bits_per_sample;

    fwrite("fmt ", 1, 4, fp);
    fwrite(&fmt_size, 4, 1, fp);
    fwrite(&audio_fmt, 2, 1, fp);
    fwrite(&num_chan, 2, 1, fp);
    fwrite(&sample_rate_le, 4, 1, fp);
    fwrite(&byte_rate, 4, 1, fp);
    fwrite(&block_align, 2, 1, fp);
    fwrite(&bps, 2, 1, fp);

    // data chunk
    fwrite("data", 1, 4, fp);
    fwrite(&data_size, 4, 1, fp);

    // 轉換 float PCM → int16 並寫入
    for (int i = 0; i < num_samples; i++) {
        // clamp to [-1, 1]
        float s = samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;

        int16_t sample_i16 = (int16_t)(s * 32767.0f);
        fwrite(&sample_i16, sizeof(int16_t), 1, fp);
    }

    fclose(fp);
    return VOXCPM_SUCCESS;
}

VoxCPMError audio_read_wav(const char* path, float** samples,
                           int* num_samples, int* sample_rate) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return VOXCPM_ERR_FILE_NOT_FOUND;

    char riff[4], wave[4], chunk_id[4];
    uint32_t chunk_size;
    uint16_t audio_fmt, num_channels, bps;
    uint32_t fmt_sample_rate, data_size;

    fread(riff, 1, 4, fp);
    fread(&chunk_size, 4, 1, fp);
    fread(wave, 1, 4, fp);

    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) {
        fclose(fp);
        return VOXCPM_ERR_INVALID_AUDIO;
    }

    // 搜尋 fmt chunk
    while (1) {
        fread(chunk_id, 1, 4, fp);
        fread(&chunk_size, 4, 1, fp);
        if (memcmp(chunk_id, "fmt ", 4) == 0) break;
        fseek(fp, chunk_size, SEEK_CUR);
    }

    fread(&audio_fmt, 2, 1, fp);
    fread(&num_channels, 2, 1, fp);
    fread(&fmt_sample_rate, 4, 1, fp);
    fseek(fp, 6, SEEK_CUR);  // skip byte_rate + block_align
    fread(&bps, 2, 1, fp);
    fseek(fp, chunk_size - 16, SEEK_CUR);  // skip remaining fmt

    // 搜尋 data chunk
    while (1) {
        fread(chunk_id, 1, 4, fp);
        fread(&data_size, 4, 1, fp);
        if (memcmp(chunk_id, "data", 4) == 0) break;
        fseek(fp, chunk_size, SEEK_CUR);
    }

    *sample_rate = fmt_sample_rate;
    *num_samples = data_size / (bps / 8) / num_channels;
    *samples = (float*)malloc(*num_samples * sizeof(float));

    // 讀取並轉換到 float
    if (bps == 16) {
        for (int i = 0; i < *num_samples; i++) {
            int16_t sample_i16;
            fread(&sample_i16, 2, 1, fp);
            (*samples)[i] = (float)sample_i16 / 32767.0f;
        }
    }

    fclose(fp);
    return VOXCPM_SUCCESS;
}
```

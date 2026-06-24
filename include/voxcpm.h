#ifndef VOXCPM_H
#define VOXCPM_H

/*
 * VoxCPM2-C Public API
 * =====================
 * Pure C inference engine for VoxCPM2 TTS model.
 *
 * Usage:
 *   VoxCPMModelConfig cfg = { .model_path = "model.vxcpm", .n_threads = 4 };
 *   VoxCPMError err;
 *   VoxCPMModel* model = voxcpm_create(&cfg, &err);
 *
 *   VoxCPMAudio audio;
 *   voxcpm_generate(model, &(VoxCPMGenConfig){.text = "Hello"}, &audio);
 *   // audio.samples, audio.num_samples, audio.sample_rate
 *   voxcpm_audio_free(&audio);
 *   voxcpm_free(model);
 *
 * Compile with: -I include/ -lm
 * Link with:   libvoxcpm2.a (static) or libvoxcpm2.so (shared)
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 * Version
 * ═══════════════════════════════════════════════════════════════ */
#define VOXCPM_VERSION_MAJOR 0
#define VOXCPM_VERSION_MINOR 1
#define VOXCPM_VERSION_PATCH 0
#define VOXCPM_VERSION "0.1.0"

const char* voxcpm_version(void);

/* ═══════════════════════════════════════════════════════════════
 * Error Codes
 * ═══════════════════════════════════════════════════════════════ */
typedef enum {
    VOXCPM_SUCCESS               =  0,
    VOXCPM_ERR_FILE_NOT_FOUND    = -1,  /* 權重檔案不存在 */
    VOXCPM_ERR_INVALID_MODEL     = -2,  /* 檔案格式錯誤 (magic/version) */
    VOXCPM_ERR_INVALID_TEXT      = -3,  /* 輸入文字無效 */
    VOXCPM_ERR_INVALID_AUDIO     = -4,  /* 輸入音檔無效 */
    VOXCPM_ERR_SHAPE_MISMATCH    = -5,  /* Tensor shape 不符合 */
    VOXCPM_ERR_OOM               = -6,  /* 記憶體不足 */
    VOXCPM_ERR_GPU               = -7,  /* GPU 錯誤 */
    VOXCPM_ERR_INTERNAL          = -8,  /* 內部錯誤 */
    VOXCPM_ERR_UNSUPPORTED       = -9,  /* 不支援的操作 */
    VOXCPM_ERR_TIMEOUT           = -10, /* 生成逾時 */
    VOXCPM_ERR_CUDA_NOT_FOUND    = -11, /* CUDA 不可用 */
} VoxCPMError;

/* 取得錯誤碼的文字說明 (thread-safe, 不回傳 NULL) */
const char* voxcpm_error_string(VoxCPMError err);

/* ═══════════════════════════════════════════════════════════════
 * Logging
 * ═══════════════════════════════════════════════════════════════ */
typedef enum {
    VOXCPM_LOG_QUIET = 0,   /* 不輸出 */
    VOXCPM_LOG_ERROR = 1,   /* 僅錯誤 */
    VOXCPM_LOG_WARN  = 2,   /* +警告 */
    VOXCPM_LOG_INFO  = 3,   /* +資訊 (預設) */
    VOXCPM_LOG_DEBUG = 4,   /* +除錯 */
} VoxCPMLogLevel;

void voxcpm_set_log_level(VoxCPMLogLevel level);
VoxCPMLogLevel voxcpm_get_log_level(void);

/* ═══════════════════════════════════════════════════════════════
 * GPU Detection
 * ═══════════════════════════════════════════════════════════════ */
/* 回傳可用的 GPU 數量 (0 = 無 GPU) */
int voxcpm_gpu_count(void);

/* 回傳 GPU i 的名稱 (buffer 至少 256 bytes) */
void voxcpm_gpu_name(int device_id, char* buffer, int buffer_size);

/* ═══════════════════════════════════════════════════════════════
 * Model Configuration
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    const char* model_path;       /* .vxcpm 權重檔案路徑 (必要) */
    int         n_threads;        /* CPU 執行緒數 (預設: 4) */
    bool        use_gpu;          /* 啟用 GPU 加速 (預設: false) */
    int         gpu_device;       /* GPU 裝置 ID (預設: 0) */
    const char* quantization;     /* 量化模式: "q4", "q8", "f16", "f32" (預設: 自動偵測) */
    size_t      max_seq_len;      /* 最大序列長度 (預設: 2048) */
    size_t      gpu_memory_limit; /* GPU 記憶體上限 (bytes, 0=自動) */
    int         verbosity;        /* 詳細程度 0-4 (預設: 2) */
} VoxCPMModelConfig;

/* 載入預設設定 (欄位填入合理的預設值) */
VoxCPMModelConfig voxcpm_model_config_default(void);

/* ═══════════════════════════════════════════════════════════════
 * Generation Configuration
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    const char* text;               /* 輸入文字 (必要) */
    const char* reference_wav;      /* 參考音檔路徑 (複製模式, 可為 NULL) */
    const char* prompt_text;        /* 參考音檔逐字稿 (終極複製, 可為 NULL) */
    const char* voice_design;       /* 語音描述 (設計模式, 可為 NULL) */
    float       cfg_value;          /* CFG 強度 (預設: 2.0, 範圍: 1.0-5.0) */
    int         inference_timesteps; /* 擴散步數 (預設: 10, 範圍: 3-50) */
    bool        denoise;            /* 啟用降噪 (預設: true) */
    bool        normalize;          /* 啟用音量正規化 (預設: true) */
    int         seed;               /* 隨機種子 (0 = 隨機) */
    int         max_new_tokens;     /* 最多生成 token 數 (預設: 1024) */
    float       temperature;        /* 採樣溫度 (預設: 1.0, 僅未來使用) */
    int         timeout_ms;         /* 生成逾時毫秒 (0 = 不限) */
} VoxCPMGenConfig;

/* 載入預設生成設定 */
VoxCPMGenConfig voxcpm_gen_config_default(void);

/* ═══════════════════════════════════════════════════════════════
 * Audio Output
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    float* samples;         /* PCM 音訊樣本 [-1.0, 1.0] */
    int    num_samples;     /* 樣本數 */
    int    sample_rate;     /* 採樣率 (24000 或 48000) */
} VoxCPMAudio;

/* 釋放音訊 (samples 為 NULL 時安全) */
void voxcpm_audio_free(VoxCPMAudio* audio);

/* 儲存音訊到 WAV 檔案 */
VoxCPMError voxcpm_audio_save(const VoxCPMAudio* audio, const char* path);

/* ═══════════════════════════════════════════════════════════════
 * Model Lifecycle
 * ═══════════════════════════════════════════════════════════════ */
typedef struct VoxCPMModel VoxCPMModel;

/*
 * 建立模型
 * 載入權重、初始化 KV cache、準備 tokenizer。
 * 成功回傳非 NULL 指標，err 設為 VOXCPM_SUCCESS。
 * 失敗回傳 NULL，err 設為對應錯誤碼。
 */
VoxCPMModel* voxcpm_create(const VoxCPMModelConfig* config, VoxCPMError* err);

/*
 * 釋放模型
 * model 為 NULL 時安全。
 */
void voxcpm_free(VoxCPMModel* model);

/*
 * 取得模型資訊 (JSON 格式字串，需呼叫者 free)
 * 包含: d_model, n_layers, quant_type, languages, sample_rate 等
 */
char* voxcpm_model_info(const VoxCPMModel* model);

/*
 * 上傳模型權重到 GPU（需 VOXCPM_CUDA 編譯）。
 * 回傳 VOXCPM_ERR_CUDA_NOT_FOUND 若 GPU 不可用。
 * 成功後 voxcpm_generate 會使用 GPU 進行所有運算。
 */
VoxCPMError voxcpm_to_cuda(VoxCPMModel* model);

/* ═══════════════════════════════════════════════════════════════
 * Text-to-Speech Generation
 * ═══════════════════════════════════════════════════════════════ */

/*
 * 同步 TTS 生成
 * 生成完整的音訊後回傳。
 * 支援: TTS / Voice Design / Controllable Clone / Ultimate Clone
 * (依 config 中的欄位決定模式)
 */
VoxCPMError voxcpm_generate(
    VoxCPMModel*        model,
    const VoxCPMGenConfig* config,
    VoxCPMAudio*        output
);

/* ═══════════════════════════════════════════════════════════════
 * Streaming Generation
 * ═══════════════════════════════════════════════════════════════ */

/* 串流 callback:
 *   當新 chunk 可用時呼叫。
 *   chunk: PCM samples [-1.0, 1.0], num_samples: 本 chunk 的樣本數
 *   user_data: 呼叫者註冊的自訂資料指標
 *   回傳 true 繼續生成, false 取消。
 */
typedef bool (*voxcpm_stream_callback)(const float* chunk, int num_samples, void* user_data);

/*
 * 串流 TTS 生成
 * 每次生成一個 chunk 就呼叫 callback。
 * 適合即時播放或 pipe 輸出。
 */
VoxCPMError voxcpm_generate_streaming(
    VoxCPMModel*            model,
    const VoxCPMGenConfig*  config,
    voxcpm_stream_callback  callback,
    void*                   user_data
);

/* ═══════════════════════════════════════════════════════════════
 * Utility Functions
 * ═══════════════════════════════════════════════════════════════ */

/* 取得模型 RAM 使用量 (bytes, 不含權重 mmap) */
size_t voxcpm_memory_used(const VoxCPMModel* model);

/* 取得 GPU VRAM 使用量 (bytes, 僅 GPU 模式) */
size_t voxcpm_gpu_memory_used(const VoxCPMModel* model);

/* 取得 GPU 總 VRAM (bytes) */
size_t voxcpm_gpu_memory_total(const VoxCPMModel* model);

/* 設定生成逾時 (毫秒, 0 = 不限) */
void voxcpm_set_timeout(VoxCPMModel* model, int timeout_ms);

/* 取消當前生成 (從另一個執行緒呼叫) */
void voxcpm_cancel(VoxCPMModel* model);

/* ═══════════════════════════════════════════════════════════════
 * Backward Compatibility
 * ═══════════════════════════════════════════════════════════════ */

/* 簡化單次 TTS (一次呼叫完成所有動作) */
VoxCPMError voxcpm_tts(
    const char* model_path,
    const char* text,
    const char* output_path
);

#ifdef __cplusplus
}
#endif

#endif /* VOXCPM_H */

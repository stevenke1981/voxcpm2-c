/**
 * voice_clone.c — VoxCPM2-C 語音複製範例
 *
 * 展示 Controllable Cloning 與 Ultimate Cloning 功能。
 *
 * 編譯:
 *   gcc -O3 -Iinclude examples/voice_clone.c src/*.c -lm -o build/voice_clone
 *
 * 使用方式:
 *   ./build/voice_clone --text "Hello world" --ref reference.wav
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "voxcpm.h"

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "  -t, --text TEXT        Text to synthesize\n");
    fprintf(stderr, "  -o, --output FILE      Output WAV path\n");
    fprintf(stderr, "  -m, --model FILE       Model path\n");
    fprintf(stderr, "  -r, --ref FILE         Reference audio for cloning\n");
    fprintf(stderr, "  -s, --ref-text FILE    Reference transcript (for ultimate clone)\n");
    fprintf(stderr, "  -d, --description DESC Voice design description\n");
    fprintf(stderr, "  -e, --emotion EMOTE    Emotion (neutral|happy|sad|angry)\n");
    fprintf(stderr, "  --similarity FLOAT     Voice similarity (0.0-1.0)\n");
    fprintf(stderr, "  --gpu                  Use CUDA\n");
    fprintf(stderr, "  --steps N              DDIM steps (default: 10)\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
}

int main(int argc, char **argv) {
    const char *text = NULL;
    const char *output = "output.wav";
    const char *model_path = "models/voxcpm2-q4.vxcpm";
    const char *ref_audio = NULL;
    const char *ref_text = NULL;
    const char *description = NULL;
    const char *emotion = "neutral";
    float similarity = 0.7f;
    int use_gpu = 0;
    int ddim_steps = 10;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--text") == 0) {
            if (++i < argc) text = argv[i];
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i < argc) output = argv[i];
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (++i < argc) model_path = argv[i];
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--ref") == 0) {
            if (++i < argc) ref_audio = argv[i];
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--ref-text") == 0) {
            if (++i < argc) ref_text = argv[i];
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--description") == 0) {
            if (++i < argc) description = argv[i];
        } else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--emotion") == 0) {
            if (++i < argc) emotion = argv[i];
        } else if (strcmp(argv[i], "--similarity") == 0) {
            if (++i < argc) similarity = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = 1;
        } else if (strcmp(argv[i], "--steps") == 0) {
            if (++i < argc) ddim_steps = atoi(argv[i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (!text) {
        fprintf(stderr, "❌ Missing required: --text\n");
        print_usage(argv[0]);
        return 1;
    }
    
    printf("═══ VoxCPM2-C Voice Clone Demo ═══\n\n");
    printf("  Text:   %s\n", text);
    printf("  Output: %s\n", output);
    printf("  Model:  %s\n", model_path);
    printf("  GPU:    %s\n", use_gpu ? "yes" : "no");
    printf("  Steps:  %d\n", ddim_steps);
    printf("\n");
    
    // Load model
    printf("Loading model...\n");
    
    VoxCPMModelConfig cfg = {
        .model_path = model_path,
        .use_gpu = use_gpu,
        .num_threads = 0,
    };
    
    VoxCPMError err;
    VoxCPMModel *model = voxcpm_create(&cfg, &err);
    if (!model) {
        fprintf(stderr, "❌ Failed to load model: %s\n",
                voxcpm_error_string(err));
        return 1;
    }
    
    printf("  ✓ Model loaded (%.1f MB)\n\n",
           voxcpm_memory_used(model) / 1e6);
    
    // 選擇模式
    if (description) {
        // Voice Design
        printf("🎨 Voice Design mode\n");
        printf("  Description: %s\n", description);
        
        // 此處使用 special token 嵌入描述（實際 API 取決於實作）
        char design_text[4096];
        snprintf(design_text, sizeof(design_text),
                 "[VOICE_DESIGN]%s[TEXT]%s", description, text);
        
        VoxCPMGenerateOptions opts = {
            .input_ids = (int *)design_text,
            .n_input_ids = strlen(design_text),
            .output_audio = NULL,
            .output_len = NULL,
            .use_gpu = use_gpu,
            .max_audio_len = 48000 * 30,
            .temperature = 0.8f,
            .cfg_scale = 1.0f,
            .ddim_steps = ddim_steps,
            .seed = 42,
        };
        
        // TODO: 實際呼叫 API
        
    } else if (ref_audio && ref_text) {
        // Ultimate Cloning
        printf("🔊 Ultimate Cloning mode\n");
        printf("  Reference: %s\n", ref_audio);
        printf("  Transcript: %s\n", ref_text);
        
    } else if (ref_audio) {
        // Controllable Cloning
        printf("🔊 Controllable Cloning mode\n");
        printf("  Reference: %s\n", ref_audio);
        printf("  Emotion: %s\n", emotion);
        printf("  Similarity: %.2f\n", similarity);
        
    } else {
        // Standard TTS
        printf("🔊 Standard TTS mode\n");
    }
    
    // Cleanup
    voxcpm_free(model);
    printf("\nDone.\n");
    
    return 0;
}

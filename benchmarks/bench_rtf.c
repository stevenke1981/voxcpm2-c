// bench_rtf.c — Real-Time Factor benchmark for VoxCPM2-C
// VoxCPM2-C Project
//
// Measures end-to-end TTS generation speed and reports RTF:
//   RTF = generation_time_seconds / audio_duration_seconds
//
// RTF < 1.0 means faster than real-time (generates 1s of audio in < 1s).
// RTF ~1-5 is acceptable for interactive use.
// RTF > 10 is suitable for offline batch generation.
//
// Compile: cmake --build build --target bench_rtf
// Usage:   build/Debug/bench_rtf [model_path] [n_runs]

#include "voxcpm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

/* ─── High-resolution timer (Windows) ────────────────────────── */
static double now_seconds(void) {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}

/* ─── Test prompts ───────────────────────────────────────────── */
static const char* test_prompts[] = {
    /* short */
    "Hello.",
    /* medium */
    "The quick brown fox jumps over the lazy dog near the bank of the river.",
    /* Chinese */
    "今天天氣真好，適合出門走走。",
    /* longer with punctuation */
    "Welcome to VoxCPM2-C, a pure C inference engine for the VoxCPM2 text-to-speech model. "
    "This benchmark measures the real-time factor of the entire generation pipeline.",
};

#define NUM_PROMPTS ((int)(sizeof(test_prompts) / sizeof(test_prompts[0])))

/* ─── Benchmark one generation ───────────────────────────────── */
static int run_benchmark(VoxCPMModel* model, const char* text, int run,
                          double* out_gen_sec, double* out_audio_sec) {
    VoxCPMGenConfig gen_cfg = voxcpm_gen_config_default();
    gen_cfg.text            = text;
    gen_cfg.cfg_value       = 2.0f;
    gen_cfg.inference_timesteps = 10;

    VoxCPMAudio audio;
    memset(&audio, 0, sizeof(audio));

    double t0 = now_seconds();
    VoxCPMError err = voxcpm_generate(model, &gen_cfg, &audio);
    double t1 = now_seconds();
    *out_gen_sec = t1 - t0;

    if (err != VOXCPM_SUCCESS) {
        fprintf(stderr, "  [run %d] generation failed: %s\n",
                run, voxcpm_error_string(err));
        voxcpm_audio_free(&audio);
        return -1;
    }

    double duration = (double)audio.num_samples / (double)audio.sample_rate;
    *out_audio_sec = duration;

    voxcpm_audio_free(&audio);
    return 0;
}

/* ─── Main ───────────────────────────────────────────────────── */
int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "models/voxcpm2-f16.vxcpm";
    int n_runs             = argc > 2 ? atoi(argv[2]) : 3;

    /* Suppress INFO/DEBUG logs during benchmark */
    voxcpm_set_log_level(VOXCPM_LOG_WARN);

    printf("════════════════════════════════════════════════════\n");
    printf("  VoxCPM2-C RTF Benchmark\n");
    printf("  Model:  %s\n", model_path);
    printf("  Runs:   %d per prompt\n", n_runs);
    printf("════════════════════════════════════════════════════\n\n");

    /* ─── Create model ────────────────────────────────────── */
    VoxCPMError err;
    VoxCPMModelConfig cfg = voxcpm_model_config_default();
    cfg.model_path = model_path;
    cfg.n_threads  = 4;

    printf("Loading model...\n");
    double t_load0 = now_seconds();
    VoxCPMModel* model = voxcpm_create(&cfg, &err);
    double t_load1 = now_seconds();
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", voxcpm_error_string(err));
        return 1;
    }
    printf("  Model loaded in %.2f s\n\n", t_load1 - t_load0);

    /* ─── Warm-up run ─────────────────────────────────────── */
    printf("Warm-up (prompt: \"Hello.\")...\n");
    {
        double gen_s, aud_s;
        int ret = run_benchmark(model, "Hello.", 0, &gen_s, &aud_s);
        if (ret != 0) {
            voxcpm_free(model);
            return 1;
        }
        printf("  %.1f ms gen, %.1f ms audio (RTF = %.2f)\n\n",
               gen_s * 1000.0, aud_s * 1000.0, gen_s / aud_s);
    }

    /* ─── Timed runs ──────────────────────────────────────── */
    printf("%-12s %8s %8s %8s %8s\n",
           "PROMPT", "RUN", "GEN(s)", "AUDIO(s)", "RTF");
    printf("──────────────────────────────────────────────────\n");

    double overall_min_rtf = 1e9, overall_max_rtf = 0, overall_sum_rtf = 0;
    int    overall_count = 0;

    for (int p = 0; p < NUM_PROMPTS; p++) {
        const char* text = test_prompts[p];
        /* Truncate display name for table */
        char display[13];
        int text_len = (int)strlen(text);
        if (text_len > 12) {
            memcpy(display, text, 9);
            display[9] = '.';
            display[10] = '.';
            display[11] = '.';
            display[12] = '\0';
        } else {
            int i;
            for (i = 0; i < text_len; i++) display[i] = text[i];
            for (; i < 12; i++) display[i] = ' ';
            display[12] = '\0';
        }

        double prompt_min = 1e9, prompt_max = 0, prompt_sum = 0;

        for (int r = 0; r < n_runs; r++) {
            double gen_s, aud_s;
            int ret = run_benchmark(model, text, r, &gen_s, &aud_s);
            if (ret != 0) continue;

            double rtf = gen_s / aud_s;
            printf("%-12s %8d %8.1f %8.1f %8.2f\n",
                   r == 0 ? display : "",
                   r + 1, gen_s, aud_s * 1000.0, rtf);

            if (rtf < prompt_min) prompt_min = rtf;
            if (rtf > prompt_max) prompt_max = rtf;
            prompt_sum += rtf;
            if (rtf < overall_min_rtf) overall_min_rtf = rtf;
            if (rtf > overall_max_rtf) overall_max_rtf = rtf;
            overall_sum_rtf += rtf;
            overall_count++;
        }

        double prompt_avg = prompt_sum / n_runs;
        printf("%-12s %8s %8s %8s min=%.2f avg=%.2f max=%.2f\n",
               "", "", "", "", prompt_min, prompt_avg, prompt_max);
        printf("──────────────────────────────────────────────────\n");
    }

    /* ─── Summary ─────────────────────────────────────────── */
    if (overall_count > 0) {
        double overall_avg = overall_sum_rtf / overall_count;
        printf("\n════════════════════════════════════════════════════\n");
        printf("  OVERALL  runs=%d  min=%.2f  avg=%.2f  max=%.2f\n",
               overall_count, overall_min_rtf, overall_avg, overall_max_rtf);
        printf("════════════════════════════════════════════════════\n");
    }

    /* ─── Cleanup ─────────────────────────────────────────── */
    voxcpm_free(model);
    return 0;
}

// main.c — CLI entry point
// VoxCPM2-C Project
// License: Apache-2.0
//
// Command-line interface for VoxCPM2-C TTS engine.
//
// Usage:
//   voxcpm2-c [command] [options]
//
// Commands:
//   tts      Text-to-Speech (default)
//   clone    Voice cloning
//   design   Voice design
//   stream   Streaming mode
//   info     Model/system information

#include "voxcpm.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════ */

#define MAX_PATH 4096
#define MAX_TEXT 65536

/* ═══════════════════════════════════════════════════════════════
 * Command definitions
 * ═══════════════════════════════════════════════════════════════ */

typedef enum {
    CMD_TTS,
    CMD_CLONE,
    CMD_DESIGN,
    CMD_STREAM,
    CMD_INFO,
    CMD_HELP,
} Command;

typedef struct {
    Command     cmd;
    char        model_path[MAX_PATH];
    char        text[MAX_TEXT];
    char        output_path[MAX_PATH];
    char        reference_wav[MAX_PATH];
    char        prompt_text[MAX_TEXT];
    char        voice_design[MAX_TEXT];
    float       cfg_value;
    int         steps;
    int         threads;
    int         seed;
    bool        use_gpu;
    bool        verbose;
    bool        quiet;
    const char* quant;
} CLIOptions;

/* ═══════════════════════════════════════════════════════════════
 * Default options
 * ═══════════════════════════════════════════════════════════════ */

static CLIOptions default_options(void) {
    CLIOptions opts;
    memset(&opts, 0, sizeof(opts));
    opts.cmd = CMD_TTS;
    opts.cfg_value = 2.0f;
    opts.steps = 10;
    opts.threads = 4;
    opts.seed = 0;
    opts.use_gpu = false;
    opts.verbose = false;
    opts.quiet = false;
    opts.quant = NULL;
    strcpy(opts.model_path, "models/voxcpm2-f16.vxcpm");
    strcpy(opts.output_path, "output.wav");
    return opts;
}

/* ═══════════════════════════════════════════════════════════════
 * Argument parsing (simple getopt-style)
 * ═══════════════════════════════════════════════════════════════ */

static void print_usage(const char* prog) {
    printf("VoxCPM2-C v" VOXCPM_VERSION " — Pure C TTS Engine\n");
    printf("Usage: %s [command] [options]\n\n", prog);
    printf("Commands:\n");
    printf("  tts       Text-to-Speech (default)\n");
    printf("  clone     Voice cloning\n");
    printf("  design    Voice design\n");
    printf("  stream    Streaming mode\n");
    printf("  info      Model/system information\n");
    printf("  help      Show this help\n\n");
    printf("Options:\n");
    printf("  -m, --model PATH       Model weights path\n");
    printf("  -t, --text TEXT        Input text\n");
    printf("  -o, --output FILE      Output audio path\n");
    printf("  -r, --reference FILE   Reference audio (clone mode)\n");
    printf("  -p, --prompt-text TEXT Reference transcript (ultimate clone)\n");
    printf("  -d, --description TEXT Voice description (design mode)\n");
    printf("  --cfg FLOAT            CFG value (default: 2.0)\n");
    printf("  --steps INT            Diffusion steps (default: 10)\n");
    printf("  --cpu                  Force CPU mode\n");
    printf("  --gpu                  Force GPU mode\n");
    printf("  --threads INT          CPU threads (default: 4)\n");
    printf("  --quant TYPE           Quantization (q4/f16/f32)\n");
    printf("  --seed INT             Random seed (0 = random)\n");
    printf("  -v, --verbose          Verbose output\n");
    printf("  -q, --quiet            Quiet mode\n\n");
    printf("Examples:\n");
    printf("  %s tts -t \"Hello world\" -o hello.wav\n", prog);
    printf("  %s info\n", prog);
}

static int parse_args(int argc, char** argv, CLIOptions* opts) {
    if (!opts) return -1;
    *opts = default_options();

    int i = 1;
    // Parse command
    if (i < argc && argv[i][0] != '-') {
        if (strcmp(argv[i], "tts") == 0)       opts->cmd = CMD_TTS;
        else if (strcmp(argv[i], "clone") == 0) opts->cmd = CMD_CLONE;
        else if (strcmp(argv[i], "design") == 0)opts->cmd = CMD_DESIGN;
        else if (strcmp(argv[i], "stream") == 0)opts->cmd = CMD_STREAM;
        else if (strcmp(argv[i], "info") == 0)  opts->cmd = CMD_INFO;
        else if (strcmp(argv[i], "help") == 0)  { opts->cmd = CMD_HELP; return 0; }
        else {
            fprintf(stderr, "Unknown command: %s\n", argv[i]);
            return -1;
        }
        i++;
    }

    // Parse options
    while (i < argc) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (++i >= argc) return -1;
            strncpy(opts->model_path, argv[i], MAX_PATH - 1);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--text") == 0) {
            if (++i >= argc) return -1;
            strncpy(opts->text, argv[i], MAX_TEXT - 1);
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i >= argc) return -1;
            strncpy(opts->output_path, argv[i], MAX_PATH - 1);
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--reference") == 0) {
            if (++i >= argc) return -1;
            strncpy(opts->reference_wav, argv[i], MAX_PATH - 1);
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt-text") == 0) {
            if (++i >= argc) return -1;
            strncpy(opts->prompt_text, argv[i], MAX_TEXT - 1);
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--description") == 0) {
            if (++i >= argc) return -1;
            strncpy(opts->voice_design, argv[i], MAX_TEXT - 1);
        } else if (strcmp(argv[i], "--cfg") == 0) {
            if (++i >= argc) return -1;
            opts->cfg_value = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--steps") == 0) {
            if (++i >= argc) return -1;
            opts->steps = atoi(argv[i]);
        } else if (strcmp(argv[i], "--cpu") == 0) {
            opts->use_gpu = false;
        } else if (strcmp(argv[i], "--gpu") == 0) {
            opts->use_gpu = true;
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (++i >= argc) return -1;
            opts->threads = atoi(argv[i]);
        } else if (strcmp(argv[i], "--quant") == 0) {
            if (++i >= argc) return -1;
            opts->quant = argv[i];
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (++i >= argc) return -1;
            opts->seed = atoi(argv[i]);
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            opts->verbose = true;
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            opts->quiet = true;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return -1;
        }
        i++;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Command handlers (stubs for Phase 0)
 * ═══════════════════════════════════════════════════════════════ */

static int cmd_info(const CLIOptions* opts) {
    (void)opts;
    printf("VoxCPM2-C v" VOXCPM_VERSION "\n");
    printf("Backend: CPU");
#ifdef VOXCPM_CUDA
    printf(" + CUDA");
#endif
#ifdef VOXCPM_OPENMP
    printf(" + OpenMP");
#endif
    printf("\n");

    int gpu_count = voxcpm_gpu_count();
    printf("GPU devices: %d\n", gpu_count);
    for (int i = 0; i < gpu_count; i++) {
        char name[256];
        voxcpm_gpu_name(i, name, sizeof(name));
        printf("  [%d] %s\n", i, name);
    }

    printf("\nModel: not loaded (use -m to specify path)\n");
    return 0;
}

static int cmd_tts(const CLIOptions* opts) {
    if (opts->verbose) {
        printf("TTS mode\n");
        printf("  Text: %s\n", opts->text);
        printf("  Model: %s\n", opts->model_path);
        printf("  Output: %s\n", opts->output_path);
    }

    // Create model config
    VoxCPMModelConfig cfg = voxcpm_model_config_default();
    cfg.model_path = opts->model_path;
    cfg.n_threads = opts->threads;
    cfg.use_gpu = opts->use_gpu;

    // Create model
    VoxCPMError err;
    printf("Loading model...\n");
    VoxCPMModel* model = voxcpm_create(&cfg, &err);
    if (!model) {
        fprintf(stderr, "Failed to create model: %s\n", voxcpm_error_string(err));
        return 1;
    }

    // Show model info
    char* info = voxcpm_model_info(model);
    if (info) {
        printf("Model info: %s\n", info);
        free(info);
    }

    // Create generation config
    VoxCPMGenConfig gen_cfg = voxcpm_gen_config_default();
    gen_cfg.text = opts->text;
    gen_cfg.cfg_value = opts->cfg_value;
    gen_cfg.inference_timesteps = opts->steps;
    gen_cfg.seed = opts->seed;

    // Generate speech
    VoxCPMAudio audio;
    memset(&audio, 0, sizeof(audio));

    printf("Generating speech...\n");
    err = voxcpm_generate(model, &gen_cfg, &audio);
    if (err != VOXCPM_SUCCESS) {
        fprintf(stderr, "Generation failed: %s\n", voxcpm_error_string(err));
        voxcpm_free(model);
        return 1;
    }

    printf("Generated %d samples at %d Hz (%.1f seconds)\n",
           audio.num_samples, audio.sample_rate,
           (float)audio.num_samples / audio.sample_rate);

    // Save to WAV
    err = voxcpm_audio_save(&audio, opts->output_path);
    voxcpm_audio_free(&audio);
    voxcpm_free(model);

    if (err != VOXCPM_SUCCESS) {
        fprintf(stderr, "Failed to save audio: %s\n", voxcpm_error_string(err));
        return 1;
    }

    printf("Saved to: %s\n", opts->output_path);
    return 0;
}

static int cmd_clone(const CLIOptions* opts) {
    printf("Voice cloning will be available in Phase 3.\n");
    printf("Reference: %s\n", opts->reference_wav);
    if (opts->prompt_text[0])
        printf("Prompt text: %s\n", opts->prompt_text);
    return 0;
}

static int cmd_design(const CLIOptions* opts) {
    printf("Voice design will be available in Phase 3.\n");
    printf("Description: %s\n", opts->voice_design);
    printf("Text: %s\n", opts->text);
    return 0;
}

static int cmd_stream(const CLIOptions* opts) {
    printf("Streaming will be available in Phase 3.\n");
    printf("Text: %s\n", opts->text);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Main entry point
 * ═══════════════════════════════════════════════════════════════ */

int main(int argc, char** argv) {
    CLIOptions opts;

    if (parse_args(argc, argv, &opts) != 0) {
        print_usage(argv[0]);
        return 1;
    }

    // Set log level
    if (opts.quiet) {
        voxcpm_set_log_level(VOXCPM_LOG_QUIET);
    } else if (opts.verbose) {
        voxcpm_set_log_level(VOXCPM_LOG_DEBUG);
    }

    // Dispatch command
    int result = 0;
    switch (opts.cmd) {
        case CMD_TTS:    result = cmd_tts(&opts); break;
        case CMD_CLONE:  result = cmd_clone(&opts); break;
        case CMD_DESIGN: result = cmd_design(&opts); break;
        case CMD_STREAM: result = cmd_stream(&opts); break;
        case CMD_INFO:   result = cmd_info(&opts); break;
        case CMD_HELP:   print_usage(argv[0]); break;
        default:         print_usage(argv[0]); result = 1; break;
    }

    return result;
}

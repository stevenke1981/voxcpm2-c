// tts_minimal.c — Minimal TTS example
// VoxCPM2-C Project
#include "voxcpm.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "models/voxcpm2-f16.vxcpm";
    const char* text = argc > 2 ? argv[2] : "Hello, world!";
    const char* output = argc > 3 ? argv[3] : "output.wav";

    printf("VoxCPM2-C Minimal TTS Example\n");
    printf("  Model:  %s\n", model_path);
    printf("  Text:   %s\n", text);
    printf("  Output: %s\n\n", output);

    // Create model
    VoxCPMModelConfig cfg = voxcpm_model_config_default();
    cfg.model_path = model_path;
    cfg.n_threads = 4;

    VoxCPMError err;
    VoxCPMModel* model = voxcpm_create(&cfg, &err);
    if (!model) {
        fprintf(stderr, "Failed to create model: %s\n", voxcpm_error_string(err));
        return 1;
    }

    // Generate speech
    VoxCPMGenConfig gen_cfg = voxcpm_gen_config_default();
    gen_cfg.text = text;
    gen_cfg.cfg_value = 2.0f;
    gen_cfg.inference_timesteps = 10;

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
    err = voxcpm_audio_save(&audio, output);
    if (err != VOXCPM_SUCCESS) {
        fprintf(stderr, "Failed to save audio: %s\n", voxcpm_error_string(err));
    } else {
        printf("Saved to: %s\n", output);
    }

    fflush(stdout);
    voxcpm_audio_free(&audio);
    voxcpm_free(model);
    return 0;
}

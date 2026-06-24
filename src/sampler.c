// sampler.c — DDIM Sampler implementation (P2-09)
// VoxCPM2-C Project
// License: Apache-2.0
//
// Implements the DDIM (Denoising Diffusion Implicit Models) sampler
// with zero-SNR noise schedule support for the VoxCPM2 diffusion model.
#include "sampler.h"

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * Internal Helpers
 * ═══════════════════════════════════════════════════════════════ */

// Simple xorshift64 PRNG (same generator as tensor.c)
static inline uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

// Find the next (lower) timestep in the inference chain.
// Given the current timestep t_cur, searches step_indices for it and
// returns the next entry (the destination timestep). If t_cur is not
// found in step_indices, falls back to t_cur - 1 (consecutive steps).
// This correctly handles both skip-step fast inference and consecutive
// diffusion.
static int find_dest_timestep(const DDIMSampler* s, int t_cur) {
    if (t_cur <= 0) return 0;
    for (int i = 0; i < s->inference_steps - 1; i++) {
        if (s->step_indices[i] == t_cur) {
            int next = s->step_indices[i + 1];
            return (next >= 0) ? next : 0;
        }
    }
    return t_cur - 1;
}

/* ═══════════════════════════════════════════════════════════════
 * Noise Schedule
 * ═══════════════════════════════════════════════════════════════ */

int sampler_precompute_alphas(int timesteps, float beta_start, float beta_end,
                               float* alphas, float* alphas_cumprod) {
    if (timesteps <= 0 || !alphas || !alphas_cumprod)
        return -1;

    float beta_range = beta_end - beta_start;
    float denom = (float)(timesteps - 1);
    if (denom <= 0.0f) denom = 1.0f;

    for (int i = 0; i < timesteps; i++) {
        // Linear beta schedule
        float beta = beta_start + beta_range * (float)i / denom;
        alphas[i] = 1.0f - beta;

        // Cumulative product of alphas
        if (i == 0) {
            alphas_cumprod[i] = alphas[i];
        } else {
            alphas_cumprod[i] = alphas_cumprod[i - 1] * alphas[i];
        }
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * DDIM Sampler Lifecycle
 * ═══════════════════════════════════════════════════════════════ */

DDIMSampler* ddim_sampler_create(int timesteps, int inference_steps,
                                  float eta, float cfg_scale) {
    if (timesteps <= 0 || inference_steps <= 0 || inference_steps > timesteps)
        return NULL;

    DDIMSampler* sampler = (DDIMSampler*)calloc(1, sizeof(DDIMSampler));
    if (!sampler) return NULL;

    sampler->timesteps       = timesteps;
    sampler->inference_steps = inference_steps;
    sampler->eta             = eta;
    sampler->cfg_scale       = cfg_scale;
    sampler->use_cfg         = (cfg_scale > 1.0001f);

    // Allocate arrays
    sampler->alphas         = (float*)calloc((size_t)timesteps, sizeof(float));
    sampler->alphas_cumprod = (float*)calloc((size_t)timesteps, sizeof(float));
    sampler->sigmas         = (float*)calloc((size_t)(timesteps + 1), sizeof(float));
    sampler->step_indices   = (int*)calloc((size_t)inference_steps, sizeof(int));

    if (!sampler->alphas || !sampler->alphas_cumprod ||
        !sampler->sigmas || !sampler->step_indices) {
        ddim_sampler_free(sampler);
        return NULL;
    }

    // Precompute alpha schedule
    if (sampler_precompute_alphas(timesteps, 0.0001f, 0.02f,
                                   sampler->alphas,
                                   sampler->alphas_cumprod) != 0) {
        ddim_sampler_free(sampler);
        return NULL;
    }

    // Precompute sigmas: sigma[t] = sqrt(1 - alpha_cumprod[t])
    // sigmas[timesteps] = 1.0 (max noise level)
    for (int i = 0; i < timesteps; i++) {
        float val = 1.0f - sampler->alphas_cumprod[i];
        sampler->sigmas[i] = (val > 0.0f) ? sqrtf(val) : 0.0f;
    }
    sampler->sigmas[timesteps] = 1.0f;

    // Compute inference step indices: evenly spaced from timesteps-1 down to 0.
    // For 1000 total timesteps and 10 inference steps, this produces:
    // [999, 888, 777, 666, 555, 444, 333, 222, 111, 0]
    if (inference_steps == 1) {
        sampler->step_indices[0] = timesteps - 1;
    } else {
        float step_size = (float)(timesteps - 1) / (float)(inference_steps - 1);
        for (int i = 0; i < inference_steps; i++) {
            int idx = (int)(step_size * (float)(inference_steps - 1 - i) + 0.5f);
            if (idx >= timesteps) idx = timesteps - 1;
            if (idx < 0)          idx = 0;
            sampler->step_indices[i] = idx;
        }
    }

    return sampler;
}

void ddim_sampler_free(DDIMSampler* sampler) {
    if (!sampler) return;
    free(sampler->alphas);
    free(sampler->alphas_cumprod);
    free(sampler->sigmas);
    free(sampler->step_indices);
    free(sampler);
}

/* ═══════════════════════════════════════════════════════════════
 * DDIM Denoising Step
 * ═══════════════════════════════════════════════════════════════
 *
 * Performs one DDIM denoising step from timestep t to the next lower
 * timestep in the inference chain (t_prev).
 *
 * If uncond_pred is provided, classifier-free guidance is applied:
 *   pred = uncond_pred + cfg_scale * (pred - uncond_pred)
 *
 * DDIM update formula (epsilon-prediction):
 *   x_0_pred = (x_t - sqrt(1 - acp_t) * pred) / sqrt(acp_t)
 *   sigma    = eta * sqrt((1 - acp_prev) / (1 - acp_t))
 *                     * sqrt(1 - acp_t / acp_prev)
 *   noise    = N(0, I) if eta > 0, else 0
 *   x_prev   = sqrt(acp_prev) * x_0_pred
 *            + sqrt(1 - acp_prev - sigma^2) * pred
 *            + sigma * noise
 * ═══════════════════════════════════════════════════════════════ */

VoxCPMError ddim_step(const DDIMSampler* sampler, const Tensor* x_t,
                       const Tensor* pred, int t,
                       const Tensor* uncond_pred, Tensor* out) {
    if (!sampler || !x_t || !pred || !out)
        return VOXCPM_ERR_INTERNAL;
    if (sampler->timesteps <= 0)
        return VOXCPM_ERR_INTERNAL;
    if (t < 0 || t >= sampler->timesteps)
        return VOXCPM_ERR_INTERNAL;

    size_t n = x_t->size;
    if (n != pred->size || n != out->size)
        return VOXCPM_ERR_SHAPE_MISMATCH;
    if (uncond_pred && uncond_pred->size != n)
        return VOXCPM_ERR_SHAPE_MISMATCH;
    if (n == 0)
        return VOXCPM_SUCCESS;

    // ── Determine destination timestep ──
    // Look up t in step_indices to find the next timestep in the chain.
    // Falls back to t-1 for consecutive-step usage.
    int t_prev = find_dest_timestep(sampler, t);

    // ── Precompute scalar values ──
    float acp_t     = sampler->alphas_cumprod[t];
    float acp_prev  = (t_prev > 0) ? sampler->alphas_cumprod[t_prev] : 1.0f;

    // Clamp for numerical stability
    if (acp_t < 1e-10f)     acp_t = 1e-10f;
    if (acp_prev < 1e-10f)  acp_prev = 1e-10f;

    float sqrt_acp_t     = sqrtf(acp_t);
    float sqrt_one_m_acp = sqrtf(1.0f - acp_t);
    float sqrt_acp_prev  = sqrtf(acp_prev);

    // ── Compute sigma (stochastic component) ──
    float sigma = 0.0f;
    if (sampler->eta > 0.0f && t_prev > 0) {
        float one_minus_acp     = 1.0f - acp_t;
        float one_minus_acp_prev = 1.0f - acp_prev;
        if (one_minus_acp > 1e-10f && acp_prev > 1e-10f) {
            float ratio      = one_minus_acp_prev / one_minus_acp;
            float correction = 1.0f - acp_t / acp_prev;
            if (correction < 0.0f) correction = 0.0f;
            sigma = sampler->eta * sqrtf(ratio) * sqrtf(correction);
        }
    }

    float sqrt_one_m_acp_prev_m_sigma2 =
        sqrtf(fmaxf(1.0f - acp_prev - sigma * sigma, 0.0f));

    // ── Generate stochastic noise if needed ──
    float* noise_buf = NULL;
    if (sigma > 1e-10f) {
        noise_buf = (float*)malloc(n * sizeof(float));
        if (!noise_buf) return VOXCPM_ERR_OOM;

        // Use deterministic seed for reproducibility of the noise
        // within this step (each step gets different noise due to
        // different preceding data). eta=0 (deterministic DDIM) skips
        // this allocation entirely.
        uint64_t seed = (uint64_t)(t + 1) * 2654435761ULL;
        for (size_t i = 0; i + 1 < n; i += 2) {
            double u1 = (double)xorshift64(&seed) / (double)UINT64_MAX;
            double u2 = (double)xorshift64(&seed) / (double)UINT64_MAX;
            if (u1 < 1e-15) u1 = 1e-15;
            double r     = sqrt(-2.0 * log(u1));
            double theta = 2.0 * M_PI * u2;
            noise_buf[i] = (float)(r * cos(theta));
            if (i + 1 < n)
                noise_buf[i + 1] = (float)(r * sin(theta));
        }
    }

    // ── Main DDIM update ──
    for (size_t i = 0; i < n; i++) {
        // Apply CFG if unconditional prediction is provided
        float eps;
        if (uncond_pred) {
            float u_cond = uncond_pred->data[i];
            float c_cond = pred->data[i];
            eps = u_cond + sampler->cfg_scale * (c_cond - u_cond);
        } else {
            eps = pred->data[i];
        }

        // Predict x_0 from noisy x_t and model output
        float x0_pred = (x_t->data[i] - sqrt_one_m_acp * eps) / sqrt_acp_t;

        // DDIM update: mix predicted x_0 and noise direction
        float x_prev = sqrt_acp_prev * x0_pred
                     + sqrt_one_m_acp_prev_m_sigma2 * eps;
        if (noise_buf) {
            x_prev += sigma * noise_buf[i];
        }

        out->data[i] = x_prev;
    }

    free(noise_buf);
    return VOXCPM_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * Sampler Convenience: Noise Generation
 * ═══════════════════════════════════════════════════════════════
 *
 * Fills the tensor with standard normal N(0,1) random values
 * using the xorshift64 PRNG and Box-Muller transform.
 * If seed is 0, a default seed is used.
 * ═══════════════════════════════════════════════════════════════ */

void sampler_generate_noise(Tensor* noise, uint64_t seed) {
    if (!noise || !noise->data || noise->size == 0) return;

    uint64_t state = (seed != 0) ? seed : 123456789ULL;

    // Box-Muller transform (same algorithm as tensor_rand_normal)
    for (size_t i = 0; i + 1 < noise->size; i += 2) {
        double u1 = (double)xorshift64(&state) / (double)UINT64_MAX;
        double u2 = (double)xorshift64(&state) / (double)UINT64_MAX;
        if (u1 < 1e-15) u1 = 1e-15;
        double r     = sqrt(-2.0 * log(u1));
        double theta = 2.0 * M_PI * u2;
        noise->data[i] = (float)(r * cos(theta));
        if (i + 1 < noise->size)
            noise->data[i + 1] = (float)(r * sin(theta));
    }
}

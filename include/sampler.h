#ifndef SAMPLER_H
#define SAMPLER_H

/*
 * sampler.h — Diffusion sampler for LocDiT
 * VoxCPM2-C Project
 * License: Apache-2.0
 *
 * DDIM and PNDM samplers for the diffusion denoising process.
 * Uses zero-SNR noise schedule as described in the VoxCPM2 paper.
 */

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 * Noise Schedule
 * ═══════════════════════════════════════════════════════════════ */

/* Precompute alpha values for the noise schedule.
 * alphas: [timesteps] array of alpha values at each step.
 * alphas_cumprod: [timesteps] array of cumulative alpha products.
 * Returns 0 on success, -1 on error. */
int sampler_precompute_alphas(
    int timesteps,
    float beta_start,      /* default: 0.0001 */
    float beta_end,        /* default: 0.02 */
    float* alphas,         /* [timesteps] output */
    float* alphas_cumprod  /* [timesteps] output */
);

/* ═══════════════════════════════════════════════════════════════
 * DDIM Sampler
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int     timesteps;           /* total diffusion steps */
    int     inference_steps;     /* subset of steps for fast inference */
    float   eta;                 /* DDIM stochastic factor (0 = deterministic) */
    bool    use_cfg;             /* enable classifier-free guidance */
    float   cfg_scale;           /* CFG scale (default: 2.0) */
    float*  alphas;              /* precomputed alpha schedule */
    float*  alphas_cumprod;      /* precomputed cumulative product */
    float*  sigmas;              /* noise schedule sigma */
    int*    step_indices;        /* indices of inference steps */
} DDIMSampler;

/* Create a DDIM sampler.
 * Returns NULL on OOM. */
DDIMSampler* ddim_sampler_create(
    int timesteps,
    int inference_steps,
    float eta,
    float cfg_scale
);

/* Free the sampler. Safe to call with NULL. */
void ddim_sampler_free(DDIMSampler* sampler);

/* Perform one DDIM denoising step.
 *   x_t:       current noisy latent [batch, latent_dim]
 *   pred:      model prediction (epsilon or x0) [batch, latent_dim]
 *   t:         current timestep index
 *   uncond_pred: unconditional prediction (for CFG, NULL if not used)
 *   out:       denoised latent for next step [batch, latent_dim]
 */
VoxCPMError ddim_step(
    const DDIMSampler* sampler,
    const Tensor* x_t,
    const Tensor* pred,
    int t,
    const Tensor* uncond_pred,
    Tensor* out
);

/* ═══════════════════════════════════════════════════════════════
 * Sampler convenience
 * ═══════════════════════════════════════════════════════════════ */

/* Generate random noise for the initial latent.
 *   shape: [batch, latent_dim]
 *   seed: random seed (0 = use default seed)
 */
void sampler_generate_noise(Tensor* noise, uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* SAMPLER_H */

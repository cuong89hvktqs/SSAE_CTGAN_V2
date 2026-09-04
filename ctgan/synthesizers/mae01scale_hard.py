"""MAE01ScaleHard — soft boundary-band hard-negative mining on CTGAN samples.

Motivation
----------
MAE01Scale/MAE01ScaleMax turn CTGAN's generated rows into pseudo-anomalies by
adding noise to them (type_train="std"/"noise_std"), uniformly over whatever
CTGAN happens to sample. This wastes training signal on two extremes:
  - CTGAN rows that already look exactly like normal data -> perturbing them
    slightly and labeling them "anomaly" injects label noise (a real normal
    point mislabeled as abnormal), which corrupts the margin instead of
    sharpening it.
  - CTGAN rows that are already far from normal -> the model already
    separates these easily; spending budget on them teaches nothing new.

The useful region is the *boundary band* in between: CTGAN rows realistic
enough to be a hard negative, but not so realistic they duplicate normal.

Where "the boundary" lives
---------------------------
The two-sided margin loss already defines this geometrically for free:
    normal   -> ||z|| -> 0        (pulled to the origin)
    anomaly  -> ||z|| -> margin   (pulled to a sphere of radius `margin`)
So "distance to normal" is just ||z||, and the boundary band is the annulus
    eff_floor <= ||z_fake|| < ceiling
around the origin, with:
    floor    = floor_k * mean(||z_normal||)      -- current normal radius;
               CTGAN rows inside this radius are statistically
               indistinguishable from real normal -> must not be mined.
    ceiling  = ceiling_frac * margin              -- CTGAN rows already near
               the anomaly target radius are "easy" -> not worth mining.
    tau      = floor + tau_pos * (ceiling - floor)  -- peak of the band
               (default: midpoint).

Soft weighting instead of a hard keep/drop cutoff
--------------------------------------------------
A boolean filter at floor/ceiling is unstable: as z drifts slightly across
steps, a borderline row flips between "fully used" and "fully discarded",
and hyperparameter choice becomes brittle. Instead we score every CTGAN row
with a smooth *bump* weight w(d) in [0, 1]:
  - a raised-cosine (Hann-style) ramp 0->1 from eff_floor to tau,
  - a raised-cosine ramp 1->0 from tau to ceiling,
  - EXACTLY 0 outside [eff_floor, ceiling] (compact support -- unlike a
    Gaussian's infinite tail, this gives a real guarantee that near-duplicate
    rows get zero weight, not just "very small").
The ramp is asymmetric: `floor_sharpness > 1` shrinks the up-ramp into a
region closer to tau, creating a wider true-zero dead zone just above floor.
This reflects that erring "too close" (label noise) is worse than erring
"too far" (wasted but harmless), so the floor side is deliberately stricter.

These weights are then used to *resample* (multinomial, with replacement)
the CTGAN batch -- rows near tau get drawn often, near-duplicate/too-easy
rows rarely or never, but the pipeline never starves (falls back to uniform
weights if the whole batch happens to score ~0).

No extra noise on mined CTGAN rows
-----------------------------------
MAE01ScaleMax adds Gaussian/categorical noise on top of *every* pseudo-
anomaly, including CTGAN rows. Once a CTGAN row has been selected as a
boundary-band sample, it is already a realistic, hard, near-boundary
example by construction -- adding noise on top would blur exactly the
signal the mining just selected for (and could push it back out of the
band). So for MAE01ScaleHard, the CTGAN-derived branch is fed straight
into M01AE training as-is, unperturbed. Noise (`_add_noise_transformed`)
is still applied, unchanged, to the *other* branch -- pseudo-anomalies
built from real normal samples (type_train="noise", and the "noise" half
of "noise_std"). This only takes effect when hard_mining=True; with
hard_mining=False this class reproduces MAE01ScaleMax exactly (ablation).

Net +6 params vs MAE01ScaleMax: hard_mining, floor_k, ceiling_frac, tau_pos,
floor_sharpness, min_gap_frac. Only `_make_pseudo_anomaly` is overridden --
everything else (gate, z-norm noise scaling, training loop) is inherited
unchanged from MAE01ScaleMax.
"""

import math

import torch

from ctgan.synthesizers.mae01scale_max import MAE01ScaleMax


class MAE01ScaleHard(MAE01ScaleMax):
    """MAE01ScaleMax + soft boundary-band hard-negative mining on CTGAN rows."""

    def __init__(
        self,
        # All MAE01ScaleMax params
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        cuda=True,
        lamda=0.1,
        margin=1.0,
        mae_loss_mode="renormal",
        mae_lr=1e-3,
        noise_std=0.3,
        noise_ratio=1.0,
        noise_type="global",
        noise_r=1.0,
        noise_k=3.0,
        cat_noise_type="none",
        cat_noise_ratio=1.0,
        type_train="noise_std",
        mae_start_ratio=0.0,
        patience=50,
        dataset="",
        path="",
        noise_scale=1.5,
        gate_stability=0.05,
        gate_window=10,
        cov_floor=0.05,
        eps_zero=0.1,
        tol=0.08,
        # ── Boundary-band hard-negative mining ─────────────────────────────
        hard_mining=True,       # False -> identical to MAE01ScaleMax (ablation)
        floor_k=1.0,            # floor = floor_k * mean(||z_normal||)
        ceiling_frac=0.7,       # ceiling = ceiling_frac * margin
        tau_pos=0.5,            # peak position in [floor, ceiling], 0.5 = midpoint
        floor_sharpness=1.5,    # >=1; higher = wider zero dead-zone above floor
        min_gap_frac=0.05,      # min (ceiling - floor) as a fraction of margin
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            generator_decay=generator_decay,
            discriminator_lr=discriminator_lr,
            discriminator_decay=discriminator_decay,
            batch_size=batch_size,
            discriminator_steps=discriminator_steps,
            log_frequency=log_frequency,
            verbose=verbose,
            epochs=epochs,
            pac=pac,
            cuda=cuda,
            lamda=lamda,
            margin=margin,
            mae_loss_mode=mae_loss_mode,
            mae_lr=mae_lr,
            noise_std=noise_std,
            noise_ratio=noise_ratio,
            noise_type=noise_type,
            noise_r=noise_r,
            noise_k=noise_k,
            cat_noise_type=cat_noise_type,
            cat_noise_ratio=cat_noise_ratio,
            type_train=type_train,
            mae_start_ratio=mae_start_ratio,
            patience=patience,
            dataset=dataset,
            path=path,
            noise_scale=noise_scale,
            gate_stability=gate_stability,
            gate_window=gate_window,
            cov_floor=cov_floor,
            eps_zero=eps_zero,
            tol=tol,
        )
        self.hard_mining     = bool(hard_mining)
        self.floor_k         = float(floor_k)
        self.ceiling_frac    = float(ceiling_frac)
        self.tau_pos         = min(max(float(tau_pos), 0.0), 1.0)
        self.floor_sharpness = max(float(floor_sharpness), 1.0)
        self.min_gap_frac    = float(min_gap_frac)
        self._hard_mining_logged = False

    # =========================================================================
    # Override: boundary-band resampling on the CTGAN branch, fed in as-is
    # (no added noise -- see module docstring). The real-normal-derived
    # branch still goes through the unchanged M01AECTGAN._add_noise_transformed.
    # =========================================================================

    def _make_pseudo_anomaly(self, real_normal, ctgan_sample, use_ctgan):
        if not (use_ctgan and self.hard_mining and ctgan_sample is not None):
            # hard_mining=False (ablation) or no CTGAN this step ->
            # exact MAE01ScaleMax behavior (noise added to both branches).
            return super()._make_pseudo_anomaly(real_normal, ctgan_sample, use_ctgan)

        ctgan_sample = self._hard_mining_resample(real_normal, ctgan_sample)

        if self.type_train == "std":
            # Mined CTGAN rows are already realistic, near-boundary
            # pseudo-anomalies -- use them as-is, no extra noise.
            return ctgan_sample

        # noise_std: 50% noise-derived (real_normal + noise), 50% CTGAN-derived (as-is)
        half = real_normal.size(0) // 2
        pseudo_noise = self._add_noise_transformed(real_normal[:half])
        pseudo_ctgan = ctgan_sample[:half]
        return torch.cat([pseudo_noise, pseudo_ctgan], dim=0)

    def _hard_mining_resample(self, real_normal, ctgan_sample):
        """Score CTGAN rows by ||z|| and resample toward the boundary band."""
        with torch.no_grad():
            z_normal = self._mae_enc(real_normal)
            z_fake   = self._mae_enc(ctgan_sample)

            r_normal = torch.norm(z_normal, p=2, dim=1)
            d_fake   = torch.norm(z_fake,   p=2, dim=1)

            floor_radius = max(self.floor_k * r_normal.mean().item(), 1e-6)
            min_gap      = self.min_gap_frac * self.margin
            ceiling_radius = max(self.ceiling_frac * self.margin, floor_radius + min_gap)
            tau = floor_radius + self.tau_pos * (ceiling_radius - floor_radius)
            eff_floor = tau - (tau - floor_radius) / self.floor_sharpness

            w = self._bump_weight(d_fake, eff_floor, tau, ceiling_radius)

            if not self._hard_mining_logged and self._ctgan_gate_open:
                frac_kept = (w > 1e-6).float().mean().item()
                print(
                    f"  [HARD-MINING] band: floor={floor_radius:.4f} "
                    f"eff_floor={eff_floor:.4f} tau={tau:.4f} "
                    f"ceiling={ceiling_radius:.4f} | "
                    f"frac_nonzero={frac_kept:.2%} mean_w={w.mean().item():.4f}"
                )
                self._hard_mining_logged = True

            if w.sum().item() < 1e-8:      # whole batch scored ~0 -> fall back to uniform
                w = torch.ones_like(w)

            idx = torch.multinomial(w, num_samples=ctgan_sample.size(0), replacement=True)
            return ctgan_sample[idx]

    @staticmethod
    def _bump_weight(d, eff_floor, tau, ceiling):
        """Asymmetric raised-cosine bump: 0 outside [eff_floor, ceiling], peak at tau."""
        w = torch.zeros_like(d)
        up_span   = max(tau - eff_floor, 1e-8)
        down_span = max(ceiling - tau, 1e-8)

        up_mask = (d > eff_floor) & (d < tau)
        w = torch.where(
            up_mask,
            0.5 * (1 - torch.cos(math.pi * (d - eff_floor) / up_span)),
            w,
        )
        down_mask = (d >= tau) & (d < ceiling)
        w = torch.where(
            down_mask,
            0.5 * (1 + torch.cos(math.pi * (d - tau) / down_span)),
            w,
        )
        w = torch.where(d == tau, torch.ones_like(d), w)
        return w

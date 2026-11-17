"""Control B (Reviewer 4, Comment 2b): same full MAE01ScaleMax pipeline
(CTGAN training, activation gate, adaptive noise scaling, radial-margin
MiAE) -- the ONLY change is that the structure-aware pseudo-intrusion
perturbation (_add_noise_transformed: a selected FRACTION of continuous
dims perturbed, one-hot categorical spans separately shuffled/flipped) is
replaced by unstructured isotropic Gaussian noise added to EVERY dimension
(continuous and categorical alike), at the SAME magnitude (self.noise_std)
the structured method would have used that epoch -- so "matched magnitude"
is automatic: whatever adaptive scale SSAE would apply, Control B applies
the identical scalar, just isotropically instead of selectively/structurally.

If Control A (train_control_a.py) asks "does compactness alone explain the
gain", Control B asks "does ANY out-of-class noise suffice, or does the
noise have to respect the data's continuous/categorical structure (P/Q,
fraction-based selection) the way SSAE's does".

Nothing else is overridden: CTGAN quality gate, type_train (P/Q mixing),
adaptive sigma, the margin loss, and checkpoint selection are all identical
to MAE01ScaleMax -- only the shape of the noise itself differs.
"""

import torch

from ctgan.synthesizers.mae01scale_max import MAE01ScaleMax


class MAE01ScaleMaxControlB(MAE01ScaleMax):
    """MAE01ScaleMax with isotropic (unstructured) Gaussian pseudo-intrusion
    noise instead of the structure-aware P/Q perturbation operator."""

    def _add_noise_transformed(self, x):
        # Every dimension gets i.i.d. N(0, noise_std^2), clamped to the same
        # tanh-space range [-1, 1] the structured method also clamps to.
        # No continuous/categorical distinction, no noise_ratio /
        # cat_noise_ratio fraction, no shuffle/flip -- this IS the control.
        noise = torch.randn_like(x) * self.noise_std
        return torch.clamp(x + noise, -1.0, 1.0)

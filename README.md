# SSAE — Self-Supervised Autoencoder

A research fork of [SDV](https://github.com/sdv-dev/SDV) / [CTGAN](https://github.com/sdv-dev/CTGAN) built around **SSAE** (implemented as `MAE01ScaleMax`): a network-intrusion anomaly detector that jointly trains a CTGAN generator (for data diversity) with a two-sided margin autoencoder, learning to separate normal traffic from pseudo-anomalies in latent space.

This README covers **only SSAE** — how to install the project and train/evaluate the SSAE model. The repository also contains several baseline comparisons (CLAN, DROCC, DeepOD, a plain autoencoder) used for the accompanying paper; they are not documented here.

## How it works

SSAE trains two components together, end to end, every epoch:

1. **CTGAN** (generator + discriminator) — learns to synthesize realistic traffic samples, giving the model a source of *diverse*, data-driven pseudo-anomalies once training has progressed far enough.
2. **A two-sided margin autoencoder (MAE)** — learns to reconstruct *normal* traffic well while keeping perturbed/synthetic samples ("pseudo-anomalies") outside a latent-space margin. At inference time, the anomaly score combines reconstruction error and latent-space margin violation.

Pseudo-anomalies come from two sources that are blended during training:
- **P — perturbation-based**: Gaussian noise injected into real normal samples (a self-scaling amount, proportional to the mean latent norm of normal samples).
- **Q — CTGAN-generated**: samples drawn from the CTGAN generator once it is deemed reliable enough.

A **gate** decides when CTGAN-generated samples (Q) are trustworthy enough to start using — it opens automatically once the discriminator loss has stabilized (adaptively, via a coefficient-of-variation check over a sliding window), so early, unreliable GAN samples never contaminate training. The gate can also be forced open at a fixed epoch instead, for ablation studies (see [Fixed-epoch gate](#fixed-epoch-gate-ablation) below).

## Installation

Requires Python 3.9–3.14.

```bash
pip install -e .[dev]
```

Installing in editable mode is required so the local `ctgan/` and `sdv/` packages (which contain the SSAE-specific code) take precedence over any published versions of those packages.

## Data setup

SSAE ships with loaders for several public network-intrusion-detection datasets. Place the raw CSVs under `Syn_ctgan/data/<dataset>/` (see `Syn_ctgan/process_data.py` for the exact expected filenames per dataset):

| `--d` value | Dataset |
|---|---|
| `unsw` | UNSW-NB15 |
| `nsl_kdd` | NSL-KDD |
| `cic_iot` | CIC-IoT 2023 |
| `wsn_ds` | WSN-DS |

Each loader returns the dataset's official train/test split; SSAE further splits the normal-only rows of the training pool 80/20 into train/validation.

## Training SSAE

All commands below are run from inside `Syn_ctgan/`.

### Single run

```bash
cd Syn_ctgan
python train_mae01scale_max.py --d unsw --seed 42 --time 4 \
    --e 150 --b 128 --lae 0.1 --mae 2.0 \
    --noise_std 0.3 --noise_scale 1.5 \
    --gate_stability 0.05 --gate_window 10 --cov_floor 0.05
```


### Key arguments

| Flag | Default | Meaning |
|---|---|---|
| `--d` | — | Dataset name (see table above) |
| `--seed` | 42 | Random seed |
| `--per` | 1.0 | Fraction of normal-only training data to use |
| `--e` | 150 | Training epochs |
| `--b` | 128 | Batch size |
| `--mae` | 2.0 | Margin used by the two-sided margin loss |
| `--lae` | 0.1 | Weight (λ) of the latent-margin term in the total anomaly score |
| `--type_train` | `noise_std` | Pseudo-anomaly source: `noise` (P only), `std` (Q only), `noise_std` (P+Q, the full model) |
| `--noise_std` | 0.3 | Base perturbation strength for P |
| `--noise_scale` | 1.5 | Self-scaling multiplier: `noise_std_effective = noise_scale × mean‖z_normal‖` |
| `--gate_window` | 10 | Sliding-window size (epochs) used to judge discriminator-loss stability |
| `--gate_stability` | 0.05 | Coefficient-of-variation threshold for the gate to open |
| `--cov_floor` | 0.05 | Minimum denominator in the CoV computation (avoids division-by-zero on datasets where the discriminator loss converges near 0) |
| `--patience` | 50 | Early-stopping patience (epochs without validation-loss improvement) |

Run `python train_mae01scale_max.py --help` for the complete list (noise/categorical-noise controls, clustering, snapshotting, etc.).

### Fixed-epoch gate (ablation)

To compare the adaptive gate against simply opening the CTGAN source at a fixed epoch (e.g. epoch 0, 30, or 50), pass `--gate_fixed_epoch N`. This bypasses the adaptive coefficient-of-variation check entirely:

```bash
python train_mae01scale_max.py --d unsw --seed 42 --time 6 --gate_fixed_epoch 50 [...]
```

A ready-made sweep across gate epochs and seeds is provided:

```bash
bash run_gatefix.sh [per]      # Linux/macOS
run_gatefix.bat [per]          # Windows
```

## Output layout

Each run writes to `Syn_ctgan/res/mae01scale_max_se_<seed>_t_<time>/per_<per>/<dataset>_.../`:

```
checkpoints/
  MAE_<dataset>_best.pkl          # best checkpoint by validation loss
  MAE_<dataset>_best_auc.pkl      # best checkpoint by validation AUC (used for evaluation)
  best_loss/   test_mode_*.csv    # metrics for the best-loss checkpoint
  best_auc/    test_mode_*.csv    # metrics for the best-auc checkpoint
<dataset>_X_test.csv, _y_test.csv, _X_val_normal.csv   # preprocessed splits, for later re-evaluation
```

Each `test_mode_*.csv` reports AUC, F1, MCC, Precision, Recall, and Accuracy at a specific scoring/thresholding convention (reconstruction-only, latent-only, or combined "total" score; thresholded at either a fixed percentile or a distribution-based cutoff).

**Which result is the official one**: `checkpoints/best_auc/test_mode_total95.csv` is SSAE's primary result — the combined reconstruction + latent-margin ("total") score, thresholded at the 95th percentile of validation-normal scores, evaluated on the best-by-validation-AUC checkpoint. Of that file, only the **AUC** value is reported as-is (AUC is threshold-independent, so it is unaffected by the 95th-percentile choice below); its F1/MCC/Precision/Recall are tied to that specific 95th-percentile threshold and are not used for the metrics below.

This training script does **not** compute TPR@FAR=1%/5% or an MCC at a 99th-percentile threshold — those need a separate evaluation pass over the already-trained checkpoint (no retraining):
- **MCC at a 99th-percentile threshold**: `python eval_ssae_val99.py --dataset <d> --seed <s> --per <per>` — reloads the checkpoint and recomputes F1/MCC/etc. at the 99th percentile of validation-normal scores, saving `checkpoints/best_auc/test_mode_total99.csv`.
- **TPR@FAR=1% / TPR@FAR=5%**: `python val_perattack_mae01scale_max.py --d <d>` — per-attack breakdown, with each threshold set from a percentile of the test-normal score distribution (a different, FAR-based convention, independent of the 95%/99% validation threshold above).

## License

This project builds on SDV/CTGAN; see `LICENSE` for terms.

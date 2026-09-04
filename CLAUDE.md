# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research fork of SDV/CTGAN extended with custom anomaly detection synthesizers. The core contribution is a family of **M01AE** models that jointly train CTGAN (for data diversity) with a two-sided margin autoencoder (M01AE) to separate normal from pseudo-anomaly embeddings in latent space. All training scripts live in `Syn_ctgan/`.

## Commands

```bash
# Install in editable mode (required for local ctgan/ and sdv/ to take effect)
pip install -e .[dev]

# Lint
make lint          # runs ruff via invoke
make fix-lint      # auto-fix

# Tests
make test-unit           # invoke unit
make test-integration    # invoke integration
pytest tests/unit/test_synthesizers/test_ctgan.py   # single test file

# Build
python -m build --wheel --sdist
```

Training scripts are run directly from `Syn_ctgan/`:
```bash
cd Syn_ctgan

# Run one dataset, 3 seeds (42/43/44):
bash run_mae01scale_max.sh unsw      # Linux
run_mae01scale_max.bat unsw          # Windows

# Or single run:
python train_mae01scale_max.py --d unsw --seed 42 --time 4 \
  --e 150 --b 128 --lae 0.1 --mae 2.0 \
  --noise_std 0.3 --noise_scale 1.5 \
  --gate_stability 0.05 --gate_window 10 --cov_floor 0.05
```

## Architecture

### Synthesizer class hierarchy

```
BaseSynthesizer  (ctgan/synthesizers/base.py)
└── CTGAN  (ctgan/synthesizers/ctgan.py)
└── MAECTGANSynthesizer base  (ctgan/synthesizers/maectgan.py)
    └── M01AECTGAN  (ctgan/synthesizers/m01aectgan.py)
            Joint CTGAN + M01AE training, two-sided margin loss,
            fixed mae_start_ratio warmup, dual best-checkpoint tracking.
        └── M01AECAdapt  (ctgan/synthesizers/m01aecadapt.py)
                Adds adaptive noise_std (separation-target zone) +
                data-driven CTGAN gate (discriminator CoV).
            └── MAE01Scale  (ctgan/synthesizers/mae01scale.py)
                    Replaces separation-target noise adapt with
                    z-norm self-scaling: noise_std = noise_scale × mean‖z_normal‖.
                └── MAE01ScaleMax  (ctgan/synthesizers/mae01scale_max.py)
                        Fixes gate CoV denominator for datasets where
                        D_loss converges near 0 (e.g. NSL-KDD):
                        cov = std / max(|mean|, cov_floor).
```

Each synthesizer in `ctgan/synthesizers/` has a corresponding **SDV wrapper** in `sdv/single_table/` with the same name (e.g. `m01aectgan.py` → `M01AECTGANSynthesizer`). Wrappers handle metadata detection, `HyperTransformer` preprocessing, and the dual-checkpoint callback (`_save_best_model`).

### Training loop (M01AECTGAN and descendants)

Each epoch inner step runs in order:
1. **CTGAN discriminator step(s)** — WGAN-GP loss
2. **CTGAN generator step** → produces `fakeact` (transformed space)
3. **M01AE step** — creates pseudo-anomaly `x_pseudo`:
   - `type_train="noise"` → Gaussian noise on real normal samples
   - `type_train="std"` → noise on CTGAN samples
   - `type_train="noise_std"` → 50/50 mix
   - Before gate opens: always falls back to noise-only

End of each epoch: validate on held-out normal samples, save dual checkpoints (best_loss + best_auc).

### CTGAN quality gate (M01AECAdapt / MAE01Scale)

Replaces the fixed `mae_start_ratio` warmup. Gate opens when:
```
CoV = std(EMA_disc_loss[-gate_window:]) / max(|mean(...)|, cov_floor)  ≤  gate_stability
```
Uses **discriminator loss** (not generator): D loss is a Wasserstein proxy that converges stably; G loss oscillates throughout WGAN-GP training. `cov_floor` (MAE01ScaleMax only) prevents division-by-zero when D_loss converges near 0.

### Checkpoint system

Post-training evaluation saves to two separate subfolders to avoid collision:
- `checkpoints/best_loss/` — best model by validation loss
- `checkpoints/best_auc/`  — best model by validation AUC

Result CSV files: `test_mode_re.csv`, `test_mode_latent.csv`, `test_mode_latent95.csv`, `test_mode_total95.csv`.

### Data loading

All datasets loaded via `Syn_ctgan/process_data.py`. Raw CSVs expected under `Syn_ctgan/data/<dataset>/`. Supported datasets: `unsw`, `nsl_kdd`, `cic_iot`, `wsn_ds`, `cic_ids`, `cidds_001`, `insdn`, `bot_iot`, `ton_iot`, `nb_iot`.

The loaders return `(df_full, y_full, df_test, y_test)`. Training scripts split `df_full` into train/val (80/20 of normal-only samples), use `df_test` for evaluation.

### Key path convention

`sys.path.insert(0, ROOT_DIR)` in every training script ensures the **local** `ctgan/` and `sdv/` packages take precedence over any installed versions. Always run training scripts from inside `Syn_ctgan/`.

## Adding a new synthesizer

1. Create `ctgan/synthesizers/<name>.py` — inherit from the closest existing class and override only what changes.
2. Create `sdv/single_table/<name>.py` — inherit from the corresponding SDV wrapper, update `_model_kwargs`, override `_fit()` to instantiate the new class.
3. Create `Syn_ctgan/train_<name>.py` — copy the nearest training script and change imports, `run_folder` prefix, results_dir tag, and synthesizer instantiation.
4. Create `Syn_ctgan/run_<name>.sh` and `run_<name>.bat` with seeds 42/43/44, `--time 4`.

## Baseline comparisons (external, vendored)

Two external repos are vendored into this working tree purely as comparison
baselines for the paper. **Their own core algorithm code must never be
edited** — only wrapper/data-prep scripts written alongside them may change.

### CLAN

`D:\Code_papper\CTGAN\CLGAN\CLAN-main\CLAN-main` (sibling repo, outside this
directory tree). Its own `data.load_data.get_data()` is the reference
preprocessing pipeline reused by several of this project's comparison
scripts (see below) — it MinMax-scales features to `[0, 1]` and **drops**
non-numeric categorical columns (`drop_non_numeric=True`), rather than
encoding them; that convention carries over wherever this project loads
CLAN-space data. `results/generated/<dataset>_seed_<seed>_augmented.csv`
holds CLAN's train-normal rows (`source=real`) plus its stateless
perturbation-based pseudo-anomalies (`source=clan_augmented`) — see
`export_augmented_data.py` in that repo.

Comparison scripts in `Syn_ctgan/`: `visualize_tsne_gendata.py`,
`visualize_centroid_distance_vs_clan.py`, `visualize_scalemax_gendata.py`
(SSAE-only, styled to match CLAN's polar-PCA convention). The 2
paper-highlighted rare/hard attacks per dataset (UNSW-NB15: Shellcode/Worms,
NSL-KDD: R2L/U2R, CIC-IoT 2023: Recon/Spoofing, WSN-DS: Blackhole/Grayhole)
recur across these as `RARE_ATTACKS`/`ATTACK_PAIRS`.

### DROCC (`DROCC_icml2020/`)

Unofficial code for *DROCC: Deep Robust One-Class Classification* (ICML
2020; paper at `DROCC_icml2020/DROCC.pdf`) — a third baseline. Trains a
small feed-forward net one-class-style on **normal-only** data: a
classification head pushes real normal points to label `1`, while an
adversarial inner loop searches for the "hardest" synthetic points in a
`[r, γr]` shell around each training point (Gaussian-initialized, projected
via normalized gradient ascent) and trains them toward label `0` — no
generator network, no side information, just the local-manifold assumption
that off-manifold neighborhoods are anomalous.

Only `DROCC_icml2020/tabular/` is relevant here (feed-forward `TwoLayer` net
on tabular features — see `tabular/models.py`, `tabular/train_tabular.py`,
`tabular/loss_tabular.py`). **Label convention is inverted from the rest of
this project: `1` = normal, `0` = anomaly** (matches `train_lab =
np.ones(...)` in `one_class_main_tabular.py` and the CE-toward-`0` targets
in `loss_tabular.py`'s adversarial loss — confirm this before writing any
new eval/plotting code against DROCC's outputs). Input must be pre-encoded,
purely-numeric `train_data.npy` (normal-only) / `test_data.npy` /
`test_labels.npy` (`{root_data}` in `one_class_main_tabular.py -d`); the
script re-standardizes with its own train-set mean/std internally, so any
consistent numeric scale works going in.

`Syn_ctgan/prepare_drocc_data.py` builds those `.npy` triplets for this
project's 4 CLAN-comparison datasets (unsw, nsl_kdd, cic_iot, wsn_ds) by
reusing CLAN's `get_data()` pipeline (so features are pre-encoded
identically to the CLAN baseline) and flipping the label convention.
`Syn_ctgan/run_drocc_datasets.sh` / `.bat` then invoke DROCC's own unmodified
`one_class_main_tabular.py` per dataset/seed via subprocess — hyperparameters
(`--inp_radius` especially) are picked per-dataset via
`inp_radius ≈ sqrt(num_features)`, matching the pattern in the paper's own
tabular example configs (Abalone/Arrhythmia/Thyroid: `r ≈ sqrt(d)` in each
case), since `r` sets the scale of a `~N(0, I_d)` initial perturbation in
Algorithm 1; `--lr 1e-5` for every dataset (the paper's own example lr
blows up logits during the 50-epoch CE-only warmup on this project's data,
producing NaN the instant the adversarial phase starts — see
`run_drocc_datasets.sh`'s comment for the full diagnosis).

**Two infra bugs, explicitly user-approved, patched directly in
`tabular/train_tabular.py`** (the only edits made to DROCC's own code, both
pure device-placement, no algorithm/loss change): `epoch_adv_loss_inp`'s
init tensor wasn't moved to `device` (crashed the instant the adversarial
phase started, since the CE-loss accumulator was CPU while the adversarial
loss was CUDA); `test()` hardcoded `.cuda()` instead of using its own
`device` parameter (crashes immediately with no GPU present at all — this
is why an env-var-based CPU workaround doesn't work either). Do not
"helpfully" revert these; they're required for the script to run to
completion at all on this machine.

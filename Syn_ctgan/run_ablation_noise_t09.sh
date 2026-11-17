#!/bin/bash
# Table [T9] ablation: sensitivity of TPR@FAR=1% / AUC to the perturbation
# strategy, one axis at a time, baseline (t_04) values excluded here since
# that point already exists -- same convention as the lambda/margin
# ablations (run_ablation_lambda_mae01scale_max.sh / run_ablation_m_*.sh).
#
# Axes (one changed per run, all other args at the official t_04 config):
#   continuous-attribute proportion : noise_ratio     0.25 / 0.75  (baseline 0.5)
#   categorical-column proportion   : cat_noise_ratio 0.05 / 0.20  (baseline 0.1)
#   continuous noise distribution   : noise_type      uniform      (baseline global = Gaussian)
#   categorical perturbation op     : cat_noise_type  shuffle / flip (baseline combined)
#
# All runs share one output folder, res/t_09_noise_rate/, via --out_tag
# (seed is NOT encoded in the per-config subfolder name, so if more seeds
# are added later under this ablation, give them their own tag, e.g.
# t_09_noise_rate_se_43, to avoid overwriting seed 42's runs).
#
# Usage: bash run_ablation_noise_t09.sh
set -e

SEED=42
OUT_TAG=t_09_noise_rate
PER=1.0

DATASETS=(unsw cic_iot nsl_kdd wsn_ds)

# Baseline (t_04) values, repeated here so each run only overrides one axis.
BASE_NOISE_RATIO=0.5
BASE_CAT_NOISE_RATIO=0.1
BASE_NOISE_TYPE=global
BASE_CAT_NOISE_TYPE=combined

COMMON_ARGS=(
    --e 150
    --b 128
    --lae 0.1
    --mae 2.0
    --mae_loss_mode renormal
    --type_train noise_std
    --noise_std 0.3
    --patience 50
    --noise_scale 1.5
    --gate_stability 0.05
    --gate_window 10
    --cov_floor 0.05
    --eps_zero 0.1
    --tol 0.08
    --per "$PER"
    --seed "$SEED"
    --out_tag "$OUT_TAG"
)

# label:noise_ratio:cat_noise_ratio:noise_type:cat_noise_type
CONFIGS=(
    "noise_ratio0.25:0.25:${BASE_CAT_NOISE_RATIO}:${BASE_NOISE_TYPE}:${BASE_CAT_NOISE_TYPE}"
    "noise_ratio0.75:0.75:${BASE_CAT_NOISE_RATIO}:${BASE_NOISE_TYPE}:${BASE_CAT_NOISE_TYPE}"
    "cat_noise_ratio0.05:${BASE_NOISE_RATIO}:0.05:${BASE_NOISE_TYPE}:${BASE_CAT_NOISE_TYPE}"
    "cat_noise_ratio0.20:${BASE_NOISE_RATIO}:0.20:${BASE_NOISE_TYPE}:${BASE_CAT_NOISE_TYPE}"
    "noise_type_uniform:${BASE_NOISE_RATIO}:${BASE_CAT_NOISE_RATIO}:uniform:${BASE_CAT_NOISE_TYPE}"
    "cat_noise_type_shuffle:${BASE_NOISE_RATIO}:${BASE_CAT_NOISE_RATIO}:${BASE_NOISE_TYPE}:shuffle"
    "cat_noise_type_flip:${BASE_NOISE_RATIO}:${BASE_CAT_NOISE_RATIO}:${BASE_NOISE_TYPE}:flip"
)

echo "========================================"
echo "  Table [T9] perturbation-strategy ablation, seed=$SEED"
echo "  output: res/$OUT_TAG/"
echo "========================================"

for DATASET in "${DATASETS[@]}"; do
    for CFG in "${CONFIGS[@]}"; do
        IFS=':' read -r LABEL NOISE_RATIO CAT_NOISE_RATIO NOISE_TYPE CAT_NOISE_TYPE <<< "$CFG"
        echo ""
        echo "======== dataset=$DATASET  config=$LABEL ========"
        python train_mae01scale_max.py \
            --d "$DATASET" \
            --time 9 \
            --noise_ratio "$NOISE_RATIO" \
            --cat_noise_ratio "$CAT_NOISE_RATIO" \
            --noise_type "$NOISE_TYPE" \
            --cat_noise_type "$CAT_NOISE_TYPE" \
            "${COMMON_ARGS[@]}"
    done
done

echo ""
echo "========================================"
echo "  ALL DONE: Table [T9] ablation for ${DATASETS[*]} (seed=$SEED)"
echo "========================================"

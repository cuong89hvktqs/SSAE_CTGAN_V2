#!/bin/bash
# One-off experiment: pre-one-hot-encode UNSW-NB15's proto/service/state
# columns before CTGAN (train_mae01scale_max_unsw_onehot.py), instead of
# letting CTGAN model them as categorical columns itself. Seed 42 already
# ran under res/t_11_SSAE_se_42/ -- this script adds seeds 43-51 to the
# same tag (--time 11) so all 10 seeds end up directly comparable to the
# official t_04 UNSW baseline.
# Usage: bash run_unsw_onehot_seeds43_51.sh [per]
set -e

DATASET=unsw

PER=${1:-1.0}

SEEDS=(43 44 45 46 47 48 49 50 51)

COMMON_ARGS=(
    --e 150
    --b 128
    --lae 0.1
    --mae 2.0
    --mae_loss_mode renormal
    --type_train noise_std
    --noise_std 0.3
    --noise_type global
    --noise_ratio 0.5
    --cat_noise_type combined
    --cat_noise_ratio 0.1
    --patience 50
    --noise_scale 1.5
    --gate_stability 0.05
    --gate_window 10
    --cov_floor 0.05
    --eps_zero 0.1
    --tol 0.08
    --per "$PER"
)

echo "========================================"
echo "  UNSW one-hot-before-CTGAN experiment | dataset: $DATASET"
echo "  Seeds: ${SEEDS[*]}"
echo "  per: $PER"
echo "========================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- $DATASET seed=$SEED ---"
    OUT_DIR="res/t_11_SSAE_se_${SEED}"
    mkdir -p "$OUT_DIR"
    python train_mae01scale_max_unsw_onehot.py \
        --d "$DATASET" \
        --seed "$SEED" \
        --time 11 \
        "${COMMON_ARGS[@]}" \
        > "$OUT_DIR/unsw_onehot_run.log" 2>&1
    echo "    -> log: $OUT_DIR/unsw_onehot_run.log"
done

echo ""
echo "========================================"
echo "  DONE: dataset=$DATASET  seeds=${SEEDS[*]}"
echo "========================================"

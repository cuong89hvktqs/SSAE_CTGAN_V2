#!/bin/bash
# Usage: bash run_mae01scale_max.sh [per]
set -e

# ── Datasets to run (edit this list to add/remove datasets) ──────────────────
DATASETS=(unsw cic_iot nsl_kdd wsn_ds nb_iot cic_ids)

PER=${1:-1.0}

SEEDS=(42 43 44)

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
echo "  MAE01ScaleMax | datasets: ${DATASETS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  per: $PER"
echo "========================================"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "======== dataset=$DATASET ========"
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- $DATASET seed=$SEED ---"
        python train_mae01scale_max.py \
            --d "$DATASET" \
            --seed "$SEED" \
            --time 4 \
            "${COMMON_ARGS[@]}"
    done
done

echo ""
echo "========================================"
echo "  DONE: datasets=${DATASETS[*]}  seeds=${SEEDS[*]}"
echo "========================================"

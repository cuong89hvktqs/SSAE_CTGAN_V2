#!/bin/bash
# t_07: single extra config m=9.0, lambda=1.0 (seed 42 only), 4 core datasets.
# All other args match the standard m=2.0/lambda=0.1 runs (noise_std, per=1.0).
# Usage: bash run_t07_m9_l1.sh
set -e

SEEDS=(42)
TIME=7
PER=1.0

DATASETS=(unsw cic_iot nsl_kdd wsn_ds)

MARGIN=9.0
LAMBDA=1.0

COMMON_ARGS=(
    --e 150
    --b 128
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
echo "  t_07 | m=$MARGIN  lambda=$LAMBDA"
echo "  datasets: ${DATASETS[*]}"
echo "  seeds: ${SEEDS[*]}"
echo "========================================"

for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "======== seed=$SEED  dataset=$DATASET  margin=$MARGIN  lambda=$LAMBDA ========"
        python train_mae01scale_max.py \
            --d "$DATASET" \
            --seed "$SEED" \
            --time "$TIME" \
            --lae "$LAMBDA" \
            --mae "$MARGIN" \
            "${COMMON_ARGS[@]}"
    done
done

echo ""
echo "========================================"
echo "  ALL DONE: t_07 m=$MARGIN lambda=$LAMBDA for ${DATASETS[*]} (seeds=${SEEDS[*]})"
echo "========================================"

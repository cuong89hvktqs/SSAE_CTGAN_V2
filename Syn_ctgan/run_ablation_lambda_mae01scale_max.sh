#!/bin/bash
# Ablation study for MAE01ScaleMax over lambda, margin fixed at 2.0.
# All other args match the standard m=2.0/lambda=0.1 runs (noise_std, per=1.0).
# Usage: bash run_ablation_lambda_mae01scale_max.sh
set -e

SEEDS=(42 43 44)
TIME=2
PER=1.0

DATASETS=(unsw cic_iot nsl_kdd wsn_ds)

# LAMBDA_LIST=(0.01 0.05 0.5 1.0)
LAMBDA_LIST=(0)
FIXED_MARGIN=2.0

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
echo "  Lambda ablation (margin fixed at $FIXED_MARGIN)"
echo "  lambda in: ${LAMBDA_LIST[*]}"
echo "========================================"

for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for LAMBDA in "${LAMBDA_LIST[@]}"; do
            echo ""
            echo "======== seed=$SEED  dataset=$DATASET  margin=$FIXED_MARGIN  lambda=$LAMBDA ========"
            python train_mae01scale_max.py \
                --d "$DATASET" \
                --seed "$SEED" \
                --time "$TIME" \
                --lae "$LAMBDA" \
                --mae "$FIXED_MARGIN" \
                "${COMMON_ARGS[@]}"
        done
    done
done

echo ""
echo "========================================"
echo "  ALL DONE: lambda ablation for ${DATASETS[*]} (seeds=${SEEDS[*]})"
echo "========================================"

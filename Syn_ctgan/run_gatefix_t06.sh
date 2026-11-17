#!/bin/bash
# Fixed-gate ablation for MAE01ScaleMax (m=2.0, lambda=0.1, noise_std):
# opens the CTGAN gate at a FIXED epoch (0/20/40) instead of the adaptive
# CoV/near-zero logic, via --gate_fixed_epoch (see
# ctgan/synthesizers/mae01scale_max.py's _maybe_open_gate override).
# All seeds 42..51, --time 6.
#
# Results -> res/t_06_SSAE_se_<seed>/per_<per>/<dataset>_..._catra0.1_gf<epoch>/
#
# Usage: bash run_gatefix_t06.sh [per]
set -e

DATASETS=(unsw cic_iot nsl_kdd wsn_ds)
SEEDS=(42 43 44 45 46 47 48 49 50 51)
GATE_EPOCHS=(0 20 40)

PER=${1:-1.0}

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
echo "  MAE01ScaleMax fixed-gate ablation (t_06) | datasets: ${DATASETS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Gate-fixed epochs: ${GATE_EPOCHS[*]}"
echo "  per: $PER"
echo "========================================"

for GATE_EPOCH in "${GATE_EPOCHS[@]}"; do
    echo ""
    echo "==================== gate_fixed_epoch=$GATE_EPOCH ===================="
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "======== dataset=$DATASET ========"
        for SEED in "${SEEDS[@]}"; do
            echo ""
            echo "--- $DATASET seed=$SEED gate_fixed_epoch=$GATE_EPOCH ---"
            python train_mae01scale_max.py \
                --d "$DATASET" \
                --seed "$SEED" \
                --time 6 \
                --gate_fixed_epoch "$GATE_EPOCH" \
                "${COMMON_ARGS[@]}"
        done
    done
done

echo ""
echo "========================================"
echo "  DONE: datasets=${DATASETS[*]}  seeds=${SEEDS[*]}  gate_epochs=${GATE_EPOCHS[*]}"
echo "  Results -> res/t_06_SSAE_se_<seed>/per_$PER/"
echo "========================================"

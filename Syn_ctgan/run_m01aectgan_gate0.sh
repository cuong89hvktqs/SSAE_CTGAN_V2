#!/bin/bash
# "SSAE with the gate always open" comparison run: trains the pre-gate
# ancestor class M01AECTGAN (train_M01AECTGAN.py / M01AECTGANSynthesizer --
# no CoV-based adaptive gate exists in this class at all, only a FIXED
# mae_start_ratio warmup schedule: mae_start_epoch = int(mae_start_ratio *
# epochs); use_ctgan = (epoch >= mae_start_epoch)) with --mae_start_ratio 0.0
# -- i.e. mae_start_epoch=0, so CTGAN-generated pseudo-anomalies are used
# from epoch 0 onward, with no warmup/gate delay whatsoever.
#
# No code changes: train_M01AECTGAN.py already supports this via its own
# --mae_start_ratio CLI flag (default 0.25); this script just calls it with
# 0.0 instead, and otherwise matches run_mae01scale_max.sh's own default
# hyperparameters exactly (e/b/lae/mae/noise_std/noise_type/noise_ratio/
# cat_noise_type/cat_noise_ratio/patience) so the ONLY deliberate difference
# from the main SSAE (MAE01ScaleMax) runs is: fixed-ratio warmup=0 (this
# script) vs. the adaptive CoV/near-zero gate (MAE01ScaleMax) -- gate-only
# params (gate_stability/gate_window/cov_floor/eps_zero/tol/noise_scale)
# don't exist on this ancestor class at all, so they're simply omitted here.
#
# Usage: bash run_m01aectgan_gate0.sh [per]
set -e

DATASETS=(unsw cic_iot nsl_kdd wsn_ds)

PER=${1:-1.0}

SEEDS=(42 43 44)

COMMON_ARGS=(
    --e 150
    --b 128
    --lae 0.1
    --mae 2.0
    --mae_loss_mode renormal
    --type_train noise_std
    --mae_start_ratio 0.0
    --noise_std 0.3
    --noise_type global
    --noise_ratio 0.5
    --cat_noise_type combined
    --cat_noise_ratio 0.1
    --patience 50
    --per "$PER"
)

echo "========================================"
echo "  M01AECTGAN (mae_start_ratio=0.0, gate always open) | datasets: ${DATASETS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  per: $PER"
echo "========================================"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "======== dataset=$DATASET ========"
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- $DATASET seed=$SEED ---"
        python train_M01AECTGAN.py \
            --d "$DATASET" \
            --seed "$SEED" \
            --time 1 \
            "${COMMON_ARGS[@]}"
    done
done

echo ""
echo "========================================"
echo "  DONE: datasets=${DATASETS[*]}  seeds=${SEEDS[*]}"
echo "  Results -> res/m01ae_noise_std_se_<seed>_t_01/per_$PER/"
echo "========================================"

#!/bin/bash
set -e
DATASETS=(unsw cic_iot nsl_kdd wsn_ds)
SEEDS=(43 44)

A_ARGS=(--e 150 --b 128 --lae 0.1 --patience 50 --per 1.0)
B_ARGS=(--e 150 --b 128 --lae 0.1 --mae 2.0 --mae_loss_mode renormal \
    --type_train noise_std --noise_std 0.3 --noise_type global --noise_ratio 0.5 \
    --cat_noise_type combined --cat_noise_ratio 0.1 --patience 50 \
    --noise_scale 1.5 --gate_stability 0.05 --gate_window 10 \
    --cov_floor 0.05 --eps_zero 0.1 --tol 0.08 --per 1.0)

for SEED in "${SEEDS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    echo "=== ControlA $DATASET seed=$SEED ==="
    python train_control_a.py --d "$DATASET" --seed "$SEED" --time 8 "${A_ARGS[@]}"
  done
done

for SEED in "${SEEDS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    echo "=== ControlB $DATASET seed=$SEED ==="
    python train_control_b.py --d "$DATASET" --seed "$SEED" --time 8 "${B_ARGS[@]}"
  done
done

echo "DONE seeds 43/44 Control A/B"

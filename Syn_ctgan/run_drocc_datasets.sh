#!/bin/bash
# Trains DROCC (unmodified DROCC_icml2020/tabular/one_class_main_tabular.py)
# on this project's 4 CLAN-comparison datasets, 3 seeds each, evaluating
# each model IMMEDIATELY after it finishes training (via eval_drocc.py --
# AUC/F1/MCC/... + per-rare-attack TPR@FAR, same report format as SSAE/CLAN)
# rather than waiting for the whole batch -- results for a run are visible
# and up to date in DROCC_icml2020/tabular/results_ssae/ as soon as that run
# completes, and eval_drocc.py merges each new (dataset, seed) into the
# existing summary rather than overwriting it (see its own docstring).
#
# Hyperparameters follow the paper's own tabular examples (tabular/README.md:
# hd=128, batch_size=256, epochs=200, inp_lamda=1, optim=Adam, one_class_adv=1
# throughout); --inp_radius is picked per dataset via inp_radius ~ sqrt(d),
# matching the pattern in all 3 of the paper's own configs (Abalone d=7 r=3,
# Arrhythmia d=274 r=16, Thyroid d=6 r=2.5 -- r is the scale of the
# ~N(0, I_d) perturbation Algorithm 1 initializes the adversarial search
# from).
#
# --lr: the paper's own example lr (1e-3/1e-4) makes the CE-only warmup
# (first 50 epochs, only_ce_epochs in train_tabular.py) push logits to blow
# up on this project's data -- CE loss collapses toward 0, and once the
# adversarial gradient-ascent phase starts at epoch 51, its gradient-
# normalization step (grad / grad_norm in loss_tabular.py) eventually
# divides by a degenerate near-zero/exploded gradient and produces NaN,
# crashing roc_auc_score. lr=1e-5 survives this for unsw/nsl_kdd/wsn_ds
# through all 200 epochs (verified empirically). cic_iot is the outlier --
# by far the largest training pool (255728 rows, ~999 batches/epoch vs.
# unsw's ~175), so it accumulates far more optimizer steps during the same
# 50-epoch warmup and blows up faster; even lr=2e-6 only pushes the eventual
# NaN out to ~epoch 85 instead of ~63; it is NOT fully stable through 200
# epochs at any lr tested so far, it just fails later. A checkpoint is still
# saved whenever validation F1 improves (any time past epoch 50), so a
# late-epoch crash still leaves a usable, if under-trained, model.
#
# --per (positional, default 1.0): fraction of the normal-only training
# pool to use, matching SSAE's own `run_mae01scale_max.sh [per]` / --per
# convention (see clan_style_get_data.py's docstring for the exact
# order-of-operations match: subsampling happens BEFORE the train/val
# split, shrinking both together). Data/checkpoints/results all land under
# a per_<per>/ subtree so different --per runs never collide.
#
# Usage: bash run_drocc_datasets.sh [per]
set -e

# NOTE: training itself does NOT rely on `set -e` to stop on a crashed run
# (e.g. NaN mid-training) -- `python ... | tee log.txt` makes the pipeline's
# exit status be tee's (always 0), so a python crash would otherwise pass
# silently. Each run's exit code is checked explicitly via PIPESTATUS
# instead, and failures are collected into FAILED_RUNS and reported in the
# final summary; eval still runs afterward since DROCC may have saved a
# checkpoint before crashing (see cic_iot note above).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DROCC_TABULAR="$SCRIPT_DIR/../DROCC_icml2020/tabular"

PER=${1:-1.0}
DATASETS=(unsw nsl_kdd cic_iot wsn_ds)
SEEDS=(42 43 44)

echo "========================================"
echo "  Step 1/2: prepare_drocc_data.py (per=$PER)"
echo "========================================"
python "$SCRIPT_DIR/prepare_drocc_data.py" \
    --seeds "$(IFS=,; echo "${SEEDS[*]}")" --per "$PER"

# dataset -> "inp_radius lr" (d and the sqrt(d) derivation, and the cic_iot
# lr rationale, are in the header comment above; see
# Syn_ctgan/prepare_drocc_data.py's printed `d=` values)
declare -A CFG=(
    [unsw]="6.0 0.00001"     # d=39,  sqrt(d)=6.24
    [nsl_kdd]="11.0 0.00001" # d=122, sqrt(d)=11.05
    [cic_iot]="7.0 0.000002" # d=46,  sqrt(d)=6.78 -- lowered further, see above
    [wsn_ds]="4.0 0.00001"   # d=16,  sqrt(d)=4.00
)

COMMON_ARGS=(
    --hd 128
    --batch_size 256
    --epochs 200
    --inp_lamda 1
    --one_class_adv 1
    --optim 0
    --restore 0
)

echo ""
echo "========================================"
echo "  Step 2/2: train + eval per run | datasets: ${DATASETS[*]}  seeds: ${SEEDS[*]}  per=$PER"
echo "========================================"

FAILED_RUNS=()
for DATASET in "${DATASETS[@]}"; do
    read -r RADIUS LR <<< "${CFG[$DATASET]}"
    echo ""
    echo "======== dataset=$DATASET  inp_radius=$RADIUS  lr=$LR ========"
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- $DATASET seed=$SEED : train ---"
        DATA_DIR="data_ssae/per_$PER/$DATASET/seed$SEED"
        MODEL_DIR="log_ssae/per_$PER/$DATASET/seed$SEED"
        (
            cd "$DROCC_TABULAR"
            mkdir -p "$MODEL_DIR"
            python one_class_main_tabular.py \
                -d "$DATA_DIR" \
                --model_dir "$MODEL_DIR" \
                --lr "$LR" \
                --inp_radius "$RADIUS" \
                "${COMMON_ARGS[@]}" \
                2>&1 | tee "$MODEL_DIR/train_log.txt"
        )
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            echo "  [FAILED] $DATASET seed=$SEED -- see $MODEL_DIR/train_log.txt"
            FAILED_RUNS+=("$DATASET seed=$SEED")
        fi

        echo ""
        echo "--- $DATASET seed=$SEED : eval ---"
        python "$SCRIPT_DIR/eval_drocc.py" --dataset "$DATASET" --seeds "$SEED" --per "$PER" || {
            echo "  [EVAL FAILED] $DATASET seed=$SEED -- continuing to next run"
            FAILED_RUNS+=("$DATASET seed=$SEED (eval)")
        }
    done
done

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
    echo ""
    echo "========================================"
    echo "  WARNING: ${#FAILED_RUNS[@]} run(s) crashed mid-training (likely NaN):"
    for r in "${FAILED_RUNS[@]}"; do echo "    - $r"; done
    echo "  (still evaluated above if a checkpoint existed from before the crash)"
    echo "========================================"
fi

echo ""
echo "========================================"
echo "  DONE: datasets=${DATASETS[*]}  seeds=${SEEDS[*]}  per=$PER"
echo "  Results -> DROCC_icml2020/tabular/results_ssae/per_$PER/"
echo "========================================"

"""Regenerate loss_mae_<dataset>_150e.png from an already-saved
mae01scale_max_loss_150e.csv, without retraining -- mirrors the exact M01AE
loss plotting block in train_mae01scale_max.py (kept in sync manually).

Only redraws the 12 default-config (per_1.0, m=2.0, l=0.1, noise_std) runs
across datasets x seeds 42/43/44 under mae01scale_max_se_<seed>_t_04/.

Usage: python replot_mae_loss.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR  = os.path.join(BASE_DIR, "res")

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
SEEDS    = [42, 43, 44]
LAMDA    = 0.1
EPOCHS   = 150


def find_run_dir(dataset, seed):
    pattern = os.path.join(
        RES_DIR, f"mae01scale_max_se_{seed}_t_04", "per_1.0",
        f"{dataset}_{EPOCHS}e_m2.0_l0.1_renormal_scalemax-noise_std_*",
    )
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def _vline(ax, epoch, label_prefix):
    if epoch is not None:
        ax.axvline(epoch, color="orange", linestyle="--",
                   label=f"{label_prefix} (epoch {epoch})")


def replot(dataset, seed):
    run_dir = find_run_dir(dataset, seed)
    if run_dir is None:
        print(f"  [skip] no run dir for {dataset} seed={seed}")
        return
    csv_path = os.path.join(run_dir, f"mae01scale_max_loss_{EPOCHS}e.csv")
    if not os.path.exists(csv_path):
        print(f"  [skip] missing {csv_path}")
        return

    loss_df = pd.read_csv(csv_path)
    gate_epoch = None
    if "CTGAN Active" in loss_df.columns and (loss_df["CTGAN Active"] == 1).any():
        gate_epoch = int(loss_df.loc[loss_df["CTGAN Active"] == 1, "Epoch"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_df["Epoch"], loss_df["MAE Loss"],            label="Total loss")
    ax.plot(loss_df["Epoch"], loss_df["Reconstruction Loss"], label="Reconstruction loss")
    ax.plot(loss_df["Epoch"], LAMDA * loss_df["Latent Loss"], label="Margin loss")
    _vline(ax, gate_epoch, "Gate open")
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    out_path = os.path.join(run_dir, f"loss_mae_{dataset}_{EPOCHS}e.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def main():
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"[{dataset} seed={seed}]")
            replot(dataset, seed)


if __name__ == "__main__":
    main()

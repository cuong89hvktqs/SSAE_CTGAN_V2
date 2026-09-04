"""Conceptual (schematic) diagram of the Pseudo Abnormal Generator idea:
fake-normal data forms a tight cluster, fake-anomalous (perturbed) data
scatters outside it, and a density-based "discrimination boundary" (KDE
contours at 0.2/0.5/0.8 of peak density) separates them. Panel (b) overlays
real-normal data on top to show it falls inside the same boundary as the
fake-normal cluster.

This is a purely illustrative/synthetic figure (not derived from any real
dataset or trained model) -- intended as a conceptual overview figure for
the paper, not a quantitative results plot.

Usage: python visualize_pag_concept_diagram.py
Output: res/figures_v2/pag_concept_diagram.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

BASE_DIR = Path(__file__).parent
OUT_PATH = BASE_DIR / "res" / "figures_v2" / "pag_concept_diagram.png"

SEED = 42
LIM = 6

GREEN  = "#4CAF50"  # fake-normal
RED    = "#F08080"  # fake-anomalous
PURPLE = "#8C87D6"  # real-normal

LEVELS = [0.2, 0.5, 0.8]
LEVEL_STYLES = {0.2: "--", 0.5: "-", 0.8: ":"}
FAKE_NORMAL_SIGMA = 1.35


def make_data(rng):
    n_fake_normal = 1200
    fake_normal = rng.normal(loc=[0, 0], scale=[FAKE_NORMAL_SIGMA, FAKE_NORMAL_SIGMA],
                             size=(n_fake_normal, 2))

    n_real_normal = 1200
    real_normal = rng.normal(loc=[0.1, -0.05], scale=[1.45, 1.45], size=(n_real_normal, 2))

    n_fake_anom = 170
    pts = []
    while len(pts) < n_fake_anom:
        p = rng.uniform(-LIM, LIM, size=2)
        r = np.linalg.norm(p)
        if r > 2.1 or rng.random() < 0.06:
            pts.append(p)
    fake_anom = np.array(pts)

    return fake_normal, fake_anom, real_normal


def kde_grid(points, grid_n=200):
    kde = gaussian_kde(points.T)
    xs = np.linspace(-LIM, LIM, grid_n)
    ys = np.linspace(-LIM, LIM, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    zz = zz / zz.max()
    return xx, yy, zz


def draw_boundary(ax, xx, yy, zz, sigma, with_labels):
    theta = np.deg2rad(25)
    for lvl in LEVELS:
        ax.contour(xx, yy, zz, levels=[lvl], colors="black",
                  linestyles=LEVEL_STYLES[lvl], linewidths=1.4)
        if with_labels:
            # Closed-form radius of the lvl-contour for an (approximately)
            # isotropic Gaussian KDE: density(r) = peak * exp(-r^2/(2*sigma^2)).
            # Manual placement (fixed rotation=0) instead of ax.clabel, which
            # otherwise rotates the text to follow the contour's local slope
            # and makes it hard to read.
            r = sigma * np.sqrt(-2 * np.log(lvl))
            x = (r + 0.15) * np.cos(theta)
            y = (r + 0.15) * np.sin(theta)
            text = f"Threshold = {lvl}" if lvl == 0.2 else f"{lvl}"
            ax.text(x, y, text, fontsize=9, ha="left", va="center")
    # dummy handle for the legend (one entry representing the boundary family)
    ax.plot([], [], color="black", linestyle="--", linewidth=1.4,
           label="Discrimination boundary")


def style_axes(ax, panel_label):
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_title(panel_label, loc="left", fontsize=15, fontweight="bold")


def main():
    rng = np.random.default_rng(SEED)
    fake_normal, fake_anom, real_normal = make_data(rng)
    xx, yy, zz = kde_grid(fake_normal)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

    # ── Panel (a): fake-normal + fake-anomalous + boundary ────────────────
    ax_a.scatter(fake_normal[:, 0], fake_normal[:, 1], s=14, color=GREEN,
               alpha=0.55, edgecolor="none", label="Fake-normal data")
    ax_a.scatter(fake_anom[:, 0], fake_anom[:, 1], s=22, color=RED,
               alpha=0.65, edgecolor="none", label="Fake-anomalous data")
    draw_boundary(ax_a, xx, yy, zz, FAKE_NORMAL_SIGMA, with_labels=True)
    style_axes(ax_a, "(a)")
    ax_a.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # ── Panel (b): + real-normal overlay ───────────────────────────────────
    ax_b.scatter(real_normal[:, 0], real_normal[:, 1], s=14, color=PURPLE,
               alpha=0.55, edgecolor="none", label="Real-normal data")
    ax_b.scatter(fake_normal[:, 0], fake_normal[:, 1], s=14, color=GREEN,
               alpha=0.55, edgecolor="none", label="Fake-normal data")
    ax_b.scatter(fake_anom[:, 0], fake_anom[:, 1], s=22, color=RED,
               alpha=0.65, edgecolor="none", label="Fake-anomalous data")
    draw_boundary(ax_b, xx, yy, zz, FAKE_NORMAL_SIGMA, with_labels=False)
    style_axes(ax_b, "(b)")
    ax_b.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PATH), bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

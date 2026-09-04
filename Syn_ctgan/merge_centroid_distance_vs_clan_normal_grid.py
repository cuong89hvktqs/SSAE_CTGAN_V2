"""Combine the 4 v4_normal centroid-distance-vs-clan (Normal-centroid) images
into one 2x2 grid, plain paste (no per-panel caption baked in -- the LaTeX
figure caption in TH/latex_centroid_distance_vs_clan_normal.tex already
spells out panel order (a)-(d)). Same row convention used throughout this
project: UNSW-NB15 + NSL-KDD on top, CIC-IoT 2023 + WSN-DS below.

Input:  res/figures_v2/centroid_distance_vs_clan/v4_normal/<dataset>_seed42_centroid_distance_vs_clan.png
Output: res/figures_v2/centroid_distance_vs_clan/v4_normal/grid_2x2_seed42.png
"""

from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent
SRC_DIR  = BASE_DIR / "res" / "figures_v2" / "centroid_distance_vs_clan" / "v4_normal"
OUT_PATH = SRC_DIR / "grid_2x2_seed42.png"

PAD = 10

CELLS = [
    ["unsw_seed42_centroid_distance_vs_clan.png",    "nsl_kdd_seed42_centroid_distance_vs_clan.png"],
    ["cic_iot_seed42_centroid_distance_vs_clan.png",  "wsn_ds_seed42_centroid_distance_vs_clan.png"],
]


def main():
    cells = [[Image.open(SRC_DIR / fn).convert("RGB") for fn in row] for row in CELLS]

    cell_w = max(c.width for row in cells for c in row)
    cell_h = max(c.height for row in cells for c in row)

    grid_w = cell_w * 2 + PAD
    grid_h = cell_h * 2 + PAD
    grid = Image.new("RGB", (grid_w, grid_h), "white")

    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            x = c * (cell_w + PAD)
            y = r * (cell_h + PAD)
            grid.paste(cell, (x, y))

    grid.save(OUT_PATH)
    print(f"Saved -> {OUT_PATH}  ({grid_w}x{grid_h})")


if __name__ == "__main__":
    main()

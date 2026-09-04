"""Combine 4 already-merged (CLAN|SSAE) per-attack figures into one 2x2 grid.

Layout:
    Row 1: (a) UNSW-NB15 (Generic)         | (b) CIC-IoT-2023 (Reconnaissance)
    Row 2: (c) NSL-KDD (Probe)             | (d) WSN-DS (Flooding)

Each cell already shows CLAN (left) vs SSAE (right) for that attack, with
"CLAN"/"SSAE" captioned below each half (from merge_clan_ssae_figures.py).
This script appends the standard journal-style subfigure caption -- e.g.
"(a) UNSW-NB15 (Generic)" -- BELOW each full cell, matching the usual
figure convention (caption under the panel, not above it).

Output: res/figures_v2/merged/grid_2x2_seed42.png
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).parent
MERGED   = BASE_DIR / "res" / "figures_v2" / "merged"
OUT_PATH = MERGED / "grid_2x2_seed42.png"

CAPTION_H = 34
PAD       = 10
FONT_SIZE = 22

CELLS = [
    [("unsw",    "seed42_generic.png",  "(a) UNSW-NB15 (Generic)"),
     ("cic_iot", "seed42_recon.png",    "(b) CIC-IoT-2023 (Reconnaissance)")],
    [("nsl_kdd", "seed42_1.png",        "(c) NSL-KDD (Probe)"),
     ("wsn_ds",  "seed42_flooding.png", "(d) WSN-DS (Flooding)")],
]


def get_font():
    try:
        return ImageFont.truetype("arial.ttf", FONT_SIZE)
    except Exception:
        return ImageFont.load_default()


def load_cell(dataset, filename, caption, font):
    img = Image.open(MERGED / dataset / filename).convert("RGB")
    canvas = Image.new("RGB", (img.width, img.height + CAPTION_H), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), caption, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, img.height + 6), caption, fill="black", font=font)
    return canvas


def main():
    font = get_font()
    cells = [[load_cell(ds, fn, caption, font) for ds, fn, caption in row] for row in CELLS]

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

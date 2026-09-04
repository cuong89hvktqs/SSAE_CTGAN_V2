"""Combine the 4 per-dataset centroid-distance-vs-attacks figures into one
2x2 grid (same row convention used throughout this project: UNSW-NB15 +
NSL-KDD on top, CIC-IoT 2023 + WSN-DS below), with a lettered caption under
each panel.

Input:  res/figures_v2/centroid_distance_vs_attacks/<dataset>_seed42_centroid_distance.png
Output: res/figures_v2/centroid_distance_vs_attacks/grid_2x2_seed42.png
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).parent
SRC_DIR  = BASE_DIR / "res" / "figures_v2" / "centroid_distance_vs_attacks"
OUT_PATH = SRC_DIR / "grid_2x2_seed42.png"

CAPTION_H = 34
PAD       = 10
FONT_SIZE = 22

CELLS = [
    [("unsw_seed42_centroid_distance.png",    "(a) UNSW-NB15"),
     ("nsl_kdd_seed42_centroid_distance.png", "(b) NSL-KDD")],
    [("cic_iot_seed42_centroid_distance.png", "(c) CIC-IoT 2023"),
     ("wsn_ds_seed42_centroid_distance.png",  "(d) WSN-DS")],
]


def get_font():
    try:
        return ImageFont.truetype("arial.ttf", FONT_SIZE)
    except Exception:
        return ImageFont.load_default()


def load_cell(filename, caption, font):
    path = SRC_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"missing {path} -- run visualize_centroid_distance_vs_attacks.py first")
    img = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (img.width, img.height + CAPTION_H), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), caption, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, img.height + 6), caption, fill="black", font=font)
    return canvas


def main():
    font = get_font()
    cells = [[load_cell(fn, caption, font) for fn, caption in row] for row in CELLS]

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

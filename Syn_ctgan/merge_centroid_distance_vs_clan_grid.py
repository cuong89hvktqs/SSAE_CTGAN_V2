"""Combine the 4 per-dataset centroid_distance_vs_clan figures into a 2x2
grid, once per "attack slot" (the 1st attack listed per dataset in
ATTACK_PAIRS -- Shellcode/R2L/Recon/Blackhole -- and the 2nd -- Worms/U2R/
Spoofing/Grayhole), since each dataset contributes 2 images (one per attack
used as the centroid) rather than 1. Same row convention used throughout
this project: UNSW-NB15 + NSL-KDD on top, CIC-IoT 2023 + WSN-DS below.

Works on any subfolder of res/figures_v2/centroid_distance_vs_clan/ that
holds the 8 per-(dataset,attack) PNGs (e.g. mean_normal/, v2_attack/, or a
future variant) -- pass it via --dir.

Usage: python merge_centroid_distance_vs_clan_grid.py --dir v2_attack
Output: <dir>/slot1_grid_2x2_seed42.png (Shellcode/R2L/Recon/Blackhole)
        <dir>/slot2_grid_2x2_seed42.png (Worms/U2R/Spoofing/Grayhole)
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR / "res" / "figures_v2" / "centroid_distance_vs_clan"

CAPTION_H = 34
PAD       = 10
FONT_SIZE = 22

SLOTS = {
    "slot1_grid_2x2_seed42.png": [
        [("unsw_shellcode_seed42_centroid_distance_vs_clan.png",    "(a) UNSW-NB15 (Shellcode)"),
         ("nsl_kdd_r2l_seed42_centroid_distance_vs_clan.png",       "(b) NSL-KDD (R2L)")],
        [("cic_iot_recon_seed42_centroid_distance_vs_clan.png",     "(c) CIC-IoT 2023 (Recon)"),
         ("wsn_ds_blackhole_seed42_centroid_distance_vs_clan.png",  "(d) WSN-DS (Blackhole)")],
    ],
    "slot2_grid_2x2_seed42.png": [
        [("unsw_worms_seed42_centroid_distance_vs_clan.png",        "(a) UNSW-NB15 (Worms)"),
         ("nsl_kdd_u2r_seed42_centroid_distance_vs_clan.png",       "(b) NSL-KDD (U2R)")],
        [("cic_iot_spoofing_seed42_centroid_distance_vs_clan.png",  "(c) CIC-IoT 2023 (Spoofing)"),
         ("wsn_ds_grayhole_seed42_centroid_distance_vs_clan.png",   "(d) WSN-DS (Grayhole)")],
    ],
}


def get_font():
    try:
        return ImageFont.truetype("arial.ttf", FONT_SIZE)
    except Exception:
        return ImageFont.load_default()


def load_cell(src_dir, filename, caption, font):
    path = src_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    img = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (img.width, img.height + CAPTION_H), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), caption, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, img.height + 6), caption, fill="black", font=font)
    return canvas


def build_grid(src_dir, cells, out_path, font):
    grid_cells = [[load_cell(src_dir, fn, caption, font) for fn, caption in row] for row in cells]
    cell_w = max(c.width for row in grid_cells for c in row)
    cell_h = max(c.height for row in grid_cells for c in row)
    grid_w = cell_w * 2 + PAD
    grid_h = cell_h * 2 + PAD
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    for r, row in enumerate(grid_cells):
        for c, cell in enumerate(row):
            grid.paste(cell, (c * (cell_w + PAD), r * (cell_h + PAD)))
    grid.save(out_path)
    print(f"Saved -> {out_path}  ({grid_w}x{grid_h})")


def main():
    parser = argparse.ArgumentParser("Merge centroid_distance_vs_clan images into 2x2 grids")
    parser.add_argument("--dir", type=str, required=True,
                        help="subfolder under res/figures_v2/centroid_distance_vs_clan/ "
                             "holding the 8 per-(dataset,attack) PNGs")
    args = parser.parse_args()

    src_dir = ROOT_DIR / args.dir
    if not src_dir.is_dir():
        raise FileNotFoundError(f"no such folder: {src_dir}")

    font = get_font()
    for out_name, cells in SLOTS.items():
        build_grid(src_dir, cells, src_dir / out_name, font)


if __name__ == "__main__":
    main()

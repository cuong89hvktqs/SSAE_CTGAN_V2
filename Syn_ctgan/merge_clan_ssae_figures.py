"""Merge CLAN and MAE01ScaleMax (SSAE) per-attack figures side by side.

Source:
  res/figures_v2/CLGAN/<dataset>/seed<N>_<attack>.png
  res/figures_v2/SSAE/<dataset>/seed<N>_<attack>.png

Attack names differ in casing/format between the two algorithms (CLAN uses
lowercase / bare class-id for nsl_kdd; SSAE uses the display name / "attack_N"
for nsl_kdd) -- matched here via a normalized key.

Output: res/figures_v2/merged/<dataset>/seed<N>_<attack>.png
    Left = CLAN, Right = SSAE, with a small "(a) CLAN" / "(b) SSAE" caption
    under each half (the source images themselves have no text).
"""

import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE_DIR   = Path(__file__).parent
FIG_DIR    = BASE_DIR / "res" / "figures_v2"
CLAN_DIR   = FIG_DIR / "CLGAN"
SSAE_DIR   = FIG_DIR / "SSAE"
OUT_DIR    = FIG_DIR / "merged"

DATASETS = ["unsw", "cic_iot", "nsl_kdd", "wsn_ds"]
SEED     = 42

CAPTION_H = 26
PAD       = 6
FONT_SIZE = 16


def normalize(name):
    """Normalize an attack-file stem to a matching key across CLAN/SSAE."""
    key = name.lower()
    key = re.sub(r"^attack_?", "", key)  # SSAE nsl_kdd: "attack_1" -> "1"
    key = key.replace("-", "_").replace(" ", "_")
    return key


def list_attacks(folder):
    """{normalized_key: actual_filename} for one dataset folder."""
    out = {}
    for p in sorted(folder.glob(f"seed{SEED}_*.png")):
        stem = p.stem[len(f"seed{SEED}_"):]
        out[normalize(stem)] = p.name
    return out


def get_font():
    try:
        return ImageFont.truetype("arial.ttf", FONT_SIZE)
    except Exception:
        return ImageFont.load_default()


def merge_pair(clan_path, ssae_path, out_path, font):
    img_a = Image.open(clan_path).convert("RGB")
    img_b = Image.open(ssae_path).convert("RGB")
    h = max(img_a.height, img_b.height)
    w = img_a.width + PAD + img_b.width

    canvas = Image.new("RGB", (w, h + CAPTION_H), "white")
    canvas.paste(img_a, (0, 0))
    canvas.paste(img_b, (img_a.width + PAD, 0))

    draw = ImageDraw.Draw(canvas)
    for label, x0, x1 in [("CLAN", 0, img_a.width),
                          ("SSAE", img_a.width + PAD, w)]:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        cx = x0 + (x1 - x0 - tw) // 2
        draw.text((cx, h + 4), label, fill="black", font=font)

    canvas.save(out_path)


def main():
    font = get_font()
    for dataset in DATASETS:
        clan_folder = CLAN_DIR / dataset
        ssae_folder = SSAE_DIR / dataset
        if not clan_folder.is_dir() or not ssae_folder.is_dir():
            print(f"[skip] {dataset}: missing CLGAN or SSAE folder")
            continue

        clan_map = list_attacks(clan_folder)
        ssae_map = list_attacks(ssae_folder)
        common = sorted(set(clan_map) & set(ssae_map))
        missing_clan = sorted(set(ssae_map) - set(clan_map))
        missing_ssae = sorted(set(clan_map) - set(ssae_map))

        out_folder = OUT_DIR / dataset
        out_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n[{dataset}] {len(common)} matched (of CLAN={len(clan_map)}, SSAE={len(ssae_map)})")
        if missing_clan:
            print(f"  only in SSAE (no CLAN match): {missing_clan}")
        if missing_ssae:
            print(f"  only in CLAN (no SSAE match): {missing_ssae}")

        for key in common:
            out_name = f"seed{SEED}_{key}.png"
            merge_pair(
                clan_folder / clan_map[key],
                ssae_folder / ssae_map[key],
                out_folder / out_name,
                font,
            )
        print(f"  saved {len(common)} merged images -> {out_folder}")


if __name__ == "__main__":
    main()

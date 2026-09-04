"""Lightweight mirror of one MAE01ScaleMax run folder (res/mae01scale_max_se_<SEED>_t_<TIME>/).

Does NOT touch/delete anything under res/. Instead, for every dataset run
under that folder, COPIES ONLY:
    res/.../per_<X>/<dataset_hparam_folder>/checkpoints/best_auc/test_mode_total95.csv
into a new parallel tree:
    TH_lambda/mae01scale_max_se_<SEED>_t_<TIME>/per_<X>/<dataset_hparam_folder>/test_mode_total95.csv

so you only need to download the small TH_lambda/ folder, while res/ stays
exactly as-is on the Ubuntu machine.

Only edit SEED / TIME below, then run this script from inside Syn_ctgan/:
    python cleanup_keep_total95_bestauc.py

Safety:
  - DRY_RUN=True by default: prints exactly what would be copied, touches
    NOTHING on disk. Flip to False only after checking the printout.
  - Purely additive: only ever creates directories/files under TH_lambda/;
    never writes to or deletes anything under res/.
"""

import shutil
from pathlib import Path

# ── Edit these two, nothing else ────────────────────────────────────────────
SEED = 43
TIME = 2
# ─────────────────────────────────────────────────────────────────────────

DRY_RUN = False   # <-- flip to False only after reviewing the dry-run printout

BASE_DIR    = Path(__file__).parent
RES_DIR     = BASE_DIR / "res"
DEST_DIR    = BASE_DIR / "TH_lambda"
RUN_FOLDER  = f"mae01scale_max_se_{SEED}_t_{TIME:02d}"
RUN_ROOT    = RES_DIR / RUN_FOLDER

KEEP_NAME   = "test_mode_total95.csv"
SOURCE_REL  = Path("checkpoints") / "best_auc" / KEEP_NAME


def human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def process_run_dir(run_dir: Path):
    """run_dir = res/mae01scale_max_se_<SEED>_t_<TIME>/per_<X>/<dataset_hparam_folder>/"""
    src = run_dir / SOURCE_REL
    rel = run_dir.relative_to(RES_DIR)          # mae01scale_max_se_.../per_X/dataset_folder
    dst = DEST_DIR / rel / KEEP_NAME

    if not src.exists():
        print(f"  [SKIP] no {SOURCE_REL} under {rel} -- nothing copied")
        return

    if DRY_RUN:
        print(f"  [DRY]  {src.relative_to(BASE_DIR)}  ({human(src.stat().st_size)})")
        print(f"      -> {dst.relative_to(BASE_DIR)}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp_dst)
    if tmp_dst.stat().st_size != src.stat().st_size:
        tmp_dst.unlink(missing_ok=True)
        print(f"  [ERROR] copy size mismatch for {rel} -- skipped, res/ untouched.")
        return
    tmp_dst.replace(dst)
    print(f"  [OK]   {rel} -> {dst.relative_to(BASE_DIR)}  ({human(dst.stat().st_size)})")


def main():
    if not RUN_ROOT.exists():
        print(f"[ERROR] run folder not found: {RUN_ROOT}")
        return

    print(f"{'DRY RUN -- ' if DRY_RUN else ''}Scanning {RUN_ROOT} ...")
    print(f"Destination mirror root: {DEST_DIR}\n")

    run_dirs = sorted(p for p in RUN_ROOT.glob("per_*/*") if p.is_dir())
    if not run_dirs:
        print("  no per_*/<dataset_folder> entries found.")
        return

    for run_dir in run_dirs:
        process_run_dir(run_dir)

    if DRY_RUN:
        print("\nDry run only -- nothing was copied, res/ and TH_lambda/ both untouched. "
              "Set DRY_RUN = False to actually copy.")
    else:
        print(f"\nDone. res/ was never modified; copies live under {DEST_DIR}.")


if __name__ == "__main__":
    main()

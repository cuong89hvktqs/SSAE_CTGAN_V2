import os
import re
import pandas as pd

# ====== INPUT / OUTPUT ======
input_path = "/data/unsw/UNSW_NB15_testing-set.csv"   # file bạn đang có
output_dir = "/data/unsw/per_attack_test"            # thư mục lưu kết quả
os.makedirs(output_dir, exist_ok=True)

# ====== CONFIG: tên cột loại tấn công ======
# UNSW-NB15 thường là "attack_cat"
attack_col_candidates = ["attack_cat"]

df = pd.read_csv(input_path)

attack_col = None
for c in attack_col_candidates:
    if c in df.columns:
        attack_col = c
        break

if attack_col is None:
    raise ValueError(
        f"Không tìm thấy cột loại tấn công trong {attack_col_candidates}. "
        f"Các cột hiện có: {list(df.columns)}"
    )

print("Using attack column:", attack_col)

# ====== HÀM LÀM SẠCH TÊN FILE ======
def safe_filename(name: str, max_len: int = 120) -> str:
    name = str(name).strip()
    # thay mọi ký tự không an toàn bằng "_"
    name = re.sub(r"[^\w\-\.]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "UNKNOWN"
    return name[:max_len]

# ====== TÁCH VÀ LƯU ======
# Loại các giá trị NaN (nếu có) để khỏi tạo file "nan"
for attack_name, g in df.dropna(subset=[attack_col]).groupby(attack_col):
    fname = safe_filename(attack_name) + ".csv"
    out_path = os.path.join(output_dir, fname)
    g.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  | rows={len(g)}")

print("Done. Output dir:", output_dir)
# `run_mae01scale_max.bat` — giải thích tham số (cấu hình mặc định tốt nhất)

File này chạy `train_mae01scale_max.py` (MAE01ScaleMax) tuần tự trên nhiều dataset × 3 seed,
với một bộ `COMMON_ARGS` cố định. Đây là cấu hình đã được kiểm chứng (qua ablation `m`/`λ`
và kinh nghiệm thiết kế cơ chế gate/nhiễu) và đang dùng làm **mặc định tốt nhất** cho thuật toán.

```bat
set COMMON_ARGS=^
 --e 150^
 --b 128^
 --lae 0.1^
 --mae 2.0^
 --mae_loss_mode renormal^
 --type_train noise_std^
 --noise_std 0.3^
 --noise_type global^
 --noise_ratio 0.5^
 --cat_noise_type combined^
 --cat_noise_ratio 0.1^
 --patience 50^
 --noise_scale 1.5^
 --gate_stability 0.05^
 --gate_window 10^
 --cov_floor 0.05^
 --eps_zero 0.1^
 --tol 0.08^
 --per %PER%
```

## 1. Tham số cấp script (.bat, không thuộc `train_mae01scale_max.py`)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `DATASETS` | `unsw cic_iot nsl_kdd wsn_ds nb_iot cic_ids` | Danh sách dataset chạy tuần tự, mỗi dataset × 3 seed. |
| `PER` | `%~1`, mặc định `1.0` nếu không truyền | Tham số dòng lệnh đầu tiên khi gọi `.bat`; forward vào `--per` (tỷ lệ % dữ liệu train dùng, xem bên dưới). |
| `SEEDS` | `42 43 44` | Chạy 3 seed độc lập để lấy được mean ± std đáng tin cậy (đây là nguồn dữ liệu cho các bảng ablation/kết quả chính trong `TH/`). |
| `--time 4` | `4` (hardcode trong vòng lặp, không phải biến) | Gắn thẻ `t_04` vào tên thư mục kết quả (`res/mae01scale_max_se_<seed>_t_04/...`) — tách biệt với các batch thử nghiệm khác đang tồn tại song song (`t_02` = ablation m/λ, `t_03` = batch dùng để visualize) để không ghi đè lẫn nhau. |

## 2. Nhóm cơ bản (huấn luyện chung)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `--e` | `150` | Số epoch. Đủ để: CTGAN hội tụ → gate mở → M01AE ổn định, trên tất cả 6 dataset đã kiểm tra (xem `TH/latex_distance_to_centroid_visualize.tex`: các đường distance-to-centroid đều phẳng ổn định về cuối 150 epoch). |
| `--b` | `128` | Batch size cho cả CTGAN và M01AE (dùng chung 1 batch mỗi step). |
| `--per` | forward từ `PER` | Tỷ lệ % số dòng normal-train thực sự dùng để fit (0,1]. `1.0` = dùng toàn bộ. |

## 3. Nhóm Margin-based Autoencoder (M01AE): `m` và `λ`

| Tham số | Giá trị | Ký hiệu trong bài báo | Ý nghĩa |
|---|---|---|---|
| `--lae` | `0.1` | $\lambda$ | Hệ số cân bằng giữa reconstruction loss và margin loss: $\mathcal{L}_{MAE}=\mathcal{L}_{rec}+\lambda\,\mathcal{L}_{margin}$. |
| `--mae` | `2.0` | $m$ | Bán kính margin trong latent space: normal bị kéo về $\|z\|\to 0$, pseudo-anomaly bị kéo về $\|z\|\to m$. |
| `--mae_loss_mode` | `renormal` | — | Reconstruction loss **chỉ tính trên mẫu normal** ($y=0$), bỏ qua mẫu pseudo-anomaly khi tính RE loss — tránh việc autoencoder bị ép "học tái tạo tốt" cả dữ liệu bất thường giả, điều này sẽ làm giảm khả năng phân biệt. (Tùy chọn còn lại `origin` tính RE loss trên toàn batch.) |
| `--mae_lr` | *(không truyền, mặc định `1e-3`)* | — | Learning rate của optimizer M01AE. |

**Vì sao `m=2.0`, `λ=0.1`:** đây chính là 2 giá trị baseline đã được kiểm chứng qua ablation
(`TH/ablation_m_lambda_mean_std.xlsx`, phân tích chi tiết trong `TH/latex_impact_hyperparameters.tex`):
- `m=2.0` đạt AUC tốt nhất/gần tốt nhất ở **cả 4 dataset** test; `m=1` kém ổn định hơn (std cao),
  `m=9` luôn kém nhất (margin quá lớn làm mất ổn định reconstruction).
- `λ=0.1` là giá trị **duy nhất không bao giờ tệ nhất** ở bất kỳ dataset nào, trong khi mọi giá trị
  khác (0.01, 0.05, 0.5, 1.0) đều có ít nhất 1 dataset bị sụt rõ rệt.

## 4. Nhóm sinh nhiễu (cơ chế tạo pseudo-anomaly P và G)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `--type_train` | `noise_std` | Khi gate đã mở: trộn 50% P (real-normal + nhiễu) + 50% G (CTGAN-generated + nhiễu). Khi gate **chưa** mở: luôn fallback về P only, bất kể `type_train` là gì. |
| `--noise_std` | `0.3` | Độ lệch chuẩn nhiễu Gaussian **ban đầu** (khi gate đóng), áp lên các chiều liên tục (tanh-space, $\in[-1,1]$). **Lưu ý:** sau khi gate mở, giá trị này bị cơ chế z-norm self-scaling (`--noise_scale`) ghi đè hoàn toàn — `0.3` chỉ có hiệu lực trong giai đoạn warmup. |
| `--noise_type` | `global` | Cách áp nhiễu liên tục: `global` = cộng thẳng $\epsilon\sim\mathcal N(0,\sigma^2)$ vào một tập con chiều được chọn ngẫu nhiên rồi clip về $[-1,1]$ (khác `feature` — nhiễu theo std riêng từng cột, và `feature_outlier` — set thẳng theo outlier, không dùng ở đây). |
| `--noise_ratio` | `0.5` | Tỷ lệ chiều liên tục bị nhiễu mỗi lần (50%, chọn ngẫu nhiên) — giữ lại một phần đặc trưng gốc để pseudo-anomaly không quá lệch/quá dễ đoán. |
| `--cat_noise_type` | `combined` | Với cột categorical (one-hot span): 50/50 giữa `shuffle` (đảo span giữa các sample trong batch) và `flip` (đảo giá trị one-hot). |
| `--cat_noise_ratio` | `0.1` | Chỉ 10% số cột categorical bị nhiễu mỗi lần — giữ phần lớn thông tin phân loại nguyên vẹn (phân biệt với `noise_ratio` ở trên, vốn áp cho cột liên tục). |
| `--noise_r`, `--noise_k` | *(không truyền, mặc định `1.0`, `3.0`)* | Chỉ có tác dụng khi `--noise_type=feature_outlier`; **không dùng** trong cấu hình này vì `noise_type=global`. |

## 5. CTGAN Quality Gate + Adaptive Noise Scaling (đặc trưng riêng của "ScaleMax")

**Vấn đề:** lúc đầu train, CTGAN generator còn tệ → nếu dùng ngay dữ liệu nó sinh ra để làm
pseudo-anomaly G thì sẽ gây nhiễu loạn, phản tác dụng. **Giải pháp:** gate — chỉ cho phép dùng
nguồn G sau khi CTGAN đã sinh dữ liệu đủ ổn định.

| Tham số | Giá trị | Ký hiệu | Ý nghĩa |
|---|---|---|---|
| `--gate_window` | `10` | $W$ | Số epoch gần nhất dùng để tính EMA/CoV, **đồng thời** cũng là số epoch tối thiểu trước khi gate có thể mở (tránh mở gate quá sớm khi CTGAN chưa kịp học). |
| `--gate_stability` | `0.05` | $\delta$ | Ngưỡng hệ số biến thiên (CoV) để mở gate: $\mathrm{CoV}=\dfrac{\mathrm{std}(\overline{d}_{t-W:t})}{\max(|\mathrm{mean}(\overline{d}_{t-W:t})|,\ \phi)} \le \delta$, trong đó $\overline{d}_t$ là EMA của discriminator loss (không dùng generator loss vì G loss dao động suốt WGAN-GP, còn D loss hội tụ ổn định hơn). |
| `--cov_floor` | `0.05` | $\phi$ | Sàn tối thiểu cho mẫu số của CoV — **đặc trưng riêng của bản "Max"** so với `M01AECAdapt` gốc: một số dataset (VD NSL-KDD) có D-loss hội tụ về gần 0, khiến mẫu số CoV gần 0 → chia cho số rất nhỏ → gate không bao giờ mở đúng cách. `cov_floor` chặn mẫu số không xuống dưới `0.05`. |
| `--eps_zero` | `0.1` | $\epsilon_0$ | Điều kiện nhánh dự phòng "cân bằng gần-0": $|\mathrm{mean}(\overline{d}_{t-W:t})| \le \epsilon_0$. |
| `--tol` | `0.08` | $\tau$ | Điều kiện còn lại của nhánh dự phòng: $\mathrm{std}(\overline{d}_{t-W:t}) \le \tau$. Gate mở khi **CoV branch HOẶC nhánh gần-0** thoả mãn — nhánh gần-0 dành cho các dataset mà D-loss hội tụ phẳng ngay gần 0 (khiến điều kiện CoV chính khó kích hoạt dù mô hình đã ổn định). |
| `--noise_scale` | `1.5` | $\eta$ | Hệ số nhân trong công thức tự scale nhiễu **sau khi gate mở**: $\sigma_t=\eta\cdot\max\!\big(\mathrm{mean}\|z_{normal}\|,\ 0.05\,m\big)$ — noise_std lúc này không còn cố định `0.3` nữa mà tự điều chỉnh theo độ "co cụm" hiện tại của embedding normal (implicit curriculum: đầu huấn luyện embedding còn dàn trải → nhiễu lớn → pseudo-anomaly dễ; càng về sau embedding càng nén lại → nhiễu nhỏ dần → pseudo-anomaly khó dần, siết chặt biên quyết định). |

## 6. Huấn luyện / early stopping

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `--patience` | `50` | Số epoch liên tiếp không cải thiện `val_loss` trước khi dừng sớm (early stopping). Model vẫn lưu song song 2 checkpoint (`best_loss`, `best_auc`) trong suốt quá trình train — patience chỉ chi phối *thời điểm dừng*, không chi phối *checkpoint nào được chọn để báo cáo* (theo quy ước hiện tại: dùng `best_auc`, xem `TH/latex_mae01scale_max_algorithm_section.tex`). |

## 7. Tham số KHÔNG được truyền trong `.bat` (dùng mặc định của `train_mae01scale_max.py`)

| Tham số | Giá trị mặc định | Ý nghĩa |
|---|---|---|
| `--mae_lr` | `1e-3` | Learning rate của M01AE optimizer. |
| `--noise_r` | `1.0` | Chỉ dùng khi `noise_type=feature_outlier` (không áp dụng ở đây). |
| `--noise_k` | `3.0` | Tương tự, chỉ dùng khi `noise_type=feature_outlier`. |
| `--max_clusters` | `10` | Số mode Gaussian tối đa (Variational Gaussian Mixture) mà `DataTransformer` dùng cho mỗi cột liên tục — mỗi cột liên tục sau transform sẽ có `1 + max_clusters` chiều. |
| `--snapshot_stride` | `1` (riêng `nb_iot` mặc định `20`) | Tần suất lấy mẫu khi ghi `generated_data_snapshots.csv` (dùng cho các ảnh visualize t-SNE/distance-to-centroid), không ảnh hưởng đến quá trình huấn luyện. |
| `--n_max_samples` | `500000` | Giới hạn số mẫu tối đa cho các dataset lớn (`bot_iot`, `ton_iot`). |

## Tóm tắt lựa chọn thiết kế

Bộ tham số trong `run_mae01scale_max.bat` phản ánh 3 lớp quyết định:

1. **`m=2.0`, `λ=0.1`** — chọn từ kết quả ablation định lượng (mean ± std trên 3 seed × 4 dataset),
   không phải giá trị thắng tuyệt đối mọi cột, mà là giá trị **ổn định nhất khi tổng quát hoá**
   qua nhiều dataset khác nhau.
2. **Cơ chế nhiễu (P/G) và tỷ lệ nhiễu** (`noise_ratio=0.5`, `cat_noise_ratio=0.1`, `cat_noise_type=combined`)
   — chọn sao cho pseudo-anomaly đủ khác biệt để làm tín hiệu huấn luyện, nhưng không phá vỡ
   hoàn toàn cấu trúc gốc của dữ liệu.
3. **CTGAN Quality Gate + cov_floor + adaptive noise scaling** — lớp cơ chế đặc trưng riêng của
   "ScaleMax", giải quyết 2 vấn đề: (a) không dùng dữ liệu CTGAN sinh ra khi generator còn chưa
   ổn định, và (b) tránh gate kẹt-đóng-vĩnh-viễn trên các dataset có discriminator loss hội tụ
   gần 0 (VD NSL-KDD).

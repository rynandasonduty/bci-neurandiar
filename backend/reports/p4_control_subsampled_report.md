# Laporan Eksperimen Kontrol Subsampling P4 — Subjek S3, E0 Baseline, Fitur Barlow

Tanggal: 2026-07-08

## 1. Tujuan

Eksperimen kontrol ini memisahkan dua kemungkinan penyebab penurunan akurasi
pada uji pilot P4 No-Windowing (akurasi uji 0,00% pada 106 sampel latih,
dibandingkan champion P3 windowing standar 17,62% pada ~975 sampel latih):

1. **Jumlah data latih yang jauh lebih sedikit** (106 vs ~975), atau
2. **Struktur epoch tanpa-potongan itu sendiri** (5 detik penuh vs 5 jendela
   1 detik).

Caranya: melatih model **windowing standar** (struktur epoch identik dengan
champion P3), tetapi **sengaja dibatasi** memakai jumlah sampel yang sama
persis dengan pilot P4 (106 latih / 23 validasi / 23 uji). Jika akurasi
kontrol ini mendekati 0,00% (mendekati pilot P4), penyebab utamanya adalah
jumlah data. Jika akurasi kontrol ini mendekati 17,62% (mendekati champion),
penyebab utamanya adalah struktur epoch.

## 2. Metode

1. Dataset windowing penuh untuk S3/E0_Baseline/Barlow dibangun ulang
   menggunakan `WindowedReferenceDatasetBuilder`
   (`backend/src/preprocessing/windowed_reference_processor.py`, baru),
   yang memanggil `SignalProcessor.apply_filter()` dan
   `SignalProcessor.windowing_slot()` yang sudah ada tanpa modifikasi.
   Tidak ada augmentasi, tidak ada ablasi kanal, tidak ada crop ERP —
   identik dengan resep E0_Baseline yang dipakai P3.
2. Dataset penuh dibagi **satu kali** dengan `three_way_split(random_state=42)`
   (fungsi generik dari `utils/data_utils.py`, sama seperti dipakai P3),
   menghasilkan split train/val/test penuh.
3. Fitur Barlow (28 dimensi) diekstraksi sekali dari setiap split penuh
   menggunakan `EEGFeatureExtractor` yang sudah ada (tanpa modifikasi).
   Matriks fitur dan label lengkap ini menjadi sumber untuk disubset —
   **tidak ada model yang dilatih dari dataset penuh ini**.
4. Untuk lima seed subsampling independen (**42, 43, 44, 45, 46**, terpisah
   dari seed pembagian data awal di atas), diambil subset acak tanpa
   pengembalian sebesar 106 (latih) / 23 (validasi) / 23 (uji) dari masing-
   masing split penuh, menggunakan `numpy.random.RandomState(seed)`.
5. Untuk tiap seed, `StandardScaler` dilatih **hanya dari subset latih seed
   tersebut** (`utils/data_utils.fit_and_apply_scaler`), lalu SVM
   (`ClassicalClassifier(model_type='svm', C=10)`, kernel RBF — konfigurasi
   identik dengan champion P3) dilatih dan diuji pada subset tersebut.
6. Seluruh artefak (5 model, 5 scaler, 5 pasang Xtest/ytest, 1 ringkasan
   JSON) disimpan di `backend/models/weights/P4_Control_Subsampled/E0_Baseline/`,
   dengan nomor seed pada tiap nama file.

Skrip: `backend/src/models/run_p4_control_subsampled.py` (baru).

## 3. Verifikasi Rekonstruksi Dataset

Dataset windowing penuh yang dibangun ulang menghasilkan **1.394 sampel
total**, dengan split train=975, val=209, test=210. Angka test=210 ini
**cocok persis** dengan ukuran `Xtest_SVM_barlow_E0_Baseline_S3.npy` milik
P3 yang sudah ada di disk (dievaluasi pada laporan pilot P4 sebelumnya),
mengonfirmasi bahwa rekonstruksi dataset kontrol ini setara dengan dataset
asli yang dipakai untuk melatih champion P3.

## 4. Hasil Per-Seed

| Seed | N latih | N uji | Akurasi Uji | Akurasi Validasi | Cakupan Kelas (benar min. 1x) | Kelas hadir di subset latih |
|---|---|---|---|---|---|---|
| 42 | 106 | 23 | 4,3478% (1/23) | 13,0435% | 1/19 | 19/19 |
| 43 | 106 | 23 | 8,6957% (2/23) | 13,0435% | 2/19 | 19/19 |
| 44 | 106 | 23 | 8,6957% (2/23) | 8,6957% | 2/19 | 19/19 |
| 45 | 106 | 23 | **0,0000% (0/23)** | 17,3913% | 0/19 | 19/19 |
| 46 | 106 | 23 | 8,6957% (2/23) | 13,0435% | 2/19 | 19/19 |

Catatan: pada seluruh lima seed, subset latih (106 sampel) tetap memuat
minimal satu sampel dari seluruh 19 kelas — tidak ada kelas yang hilang
total dari data latih pada eksperimen kontrol ini.

## 5. Hasil Agregat (5 Seed)

- **Rerata akurasi uji: 6,0870%**
- **Simpangan baku (sampel, ddof=1): 3,8888 poin persentase**
- Simpangan baku (populasi, ddof=0): 3,4783 poin persentase
- Rentang: 0,0000% – 8,6957%

## 6. Distribusi Sampel per Kelas pada Subset Latih (106 sampel), per Seed

| Kelas (ID) | Suku Kata | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 |
|---|---|---|---|---|---|---|
| 0 | MA | 6 | 7 | 6 | 10 | 3 |
| 1 | KAN | 1 | 8 | 11 | 7 | 4 |
| 2 | MI | 5 | 4 | 3 | 5 | 8 |
| 3 | NUM | 9 | 3 | 2 | 6 | 2 |
| 4 | BE | 4 | 6 | 7 | 2 | 5 |
| 5 | RAK | 5 | 5 | 4 | 8 | 5 |
| 6 | PI | 7 | 6 | 5 | 4 | 11 |
| 7 | PIS | 5 | 5 | 4 | 3 | 3 |
| 8 | MAN | 3 | 1 | 7 | 4 | 7 |
| 9 | DI | 5 | 3 | 7 | 5 | 5 |
| 10 | BO | 7 | 5 | 5 | 4 | 6 |
| 11 | SAN | 4 | 8 | 8 | 5 | 2 |
| 12 | LE | 10 | 6 | 6 | 2 | 5 |
| 13 | LAH | 5 | 9 | 6 | 3 | 10 |
| 14 | SA | 9 | 8 | 10 | 12 | 9 |
| 15 | KIT | 4 | 5 | 6 | 5 | 5 |
| 16 | TI | 6 | 7 | 2 | 10 | 7 |
| 17 | DUR | 6 | 4 | 4 | 2 | 5 |
| 18 | YANG | 5 | 6 | 3 | 9 | 4 |

Rata-rata hanya sekitar **5-6 sampel latih per kelas** dari 19 kelas —
jauh di bawah jumlah yang biasanya dibutuhkan untuk melatih SVM multi-kelas
yang stabil, konsisten dengan pilot P4 (yang juga hanya ~5-6 sampel/kelas).

## 7. Tabel Perbandingan Tiga Angka

| Kondisi | Struktur Epoch | N Sampel Latih | Akurasi Uji |
|---|---|---|---|
| **Champion P3 asli** | Windowing (1 detik) | ~975 | **17,6190%** |
| **Kontrol Tersubsampling P4** (rerata ± sd, 5 seed) | Windowing (1 detik) | 106 | **6,0870% ± 3,8888 pp** |
| **Pilot P4 No-Windowing** | Tanpa-potongan (5 detik) | 106 | **0,0000%** |

## 8. Analisis dan Kesimpulan

**Efek jumlah data (windowing 975 vs windowing 106 tersubsampling):**
Membatasi dataset windowing standar dari ~975 sampel latih menjadi hanya 106
sampel latih — tanpa mengubah struktur epoch sama sekali — sudah menyebabkan
penurunan akurasi uji yang besar, dari 17,62% menjadi rerata 6,09% (turun
menjadi sekitar sepertiga dari akurasi champion). Ini menunjukkan bahwa
**jumlah data latih saja sudah menjelaskan sebagian besar dari total celah
akurasi** antara champion P3 dan pilot P4.

**Efek struktur epoch tambahan (windowing 106 tersubsampling vs no-windowing
106):**
Selisih antara rerata kontrol tersubsampling (6,09%) dan pilot P4
No-Windowing (0,00%) adalah 6,09 poin persentase. Namun, dengan simpangan
baku antar-seed sebesar 3,89 poin persentase, selisih ini setara dengan
skor-z sekitar **-1,57** — berada dalam rentang variasi acak subset kecil
(23 sampel uji) dan **tidak mencapai ambang signifikansi statistik yang
umum digunakan (|z| > 1,96)**. Lebih jelas lagi, satu dari lima seed kontrol
(seed 45) juga mencapai akurasi 0,00% persis sama dengan pilot P4 —
membuktikan secara langsung bahwa akurasi 0,00% dapat muncul murni dari
variasi pengambilan subset acak pada pendekatan windowing standar, tanpa
perlu melibatkan perubahan struktur epoch apa pun.

**Kesimpulan akhir**: berdasarkan data yang diperoleh, penurunan akurasi
pada pilot P4 No-Windowing **terutama disebabkan oleh jumlah sampel latih
yang sangat sedikit (106 sampel, ~5-6 sampel/kelas)**, bukan oleh struktur
epoch tanpa-potongan itu sendiri. Eksperimen kontrol ini **tidak menemukan
bukti statistik yang jelas** bahwa struktur epoch tanpa-potongan memiliki
efek negatif tambahan di luar efek jumlah data — meskipun efek tambahan yang
kecil tetap tidak dapat sepenuhnya disingkirkan, mengingat pilot P4 berada
di ujung bawah (bukan di luar) rentang hasil kontrol yang teramati, dan
ukuran sampel eksperimen ini (5 seed, 23 sampel uji per seed) masih kecil
untuk membedakan efek sekecil itu secara pasti.

Hipotesis awal P4 — bahwa epoch tanpa-potongan akan meningkatkan akurasi
karena menghindari pencampuran fase persiapan dan fase puncak readiness
potential — **tidak didukung oleh data pilot maupun kontrol ini**. Sebelum
menguji hipotesis tersebut secara adil, jumlah data latih untuk pendekatan
tanpa-potongan perlu ditingkatkan secara substansial (misalnya melonggarkan
ambang penolakan artefak untuk epoch 5 detik, atau mengumpulkan lebih
banyak trial per subjek), karena pada skala data saat ini kedua pendekatan
(windowing maupun tanpa-potongan) sama-sama gagal melampaui tingkat peluang
acak (5,26%) ketika dibatasi pada ~106 sampel latih untuk 19 kelas.

## 9. Artefak yang Dihasilkan

- `backend/src/preprocessing/windowed_reference_processor.py` — kelas
  `WindowedReferenceDatasetBuilder` (baru, tidak mengubah `signal_processor.py`
  maupun `build_dataset.py`).
- `backend/src/models/run_p4_control_subsampled.py` — skrip eksperimen
  kontrol subsampling (baru).
- `backend/models/weights/P4_Control_Subsampled/E0_Baseline/`:
  - `SVM_P4_Control_barlow_E0_Baseline_S3_seed{42..46}.pkl` — 5 model SVM.
  - `scaler_P4_Control_SVM_barlow_E0_Baseline_S3_seed{42..46}.pkl` — 5 scaler.
  - `Xtest_P4_Control_SVM_barlow_E0_Baseline_S3_seed{42..46}.npy`,
    `ytest_P4_Control_SVM_barlow_E0_Baseline_S3_seed{42..46}.npy` — data uji
    tersimpan per seed.
  - `results_P4_Control_Subsampled_barlow_E0_Baseline_S3.json` — ringkasan
    numerik lengkap per-seed dan agregat.
- `backend/reports/p4_control_subsampled_report.md` — laporan ini.

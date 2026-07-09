# Laporan Uji Pilot P4 (No-Windowing) — Subjek S3, E0 Baseline, Fitur Barlow

Tanggal: 2026-07-08

## 1. Tujuan dan Cakupan

Uji pilot ini menguji hipotesis: apakah melatih model SVM dari epoch 5 detik
penuh (tanpa dipotong menjadi 5 jendela 1 detik seperti pada P1/P2/P3)
menghasilkan akurasi lebih baik dibandingkan pendekatan windowing standar.

Cakupan pilot dibatasi secara sengaja sesuai arahan:
- Subjek: **S3 saja**
- Konfigurasi: setara **E0 Baseline** (band-pass 0,5-50 Hz, tanpa augmentasi)
- Fitur: **Barlow saja** (28 dimensi = 2 fitur x 14 kanal)

## 2. Metode

- Epoch 5 detik penuh diekstraksi dari setiap slot marker menggunakan fungsi
  baru `extract_full_epoch()` (`backend/src/preprocessing/full_epoch_processor.py`),
  yang menerapkan filter band-pass identik dengan P1/P2/P3 (memanggil
  `SignalProcessor.apply_filter()` yang sudah ada, tanpa modifikasi) dan
  ambang penolakan artefak ~100 microvolt yang sama (`SignalProcessor.reject_artifacts()`),
  namun dievaluasi **satu kali atas seluruh 5 detik**, bukan per jendela
  1 detik seperti `windowing_slot()`.
- Fitur Barlow diekstraksi menggunakan `EEGFeatureExtractor` yang sudah ada
  dari `extract_eeg_features.py` (tidak dimodifikasi; fungsi ini generik
  terhadap panjang epoch).
- Pembagian data train/val/test memakai `three_way_split()` (70/15/15,
  stratified) dan `StandardScaler` yang dilatih hanya dari data latih,
  keduanya dari `utils/data_utils.py` (utilitas generik yang sudah dipakai
  seluruh paradigma, tidak dimodifikasi).
- Model: `ClassicalClassifier(model_type='svm', C=10)` (SVM kernel RBF),
  konfigurasi identik dengan yang dipakai P3.
- Seluruh artefak (model, scaler, Xtest/ytest, hasil JSON) disimpan di
  `backend/models/weights/P4_NoWindowing/E0_Baseline/`.

## 3. Hasil

### 3.1 Jumlah Sampel

| Pendekatan | Total sampel | Train | Val | Test |
|---|---|---|---|---|
| **P4 No-Windowing (pilot ini)** | **152** | 106 | 23 | 23 |
| P3 Windowing standar (S3/E0/Barlow), test set tersimpan | tidak diketahui persis*| tidak tersimpan | tidak tersimpan | **210** |

\* Hanya `Xtest`/`ytest` yang disimpan oleh pipeline windowing yang sudah ada
(`run_e8_classical.py` tidak menyimpan split train/val ke disk). Dengan rasio
split tetap 15% untuk test (`three_way_split`), total sampel windowing dapat
diestimasi sekitar 210 / 0,15 ≈ **1.400 sampel** — namun ini estimasi dari
rasio, bukan penghitungan ulang langsung.

Pendekatan tanpa-potongan menghasilkan sampel jauh lebih sedikit dari
windowing, sesuai dugaan: ambang artefak (~100 microvolt) dievaluasi atas
seluruh 5 detik penuh sekaligus (bukan per jendela 1 detik terpisah),
sehingga satu lonjakan artefak di mana pun dalam slot 5 detik menggugurkan
seluruh epoch, alih-alih hanya satu dari lima jendela.

### 3.2 Akurasi

| Pendekatan | Akurasi Uji (Test Accuracy) | Sampel Uji |
|---|---|---|
| **P4 No-Windowing (pilot ini)** | **0,0000%** (0/23 benar) | 23 |
| P3 Windowing standar (S3/E0_Baseline/Barlow, dievaluasi ulang dari artefak tersimpan) | **17,6190%** (37/210 benar) | 210 |

Catatan: angka pembanding 17,6190% **bukan angka yang sudah dipublikasikan**
di laporan mana pun dalam repositori ini sebelumnya — angka ini dihitung
langsung dari model SVM dan test split E0_Baseline/S3/Barlow yang sudah ada
di `backend/models/weights/P3_SVM/E0_Baseline/` (tanpa retraining, tanpa
mengubah file P3 apa pun), untuk memastikan pembandingan yang jujur dan
dapat diverifikasi. Angka ini konsisten dengan referensi "17,62%" yang
diberikan sebagai acuan di awal tugas.

Akurasi validasi (val set, 23 sampel) untuk P4 pilot: **13,0435%** (3/23 benar).

Sebagai konteks, tingkat peluang acak (chance level) untuk klasifikasi 19
kelas adalah 1/19 ≈ **5,26%**. Akurasi uji P4 pilot (0,00%) berada di bawah
tingkat peluang acak.

### 3.3 Cakupan Kelas

Definisi cakupan kelas mengikuti definisi yang sudah dipakai pada matriks
juara P3 (`T5_champion_grand_matrix.csv`): jumlah kelas yang berhasil
diprediksi **dengan benar** minimal sekali pada test set (bukan sekadar
kelas yang pernah ditebak oleh model).

| Pendekatan | Kelas benar minimal sekali | Kelas yang pernah ditebak (benar atau salah) |
|---|---|---|
| **P4 No-Windowing (pilot ini)** | **0/19** | 13/19 |
| P3 Windowing standar (S3/E0_Baseline/Barlow) | **17/19** | tidak dihitung |

Model SVM pada pilot P4 tetap menebak 13 kelas berbeda dari 23 prediksi
(bukan hanya menebak satu kelas berulang-ulang), namun tidak satu pun dari
tebakan tersebut cocok dengan label sebenarnya pada data uji.

## 4. Kesimpulan

Hasil uji pilot ini **tidak mendukung hipotesis** bahwa epoch 5 detik penuh
tanpa windowing menghasilkan akurasi lebih baik dibandingkan pendekatan
windowing standar. Akurasi uji turun dari 17,6190% (windowing) menjadi
0,0000% (tanpa-potongan) — sebuah **penurunan signifikan**, bukan
peningkatan atau hasil yang setara.

Namun, hasil ini harus dibaca dengan hati-hati karena ada perancu (confound)
yang jelas: pendekatan tanpa-potongan menghasilkan sampel jauh lebih sedikit
(152 total, hanya 106 untuk melatih model 19-kelas) dibandingkan pendekatan
windowing (~1.400 sampel, estimasi). Dengan hanya ~5-6 sampel latih rata-rata
per kelas, SVM pada pilot ini kemungkinan besar mengalami *data starvation*,
sehingga penurunan akurasi yang teramati bisa jadi lebih disebabkan oleh
kekurangan data ekstrem, bukan murni oleh perubahan struktur epoch itu
sendiri. Pilot satu-subjek, satu-fold ini juga tidak memberikan estimasi
varians (tidak ada pengulangan lintas subjek atau lintas seed).

**Rekomendasi**: berdasarkan data pilot ini saja, perluasan ke skala penuh
(12 subjek x 8 ablasi x 5 kelompok fitur) **belum layak dilanjutkan tanpa
perubahan desain lebih lanjut**, karena masalah data starvation yang
teramati kemungkinan akan berulang pada subjek lain dan berpotensi
menghasilkan model yang tidak dapat dilatih secara bermakna pada sebagian
besar konfigurasi. Keputusan akhir mengenai arah lanjutan pilot ini
diserahkan kepada pengguna.

## 5. Artefak yang Dihasilkan

- `backend/src/preprocessing/full_epoch_processor.py` — `extract_full_epoch()`
  dan `FullEpochDatasetBuilder` (baru, tidak mengubah `signal_processor.py`).
- `backend/src/models/run_p4_no_windowing.py` — skrip pelatihan pilot P4 (baru).
- `backend/models/weights/P4_NoWindowing/E0_Baseline/`:
  - `SVM_P4_NoWindowing_barlow_E0_Baseline_S3.pkl` — model SVM terlatih.
  - `scaler_P4_NoWindowing_SVM_barlow_E0_Baseline_S3.pkl` — StandardScaler.
  - `Xtest_P4_NoWindowing_SVM_barlow_E0_Baseline_S3.npy`,
    `ytest_P4_NoWindowing_SVM_barlow_E0_Baseline_S3.npy` — data uji tersimpan.
  - `results_P4_NoWindowing_barlow_E0_Baseline_S3.json` — ringkasan hasil numerik.
- `backend/reports/p4_no_windowing_pilot_report.md` — laporan ini.

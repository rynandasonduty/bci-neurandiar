# P7 Coarse Sub-model Ablation Pipeline -- Implementation Summary

Status: **kode saja, belum dijalankan.** Sesuai instruksi eksplisit, tidak ada `python run_*.py`,
smoke test, dry-run, atau eksekusi notebook yang dilakukan dalam sesi ini. Verifikasi dilakukan lewat
(1) `python -m py_compile` pada seluruh file baru (cek sintaks murni, tidak menjalankan logika), dan
(2) audit manual referensi lintas-modul (memastikan setiap nama fungsi/konstanta yang diimpor benar-benar
ada di modul sumbernya). Seluruh training sungguhan (P6 penuh maupun rangkaian ablation P7) akan
dijalankan peneliti di PC Lab.

## 1. Daftar File Baru dan Fungsinya

| File | Bagian | Fungsi |
|---|---|---|
| `run_p6_transfer_overt_imagined.py` | 1 | **Sudah diperbaiki sesi sebelumnya, diverifikasi ulang sesi ini** -- `run_fullscale()` mengembalikan ringkasan (`n_total`/`n_processed`/`n_skipped_missing_baseline`); `__main__` exit 1 jika seluruh subjek gagal (baseline hilang), exit 0 jika sebagian (auto-resume normal), exception tak tertangani tetap exit non-zero. Audit path hardcoded: **tidak ditemukan** path absolut di seluruh `experiments_p4_p7/` (sudah relatif berbasis `__file__` sejak sesi sebelumnya). |
| `p7_coarse_cache.py` | 2 | Cache raw 3D epoch (coarse-filtered, 9 kelas) + fitur Barlow (belum di-scale) + label suku kata mentah, per subjek, ke `P7_CoarseToFine/_cache/{subject_id}/`. `get_or_build_cached_coarse(subject_id)` -- load jika ada, build sekali jika belum. Inilah yang membuat Fase 1 (5 varian) hanya membayar biaya baca-CSV-mentah SATU KALI per subjek, bukan 5 kali. |
| `classical_models_ext.py` | 3 (Varian A) | `WeightedClassicalClassifier` -- subclass `ClassicalClassifier`, override `__init__` saja untuk menambah `class_weight='balanced'` ke SVC. `classical_models.py` tidak disentuh. |
| `run_p7_coarse_ablation.py` | 3 | Fase 1: melatih Varian A (balanced), B (augmented E5), C (C-tuned per subjek), D (fitur `all`), E (dikalibrasi) -- masing-masing dibandingkan ke baseline `coarse` yang sudah ada. Menulis `P7_CoarseAblation_Phase1_report.md` dan `phase1_summary.json`. |
| `run_p7_coarse_combined.py` | 4 | Fase 2: baca `phase1_summary.json`, terapkan aturan inklusi otomatis (delta > 1.0pp; Varian E selalu disertakan struktural), latih SATU model coarse final per subjek yang menggabungkan faktor-faktor lolos, simpan ke `coarse_final_combined/`. Tambahkan section "Fase 2" ke laporan yang sama. Juga mendefinisikan `SoftPredictBundle` (dipakai ulang Fase 3/4). |
| `run_p7_postprocessing.py` | 5 | Fase 3: confidence-gated coarse->fine (threshold dituning di VAL, dievaluasi sekali di test) dan ensemble voting (Varian A + Varian C + kombinasi final). Bandingkan 3 strategi di level end-to-end, pilih otomatis yang tertinggi. Tambahkan section "Fase 3" ke laporan yang sama. |
| `run_p7_final_integration.py` | 6 | Fase 4: hitung ulang KEDUA metrik end-to-end (suku kata pertama, kata penuh) memakai strategi pemenang Fase 3 + `fine_A/I/E/sa_branch` yang tidak berubah. Tulis section besar baru "Ablation Study & Final Combined Model" ke `P7_CoarseToFine_report.md` (laporan ASLI, tidak dihapus) berisi seluruh tabel Fase 1-3 plus tabel before/after Fase 4 plus perbandingan champion P3 dengan catatan kejujuran. |
| `run_followup_orchestrator.py` | -- | Orkestrator baru: P6 -> cache warm-up -> Fase 1 -> Fase 2 -> Fase 3 -> Fase 4 -> notebook, sekali jalan, dengan flag skip per tahap. |
| `P4_P7_Analysis.ipynb` | 6 | 4 sel baru ditambahkan setelah section P7 yang sudah ada (section baru, TIDAK menghapus section P7 asli): load `phase4_final_summary.json`, tabel before/after, uji Wilcoxon, plot -- dengan pesan informatif "belum memiliki hasil" jika file belum ada, mengikuti pola yang sudah ada di sel-sel lain. |

File yang **dihapus** (bukan dimodifikasi): `run_p7_coarse_improve.py` (sesi sebelumnya) -- digantikan
sepenuhnya oleh rangkaian ablation baru yang jauh lebih lengkap (5 varian vs. 2 varian sebelumnya, plus
Fase 2-4). File lama itu tidak pernah dijalankan sama sekali (belum ada artefak yang bergantung padanya),
jadi aman dihapus tanpa kehilangan hasil apa pun.

## 2. Alur Data (Ringkas per File)

- **`p7_coarse_cache.py`**: input = CSV mentah + log eksperimen (`RAW_DATA_DIR`) lewat
  `run_p7_coarse_to_fine.build_standard_e0_split_raw()` (baca saja). Transformasi:
  `three_way_split(seed=42)` (identik dengan baseline) -> filter ke 9 kelas suku kata pertama
  (`filter_split_by_labels`) -> mapping ke id kelompok vokal (`map_labels_to_vowel_group_ids`) ->
  ekstraksi fitur Barlow. Output: 12 array `.npy` per subjek (3D mentah x3, label vokal x3, label
  suku kata mentah x3, fitur Barlow x3) ke `P7_CoarseToFine/_cache/{subject}/`.
- **`run_p7_coarse_ablation.py`**: input = cache di atas + baseline `results_{subject}.json` yang
  sudah ada. Transformasi: 5 jalur pelatihan independen (lihat tabel varian di atas), semuanya
  dievaluasi ke split test yang SAMA. Output: model+scaler+Xtest/ytest per varian per subjek ke
  `coarse_variant_{a..e}_*/`, plus `phase1_comparison_{subject}.json`, `phase1_summary.json`,
  `P7_CoarseAblation_Phase1_report.md`.
- **`run_p7_coarse_combined.py`**: input = `phase1_summary.json` + cache. Transformasi: aturan
  inklusi (>1pp) -> komposisi faktor -> satu model `CalibratedClassifierCV` per subjek. Output:
  `coarse_final_combined/` (model+scaler+Xtest/ytest+`results_{subject}.json`),
  `phase2_summary.json`, section "Fase 2" ditambahkan ke laporan Fase 1.
- **`run_p7_postprocessing.py`**: input = model kombinasi Fase 2 + Varian A/C (Fase 1) + `fine_A/I/E`
  yang sudah ada (semua read-only via `SoftPredictBundle`) + cache (raw epoch val/test + label suku
  kata mentah). Transformasi: threshold tuning di VAL (5 nilai), evaluasi 3 strategi di TEST. Output:
  `phase3_postprocessing_{subject}.json`, `phase3_summary.json`, section "Fase 3" ditambahkan ke laporan.
- **`run_p7_final_integration.py`**: input = `phase3_summary.json` (strategi pemenang) + model Fase 2
  + `fine_A/I/E/sa_branch` + `OfflineTrialReader` (raw CSV, hanya untuk rekonstruksi trial kata
  penuh -- bukan re-training). Transformasi: prediksi suku-kata-pertama & kata-penuh pakai strategi
  pemenang. Output: `final_e2e_{subject}.json`, `phase4_final_summary.json`, section besar baru di
  `P7_CoarseToFine_report.md` (laporan asli, ditambah bukan diganti).
- **`run_followup_orchestrator.py`**: tidak memproses data sendiri -- murni menjalankan 7 subprocess
  di atas berurutan, mencatat SUCCESS/FAILED per tahap ke `followup_orchestrator_run_log.md`, lanjut
  meski satu tahap gagal, exit 1 di akhir jika ada yang gagal.

## 3. Urutan Menjalankan di PC Lab

Sync kode dulu (`git pull`), lalu **satu perintah**:

```bash
cd backend/src/experiments_p4_p7
python run_followup_orchestrator.py
```

Atau manual satu-satu (urutan WAJIB persis seperti ini -- setiap tahap bergantung pada file summary
dari tahap sebelumnya, kecuali P6 yang independen dan boleh kapan saja):

```bash
cd backend/src/experiments_p4_p7

python run_p6_transfer_overt_imagined.py              # Bagian 1 -- independen, boleh kapan saja

python p7_coarse_cache.py                              # Bagian 2 -- opsional (Fase 1 auto-build jika dilewati)
python run_p7_coarse_ablation.py                        # Bagian 3 -- Fase 1
python run_p7_coarse_combined.py                        # Bagian 4 -- Fase 2 (butuh phase1_summary.json)
python run_p7_postprocessing.py                         # Bagian 5 -- Fase 3 (butuh phase2_summary.json)
python run_p7_final_integration.py                      # Bagian 6 -- Fase 4 (butuh phase3_summary.json)

cd ../../../notebooks
jupyter nbconvert --to notebook --execute --inplace P4_P7_Analysis.ipynb --ExecutePreprocessor.timeout=1800
```

Jika environment untuk `run_p*.py` (perlu sklearn/numpy/pandas/scipy) berbeda dari environment yang
punya jupyter+nbconvert (seperti di mesin dev tempat kode ini ditulis: `backend/venv` punya sklearn
tapi tidak jupyter, `.venv` di root punya keduanya), pakai:

```bash
python run_followup_orchestrator.py --notebook-python "/path/to/python/yang/punya/jupyter"
```

## 4. Estimasi Waktu (asumsi caching bekerja sesuai desain, TIDAK diverifikasi lewat eksekusi nyata)

| Tahap | Estimasi | Alasan |
|---|---|---|
| P6 (Bagian 1) | ~15-45 menit | 12 subjek, masing-masing 1x baca CSV mentah (imagined+overt) + 1x training SVM. Skala sama seperti P7 Stage B asli. |
| Cache warm-up (Bagian 2) | ~15-40 menit | 12 subjek x 1x baca CSV mentah (biaya dominan, sama seperti build Stage B P7 asli). Jika dilewati, biaya ini pindah ke awal Fase 1 (bukan hilang). |
| Fase 1 -- Ablation (Bagian 3) | ~10-25 menit | Cache sudah hangat -- hanya ekstraksi fitur ringan (Varian B/D) + training SVM kecil x banyak (Varian C = 5x per subjek, Varian E = 5-fold CV = 5x per subjek). Tidak ada baca CSV lagi. |
| Fase 2 -- Kombinasi (Bagian 4) | ~5-15 menit | 1 model `CalibratedClassifierCV` (5-fold internal) per subjek, fitur dari cache (kecuali jika Varian D lolos, sedikit ekstraksi tambahan). |
| Fase 3 -- Post-processing (Bagian 5) | ~10-20 menit | Tidak ada training -- inferensi per-sampel (threshold grid x val set, 3 strategi x test set) memakai model yang sudah ada. Didominasi jumlah panggilan `predict`/`predict_proba` per epoch dalam loop Python, bukan I/O. |
| Fase 4 -- Final Integration (Bagian 6) | ~10-20 menit | Rekonstruksi trial kata-penuh via `OfflineTrialReader` (baca CSV mentah lagi, tapi hanya untuk daftar trial+parsing, bukan windowing penuh) + inferensi. |
| Notebook | < 5 menit | Hanya baca JSON + inferensi read-only ke model P1-P3 yang sudah ada, tidak training apa pun. |
| **Total** | **~1.5-3 jam** | Di bawah target 2-5 jam dan jauh di bawah budget 24 jam, dengan asumsi caching bekerja seperti dirancang. Jika ternyata jauh melebihi ini, kemungkinan besar cache TIDAK ter-reuse dengan benar (`p7_coarse_cache.py` mem-build ulang tiap subjek) -- ini adalah tanda untuk diperiksa, bukan hal yang diharapkan terjadi. |

## 5. Verifikasi yang Dilakukan (Bukan Eksekusi)

1. `python -m py_compile` pada seluruh 8 file baru/relevan -- **semua lolos, tidak ada syntax error.**
2. Audit manual: setiap nama fungsi/konstanta yang di-`import` lintas modul (`p7base`, `ablation`,
   `combined`, `postproc`) ditelusuri balik ke definisi aslinya untuk memastikan tidak ada typo/nama
   yang tidak ada -- termasuk pengecekan setelah rename `_baseline_svc_pipeline` ->
   `baseline_svc_pipeline` dan `_classes_covered` -> `classes_covered_from_predictions` (semula
   private, diubah jadi publik karena dipakai lintas file).
3. Validasi JSON notebook (`json.load` murni, tidak menjalankan sel apa pun) -- struktur 27 sel valid,
   4 sel baru berada persis di antara section P7 asli dan "Final Summary".
4. `git status`/`git diff --stat` -- dikonfirmasi hanya file di `experiments_p4_p7/` dan notebook yang
   tersentuh; P1/P2/P3, P4, P5, dan ketujuh file terlarang (`signal_processor.py`, `build_dataset.py`,
   `extract_eeg_features.py`, `classical_models.py`, `data_utils.py`, `transfer_learning.py`,
   `main.py`) -- nol perubahan, dikonfirmasi lewat `git diff --stat` eksplisit terhadap masing-masing.

**Belum diverifikasi lewat eksekusi nyata** (menunggu PC Lab): apakah `CalibratedClassifierCV`
benar-benar bisa fit untuk setiap subjek (butuh cukup sampel per kelas untuk 5-fold CV -- lihat
catatan di docstring `train_variant_e`), apakah performa memori/waktu sesuai estimasi Bagian 4 di
atas, dan tentu saja seluruh angka akurasi/p-value yang menjadi tujuan utama pipeline ini.

## 6. git status / git diff --stat (Sesi Ini)

```
 Changes not staged for commit:
	deleted:    backend/src/experiments_p4_p7/run_p7_coarse_improve.py
	modified:   notebooks/P4_P7_Analysis.ipynb

 Untracked files:
	backend/src/experiments_p4_p7/p7_coarse_cache.py
	backend/src/experiments_p4_p7/run_followup_orchestrator.py
	backend/src/experiments_p4_p7/run_p7_coarse_ablation.py
	backend/src/experiments_p4_p7/run_p7_coarse_combined.py
	backend/src/experiments_p4_p7/run_p7_final_integration.py
	backend/src/experiments_p4_p7/run_p7_postprocessing.py

 2 files changed, 31 insertions(+), 541 deletions(-)  (deleted file + notebook cell inserts)
```

`run_p6_transfer_overt_imagined.py` dan `classical_models_ext.py` tidak muncul di diff sesi ini karena
sudah di-commit otomatis pada sesi sebelumnya (`bc4de62`) dan tidak diubah lagi sekarang -- keduanya
diverifikasi ulang (dibaca, bukan diedit) dan sudah benar.

Belum di-commit sesuai permintaan (peneliti akan commit sendiri).

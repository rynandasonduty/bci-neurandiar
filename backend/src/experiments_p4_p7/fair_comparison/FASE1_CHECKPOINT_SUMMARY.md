# FASE 1 CHECKPOINT SUMMARY — P3 vs P6 Fair Comparison (Plan B, Poin 0)

**Status: Fase 1 SELESAI, semua 11 task berjalan sukses. Dokumen ini adalah checkpoint wajib —
STOP di sini, menunggu review dan konfirmasi peneliti sebelum Fase 2 (revisi notebook) dimulai.**

Semua script ada di `backend/src/experiments_p4_p7/fair_comparison/`. Semua output JSON/MD ada di
`backend/src/experiments_p4_p7/fair_comparison/results/`. **Tidak ada training/retraining apa pun
yang dijalankan** — seluruh task ini adalah load artefak tersimpan + inference/analisis, sesuai
ATURAN #0.

**Verifikasi `git status` (dijalankan eksplisit, bukan asumsi):** **nol file `.py` yang
termodifikasi** — semua 12 file Python baru untuk Fase 1 muncul sebagai untracked (`??`), tidak ada
satu pun file kode lama (P1-P3, P4/P5 base training, word assembler, sentence refiner, P6/P7 base
scripts) yang berubah. Dua file laporan `.md` (`P7_Ablation_Implementation_Summary.md`,
`P7_CoarseToFine_report.md`) memang muncul sebagai *modified*, TAPI ini adalah state
**uncommitted dari SEBELUM sesi ini dimulai** (commit terakhir yang menyentuh file itu: 2026-07-10,
2 hari sebelum sesi ini) — tidak ada Write/Edit tool yang dipakai pada file itu di sesi ini, dan
tidak ada script Fase 1 yang memanggil `write_report()`-nya `run_p7_coarse_to_fine.py`. Dilaporkan
di sini demi kejujuran, bukan disembunyikan, tapi bukan hasil kerja Fase 1 ini.

---

## 1. Status Tiap Task

| Task | Status | Catatan |
|---|---|---|
| 1.1 P3 first-syllable-only accuracy | **Selesai** | Dihitung baru dari artefak tersimpan (model+Xtest/ytest champion P3). |
| 1.2 P6 class coverage check | **Selesai** | 100% dari membaca ulang `results_S{n}.json` yang sudah ada — tidak ada model di-load ulang, sesuai ekspektasi prompt. |
| 1.3 Sanity check Stage B baseline | **Selesai, LENGKAP** (termasuk Part C) | Lihat detail di bawah — semua angka Stage B TERVERIFIKASI genuine. |
| 1.4 Latency measurement | **Selesai** | Pengukuran REAL (bukan cuma proyeksi teoritis) berhasil dilakukan untuk S3. Ditemukan isu metodologi penting — lihat Section 6. |
| 1.5 Error decomposition | **Selesai** | S3 saja (sesuai scope prompt), n=36 trial test full-word. |
| 1.6 Theoretical ceiling | **Selesai** | S3 + 12 subjek (murah, dari data tersimpan). |
| 1.7 Feature importance cross-architecture | **Selesai** | S3, permutation_importance n_repeats=30 (metodologi sama persis dengan P3). |
| 1.8 Calibration comparison | **Selesai** | S3, Brier+ECE dihitung untuk P3 champion DAN P6 coarse baseline (keduanya `probability=True` native, tidak perlu CalibratedClassifierCV). |
| 1.9 Stage-1 vs e2e consistency | **Selesai** | 12 subjek, murni dari tabel yang sudah ada. |
| 1.10 MCDA scoring table | **Selesai** | Skor mentah tersaji, **TANPA keputusan champion final** (menunggu konfirmasi bobot dari peneliti). |
| 1.11 Quantitative skip justification (P1/P2/P4/P5) | **Selesai** | Semua 4 paradigma, murni dari CSV/JSON yang sudah ada. |

**Tidak ada task yang gagal atau butuh data tambahan.** Semua artefak yang dibutuhkan (model P3
champion, model+Xtest/ytest 5 sub-model P6 x 12 subjek, `results_S{n}.json` P4/P5/P6, tabel-tabel
`T*.csv`) sudah tersedia di repo ini, persis seperti diklaim di ATURAN #0.

---

## 2. Tabel Final 3-Tier: P3 vs P6

| Tier | Definisi | P3 | P6 | Selisih |
|---|---|---|---|---|
| **Tier 1** — Flat 19-kelas | Klasifikasi langsung 19 kelas, satu model | **18.10%** (S3, E5_Data_Augmentation, barlow — champion existing, bukan hasil baru) | *tidak berlaku* (P6 tidak pernah melakukan klasifikasi flat 19-kelas) | — |
| **Tier 2** — First-syllable e2e (9-kelas, apple-to-apple) | Restricted ke 9 kelas suku-kata-pertama | **20.00%** (S3, dihitung baru — Task 1.1) | **28.57%** (S3, Stage B existing, diverifikasi ulang — Task 1.3) | P6 lebih tinggi **+8.57pp** |
| **Tier 3** — Full-word e2e | Kata utuh benar | **11.11%** (S3, satu-satunya titik data, dari `train_word_assembler_s3.py`, TIDAK dihitung ulang) | **13.89%** (S3, Stage B existing, diverifikasi ulang — Task 1.3) | P6 lebih tinggi **+2.78pp** |

Catatan penting Tier 2: P3 dievaluasi 9-way TAPI modelnya tetap memprediksi bebas ke 19 kelas (tidak
dibatasi struktural). Dari 84 prediksi salah pada subset 9-kelas ini, **47 (56%)** adalah tebakan ke
arah kelas suku-kata-KEDUA (mustahil benar untuk soal ini) — P6 tidak bisa melakukan kesalahan jenis
ini karena struktur coarse-nya memang cuma punya 4 kemungkinan output. Ini bagian dari cerita
arsitektural, bukan sekadar "P6 menang telak" — P3 "dirugikan" oleh fleksibilitas 19-kelasnya
sendiri di metrik ini.

---

## 3. Tabel MCDA — Bobot DIKONFIRMASI Peneliti (2026-07-12)

**Keputusan peneliti:** bobot akurasi (Tier 2 + Tier 3) dinaikkan ke total 60 (30/30). Empat
kriteria lain diturunkan proporsional dari usulan awal (20/10/15/5, total 50) ke total sisa 40,
menjaga proporsi relatif: class_coverage 16, system_complexity 8, latency 12, calibration_quality
4. Latensi memakai angka **fair same-basis** (bukan T9's 0.447ms) sesuai keputusan peneliti poin 1.
Metodologi normalisasi tidak berubah: skor performer terbaik = 100, yang lain = 100 × rasio.

| Kriteria | Bobot (dikonfirmasi) | P3 (raw) | P6 (raw) | Skor P3 | Skor P6 |
|---|---|---|---|---|---|
| Tier 2 accuracy (first-syllable e2e) | 30 | 20.00% | 28.57% | 70.00 | 100.00 |
| Tier 3 accuracy (full-word e2e) | 30 | 11.11% | 13.89% | 79.99 | 100.00 |
| Class coverage (% dari ruang kelas sendiri) | 16 | 94.74% (18/19) | 100% (9/9) | 94.74 | 100.00 |
| Kompleksitas sistem (jumlah model) | 8 | 1 | 5 | 100.00 | 20.00 |
| Latensi (ms, fair same-basis) | 12 | 652.48ms | 1342.49ms | 100.00 | 48.60 |
| Kualitas kalibrasi (ECE) | 4 | 0.1898 | 0.0771 | 40.62 | 100.00 |
| **TOTAL TERTIMBANG** | **100** | | | **81.78** | **87.43** |

Dengan bobot terkonfirmasi ini, **P6 unggul lebih jelas (87.43 vs 81.78, selisih 5.65 poin)** —
didorong oleh bobot akurasi yang lebih besar, di mana P6 menang di kedua tier. Skor "class
coverage" P6 (100%) tetap perlu dibaca hati-hati: itu 9/9 dari ruang kelas coarse-nya SENDIRI yang
sudah dipersempit secara struktural, bukan 9/19 dari total ruang kelas 19. Keputusan champion final
berdasarkan angka ini tetap di tangan peneliti — dokumen ini hanya menyajikan skor.

---

## 4. Ringkasan Temuan Kunci (Task 1.5–1.9)

**Task 1.5 — Error decomposition (S3, n=36 trial full-word test, 31 salah):**
**61.29%** kesalahan berasal dari tahap COARSE (grup vokal salah tebak — otomatis first-syllable
dan kata salah). **32.26%** dari tahap FINE (coarse benar, tapi model fine A/I/E salah). Hanya
**6.45%** dari sa_branch (KIT vs YANG). Ini kuantifikasi langsung: coarse memang bottleneck utama
pipeline, konsisten dengan rasional ablation study yang menargetkan perbaikan coarse.

**Task 1.6 — Theoretical ceiling vs actual (asumsi independensi coarse/fine):**
S3: ceiling teoritis 29.53% vs aktual 28.57% (selisih -0.96pp). Rerata 12 subjek: selisih -2.53pp,
**11 dari 12 subjek** aktualnya di BAWAH ceiling teoritis → error antar tahap **berkorelasi positif
lemah** (compounding tipis, kasus yang sulit di coarse cenderung juga sedikit lebih sulit di fine),
tapi efeknya kecil (~1-3pp), bukan compounding drastis.

**Task 1.7 — Feature importance cross-architecture (S3, permutation_importance n_repeats=30):**
Top-5 P6-coarse: `F3_freq, O1_amp, AF4_amp, T7_amp, F8_freq`. Top-5 P3: `O1_freq, O1_amp, F3_freq,
P7_amp, AF3_freq`. **Overlap 2/5 channel** (F3, O1) — dominasi channel oklusi (O1) SEBAGIAN
konsisten lintas arsitektur (O1 muncul di kedua top-5), tapi tidak identik. Temuan "O1 dominan"
dari P3 tidak sepenuhnya generalisasi ke P6.

**Task 1.8 — Calibration (S3, formula Brier/ECE identik dengan yang dipakai Varian E ablation):**
P3 champion (19-kelas): Brier 0.9839, ECE 0.1898. P6 coarse baseline (4-kelas, TANPA
CalibratedClassifierCV — SVC bawaan sudah `probability=True`): Brier 0.6619, ECE 0.0771. P6 Varian
E (dikutip, bukan dihitung ulang): Brier 0.6720, ECE 0.0844. **Catatan penting:** Brier score P3
yang lebih tinggi TIDAK serta-merta berarti "kalibrasi lebih buruk" — Brier score multi-kelas secara
matematis cenderung lebih tinggi untuk soal dengan lebih banyak kelas (19 vs 4), jadi bukan
perbandingan apple-to-apple murni. ECE (top-label) lebih valid untuk dibandingkan lintas jumlah
kelas, dan di situ juga P6 baseline lebih baik (0.0771 vs 0.1898).

**Task 1.9 — Konsistensi Stage-1 vs full-word e2e (12 subjek):**
Pearson r=0.001 (p=0.996), Spearman r=-0.018 (p=0.957) — **korelasi nyaris nol**, tidak signifikan.
Ini mengonfirmasi kuat bahwa S3 (first-syllable e2e tertinggi) vs S9 (full-word e2e tertinggi,
18.92%) bukan kontradiksi — dengan korelasi sedekat nol ini, sample size kecil per subjek (n_test
10-40 trial) di full-word e2e memang menghasilkan varians tinggi yang independen dari performa
Stage-1.

---

## 5. Ringkasan Argumen Skip End-to-End: P1, P2, P4, P5

- **P1 (Global):** Class coverage hanya **1/19** (confusion matrix menunjukkan collapse ekstrem, ~98%
  prediksi jatuh ke satu kelas dominan). Dari 10 kata target, **0 kata** yang kedua suku-katanya
  berada dalam himpunan kelas yang pernah ter-cover — struktural mustahil 100% sebelum pipeline
  apa pun dibangun.
- **P2 (Subject-Dependent EEGNet):** **12 dari 12 subjek** (100%) gagal gate class-coverage 8/19
  yang sama dipakai algoritma champion-selection — bahkan konfigurasi TERBAIK per subjek pun gagal.
  Tidak ada model Stage-1 yang punya cukup coverage untuk dijadikan fondasi pipeline hierarkis.
- **P4 (No-Windowing):** Akurasi rerata 5.98% (11/12 subjek, S6 hilang) — 1.14x chance (5.26%),
  praktis chance level. Proyeksi P(kata utuh benar) ≈ akurasi² ≈ **0.36%**, bahkan ini kemungkinan
  overestimate karena akurasi near-chance sering terkonsentrasi di satu kelas collapse (seperti
  terlihat langsung pada P1), bukan tersebar merata.
- **P5 (Shifted Bandpass):** Delta rerata **-1.11pp** vs P3 E0_Baseline/barlow (baseline yang sama,
  BUKAN champion E5), Wilcoxon p=0.3013 (tidak signifikan, n=12). Angka delta ini **cocok persis**
  dengan yang dikutip di ATURAN #0 — argumen "redundant confirmation" (bukan compounding error)
  didukung data. **Lihat Section 6 untuk satu ketidakcocokan kecil terkait arah delta per subjek.**

---

## 6. Anomali dan Kejutan (WAJIB Dibaca)

### 6.1 — TEMUAN PENTING: Klaim "0.447ms, real-time feasible" untuk P3 mengecualikan feature extraction

Ini kemungkinan temuan paling signifikan dari seluruh Fase 1, dengan dampak POTENSIAL di luar
scope P3-vs-P6 saja.

Task 1.4 awalnya mengukur latensi cascade P6 REAL (bukan hanya proyeksi) via
`predict_word_for_trial`: **1232ms (non-SA case)** dan **1904ms (kasus SA)**. Dibandingkan angka
P3 yang dikutip dari `T9_inference_latency.csv` (0.447ms), rasio-nya ~1300-4200x — jauh melampaui
proyeksi teoritis sederhana (2x/3x jumlah model call).

**Investigasi akar masalah:** `_measure_p3_latency()` di `notebooks/gen_nb_new.py` HANYA mengukur
waktu `StandardScaler.transform()` + `SVC.predict()` pada vektor yang SUDAH diekstraksi fitur
(dimuat langsung dari `Xtest_SVM_*.npy`) — **TIDAK PERNAH mengukur waktu ekstraksi fitur Barlow
dari epoch mentah**. Sebaliknya, pengukuran P6 (`SubModelBundle.predict_single`) SELALU melakukan
ekstraksi fitur dari epoch mentah di setiap panggilan, karena itulah cara kerja produksi
sesungguhnya (di dunia nyata, EEG mentah masuk, fitur harus diekstrak setiap saat — tidak ada
"fitur pre-extracted" yang gratis).

**Verifikasi:** setelah mengukur P3 dengan basis yang SAMA (mulai dari epoch mentah S3, lewat
ekstraksi fitur Barlow + scaler + predict, persis metodologi `SubModelBundle`), didapat P3 fair
= **652.48ms**. Rasio P6/P3 pada basis yang sama: non-SA **1.89x** (teori: 2x), SA-branch
**2.77x** (teori: 3x) — **sangat dekat dengan proyeksi teoritis**, mengonfirmasi bahwa gap ~1300x
yang terlihat di awal murni artefak perbandingan basis yang tidak setara, BUKAN P6 secara
arsitektural jauh lebih lambat.

**Implikasi yang perlu didiskusikan dengan peneliti:**
1. Untuk MCDA (Section 3) dan perbandingan P3-vs-P6 di skripsi, angka latensi yang dipakai
   **HARUS** angka fair (652ms vs 1233-1806ms), bukan T9's 0.447ms — sudah diterapkan di Section 3
   di atas.
2. **Ekstraksi fitur Barlow ternyata mahal (~650ms per panggilan)** — jika ini akurat merefleksikan
   produksi sesungguhnya, klaim real-time feasibility SISTEM (bukan cuma model call) di
   `T9_inference_latency.csv` / bagian NFR skripsi manapun yang mengutip "0.447ms, Yes real-time
   feasible" untuk P3 **berpotensi menyesatkan** kalau dibaca sebagai "waktu dari EEG mentah sampai
   prediksi" — angka itu sebenarnya cuma satu potongan kecil dari pipeline produksi sesungguhnya.
   Ini di luar scope literal prompt ini (yang minta perbandingan P3 vs P6, bukan audit NFR P3
   standalone), tapi cukup penting untuk diangkat eksplisit di sini karena bisa mempengaruhi
   klaim NFR/real-time yang sudah ada di draf skripsi atau notebook lain. **Tidak diubah apa pun
   di T9 atau file lain — hanya dilaporkan sebagai temuan, keputusan tindak lanjut di tangan
   peneliti.**
3. Detail lengkap (termasuk breakdown per skenario, warm-up run, dsb) ada di
   `results/p6_latency_measurement.json`, field `fair_comparison_same_basis`.

### 6.2 — P5 direction-consistency: 7/12, bukan 12/12 seperti disebut di konteks prompt

ATURAN #0 menyebutkan P5 "12/12 subjek konsisten arah negatif tipis." Rekomputasi Task 1.11
(P5 E0/barlow vs P3 E0/barlow per subjek, dari `T0_pillar3_raw_fresh.csv` dan
`results_S{n}.json` P5) menghasilkan **mean delta -1.11pp yang cocok PERSIS** dengan angka yang
dikutip — tapi arah per-subjek hanya **7 dari 12 negatif** (58%), bukan 12/12. Kemungkinan
penjelasan: klaim "12/12" di prompt merujuk ke perbandingan lain (mis. P5 vs P3 champion E5,
bukan vs P3 E0 baseline), atau ke sumber data yang berbeda, atau sekadar penyederhanaan yang
kurang presisi saat prompt ditulis. Kesimpulan STATISTIK (delta kecil, tidak signifikan) tetap
sama persis — ini TIDAK mengubah argumen "redundant confirmation" — tapi detail "12/12 konsisten"
sebaiknya JANGAN dipakai apa adanya di narasi Bab 6 nanti; gunakan "7/12 arah negatif, delta rerata
-1.11pp, tidak signifikan (Wilcoxon p=0.30)" yang sudah diverifikasi ulang di sini.

### 6.3 — Semua angka Stage B P6 terverifikasi 100%, termasuk yang paling berisiko

Task 1.3 (sanity check) berhasil MENYELURUH, termasuk Part C (replay `first_syllable_e2e` via
rebuild raw epoch lewat `DatasetBuilder` — jalur paling berisiko dari sisi memori mesin ini karena
`build_dataset.py` tidak punya mitigasi `usecols`/`chunksize` seperti `OfflineTrialReader`).
**60/60 akurasi sub-model, 3/3 full-word e2e (S3/S9/S1), dan first-syllable e2e S3 — SEMUA match
persis (abs_diff=0.0)** dengan angka yang sudah dilaporkan di `P7_CoarseToFine_report.md`. Tidak
ada indikasi angka Stage B itu placeholder atau salah hitung.

### 6.4 — S3 satu-satunya subjek dengan pipeline coverage penuh 9/9

Dari `results/p6_first_syllable_pipeline_coverage.json` (12 subjek): S3 = 9/9 (satu-satunya).
Subjek lain berkisar 3/9 (S6) sampai 8/9 (S2, S4, S7, S10, S11). Ini memperkuat rasional pemilihan
S3 sebagai champion candidate P6 di ATURAN #0 poin 4 — bukan cuma soal first-syllable e2e
tertinggi, tapi juga satu-satunya subjek tanpa "blind spot" struktural di pipeline coarse→fine.

---

## Konfirmasi Peneliti (2026-07-12) — Fase 1 DITUTUP, Fase 2 Dimulai

1. **Latensi:** dipakai angka **fair same-basis** (652ms vs 1233-1806ms) untuk MCDA maupun untuk
   skripsi — BUKAN T9's 0.447ms. Sudah diterapkan di Section 3.
2. **Bobot MCDA:** akurasi (Tier 2+Tier 3) dinaikkan ke total 60 (30/30); 4 kriteria lain
   diturunkan proporsional ke total 40 (coverage 16, complexity 8, latency 12, calibration 4).
   Sudah diterapkan di Section 3 — hasil akhir **P3 81.78 vs P6 87.43**.
3. **Temuan 6.1 (feature-extraction/real-time-feasibility gap):** tidak perlu ditindaklanjuti di
   luar scope ini — dilaporkan apa adanya, tidak ada perubahan lebih lanjut ke `T9_inference_latency.csv`
   atau file lain.
4. **Klaim "P5 12/12 konsisten negatif":** dikonfirmasi SALAH, akan dikoreksi ke "7/12" di narasi
   Bab 6 skripsi oleh peneliti sendiri (di luar scope kode/notebook ini).

**Fase 1 resmi selesai dan dikonfirmasi. Melanjutkan ke Fase 2 (revisi notebook) di bawah.**

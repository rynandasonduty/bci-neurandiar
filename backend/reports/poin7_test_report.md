# Laporan Pengujian Fungsional dan Evaluasi Sistem — Poin 7

**Tanggal pengujian:** 2026-07-07
**Sistem yang diuji:** Backend FastAPI NeurAndiAr, mode Model Offline (replay data mentah nyata), champion SVM P3_SVM/E5_Data_Augmentation/S3/Barlow.
**Environment:** `backend/venv/Scripts/python.exe`, server `uvicorn main:app` di `127.0.0.1:8123`.
**Model word assembler aktif pada demo:** `logreg_assembler_svm_S3only_barlow_E5_Data_Augmentation.pkl` (dilatih khusus dari trial subjek S3).

Catatan penting metodologis yang berlaku di seluruh laporan ini:
- Tidak ada data dummy/sintetis yang digunakan pada pengujian mana pun. Seluruh trial yang diuji adalah trial nyata dari rekaman EEG subjek S3.
- Latensi tidak dioptimasi dan dilaporkan apa adanya, sesuai keputusan yang sudah diambil.
- Rumusan masalah/tujuan penelitian Bab 1 yang eksplisit **tidak ditemukan** dalam repository (tidak ada `INSIGHT_id.docx` maupun file serupa, dan pencarian pada kedua notebook analisis tidak menemukan bagian "rumusan masalah"/"problem formulation"/"tujuan penelitian"). Tabel B.5 karena itu memakai rumusan umum sebagai fallback, sebagaimana diinstruksikan.

---

## B.1 Pengujian Fungsional Per Modul

| Modul | Skenario Uji | Hasil yang Diharapkan | Hasil Aktual | Status |
|---|---|---|---|---|
| Pembacaan data offline (`offline_trial_reader.py`) | Baca trial nyata subjek S3 (CSV mentah + log eksperimen), ekstrak 2 epoch (slot 1 & slot 2) | Epoch berbentuk (14 kanal, 256 sampel) per slot, label kata sesuai log eksperimen | Terverifikasi berulang kali (25 percobaan B.2 di bawah + smoke test kalibrasi): bentuk `(14, 256)` konsisten, kata target sesuai log (mis. trial 22 → "MANDI") | **PASS** |
| Preprocessing/filter sinyal | Bandpass Butterworth broadband (0.5–50 Hz) diterapkan ke sinyal mentah sebelum windowing, parameter identik resep E5 | Sinyal terfilter tanpa error, waktu proses tercatat (`signal_read_filter_ms`) | Median 0,23 ms per siklus (sinyal subjek sudah di-cache di memori setelah pemuatan pertama saat startup) | **PASS** |
| Ekstraksi fitur Barlow | `EEGFeatureExtractor(groups=['barlow'])` terhadap 1 epoch (14, 256) | Vektor fitur 28 dimensi (2 fitur x 14 kanal), tanpa NaN | Terverifikasi: bentuk `(1, 28)` pada setiap ekstraksi (log training S3-only dan pooled) | **PASS** |
| Inferensi SVM champion (slot 1 & slot 2) | `predict_proba_full()` pada epoch nyata slot 1 dan slot 2 secara independen | Dua vektor probabilitas 19 dimensi yang **berbeda**, tanpa NaN, jumlah ≈ 1 | Diverifikasi (sesi sebelumnya, trial S3 #0): `p1.shape=(19,)`, `p2.shape=(19,)`, `identical? False`, tanpa NaN. Konsisten pada seluruh 25 percobaan B.2 | **PASS** |
| Word assembler S3-only | `assemble_word_with_confidence()` pada pasangan `prob_slot1`/`prob_slot2` nyata | Kata dari 10 kelas target + confidence dalam rentang [0,100]% | Terverifikasi pada seluruh 25 percobaan B.2 (lihat tabel di bawah); confidence teramati 12,31%–27,4% | **PASS** (akurasi model itu sendiri dilaporkan terpisah di bagian S3-only vs Pooled, bukan bagian dari status fungsional ini) |
| Sentence refinement (rule-based) | `refine_sentence_rule_based(decoded_word)` menghasilkan kalimat AAC yang sesuai tabel | Kalimat sesuai `REFINEMENT_TABLE`, bukan LLM generatif | Terverifikasi pada seluruh 25 percobaan; contoh: "MANDI" → "Saya ingin mandi.", "BERAK" → "Saya ingin buang air besar." | **PASS** |
| TTS (Web Speech API, frontend) | Toggle "Audio Feedback" mengendalikan suara + animasi gelombang secara bersamaan, mengucapkan `refined_sentence` | Suara id-ID (atau fallback) berbunyi hanya saat toggle ON, animasi tersinkron `onstart`/`onend` | **Diverifikasi statis** (TypeScript compile bersih, alur kode sesuai — lihat sesi sebelumnya). **Belum diverifikasi manual di browser** — tidak diklaim PASS penuh sampai dikonfirmasi langsung oleh pengguna | **PERLU VERIFIKASI MANUAL** |
| Endpoint kalibrasi (`/api/v1/calibrate`) | Kirim 4 epoch nyata (2 trial S3) + label sebagai smoke test jalur `champion_type="svm"` | HTTP 200, model kalibrasi tersimpan, tanpa exception | `status_code: 200`, `{"status":"success","message":"Kalibrasi selesai.","model_path":"...calibrated_SVM_S_TEST_CALIBRATE.pkl"}`. Artefak uji dibersihkan setelah pengujian | **PASS** (uji fungsional/smoke test — bukan evaluasi kualitas kalibrasi, karena sampel terlalu kecil untuk bermakna) |
| Endpoint metrics/latensi (`/api/metrics`) | Setelah 25 siklus inferensi nyata, cek `overview.latency_is_proxy` dan angka median/P95 | `latency_is_proxy: false`, angka median/P95 sesuai perhitungan mandiri dari CSV mentah | `{"median_latency": 1058.1, "p95_latency": 1104.7, "latency_is_proxy": false, "latency_note": "Diukur nyata (time.perf_counter()) dari 25 siklus inferensi."}` — cocok dengan perhitungan independen di B.3 (selisih hanya pembulatan/metode kuantil) | **PASS** |

---

## B.2 Pengujian Integrasi End-to-End (25 Trial Nyata)

Dijalankan lewat `run_poin7_evaluation.py` terhadap server sungguhan (`/ws/inference`), masing-masing percobaan memilih satu trial nyata **secara acak** dari seluruh trial valid subjek S3 (178 trial), sehingga secara alami mencakup variasi kata target.

| # | Trial ID | Kata Target | Kata Prediksi | Kalimat Refined | Latensi Total (ms) | Status |
|---|---|---|---|---|---|---|
| 1 | 22 | MANDI | MANDI | Saya ingin mandi. | 1066,88 | Berhasil — **benar** |
| 2 | 190 | SAYANG | SAYANG | Saya sayang kamu. | 1053,83 | Berhasil — **benar** |
| 3 | 91 | SAKIT | PIPIS | Saya ingin buang air kecil. | 1104,54 | Berhasil — salah |
| 4 | 75 | TIDUR | SAYANG | Saya sayang kamu. | 1026,17 | Berhasil — salah |
| 5 | 43 | SAKIT | MANDI | Saya ingin mandi. | 1065,32 | Berhasil — salah |
| 6 | 53 | PIPIS | TIDUR | Saya ingin tidur. | 1032,12 | Berhasil — salah |
| 7 | 35 | LELAH | SAYANG | Saya sayang kamu. | 1099,47 | Berhasil — salah |
| 8 | 149 | MANDI | PIPIS | Saya ingin buang air kecil. | 1063,75 | Berhasil — salah |
| 9 | 65 | TIDUR | PIPIS | Saya ingin buang air kecil. | 1027,39 | Berhasil — salah |
| 10 | 76 | LELAH | PIPIS | Saya ingin buang air kecil. | 1056,41 | Berhasil — salah |
| 11 | 36 | BOSAN | TIDUR | Saya ingin tidur. | 1021,20 | Berhasil — salah |
| 12 | 26 | MAKAN | MAKAN | Saya ingin makan. | 1097,25 | Berhasil — **benar** |
| 13 | 150 | MAKAN | MAKAN | Saya ingin makan. | 1058,07 | Berhasil — **benar** |
| 14 | 123 | TIDUR | TIDUR | Saya ingin tidur. | 1033,10 | Berhasil — **benar** |
| 15 | 27 | SAKIT | LELAH | Saya merasa lelah. | 1045,18 | Berhasil — salah |
| 16 | 43 | SAKIT | MANDI | Saya ingin mandi. | 1052,96 | Berhasil — salah |
| 17 | 172 | MINUM | MAKAN | Saya ingin makan. | 1102,89 | Berhasil — salah |
| 18 | 159 | BERAK | SAYANG | Saya sayang kamu. | 1068,37 | Berhasil — salah |
| 19 | 16 | BERAK | BERAK | Saya ingin buang air besar. | 1099,46 | Berhasil — **benar** |
| 20 | 76 | LELAH | PIPIS | Saya ingin buang air kecil. | 1032,40 | Berhasil — salah |
| 21 | 172 | MINUM | MAKAN | Saya ingin makan. | 1054,15 | Berhasil — salah |
| 22 | 108 | BERAK | MINUM | Saya ingin minum. | 1140,69 | Berhasil — salah |
| 23 | 137 | SAKIT | SAKIT | Saya merasa sakit. | 1068,78 | Berhasil — **benar** |
| 24 | 46 | BOSAN | BOSAN | Saya merasa bosan. | 1049,06 | Berhasil — **benar** |
| 25 | 5 | MANDI | SAYANG | Saya sayang kamu. | 1104,71 | Berhasil — salah |

**Ringkasan integrasi:**
- **25/25 (100%) percobaan berhasil dijalankan tanpa error** dari data mentah sampai kalimat akhir (tidak ada `status: "error"` maupun exception).
- **8/25 (32,0%) kata hasil prediksi benar** pada sampel ini.

**Peringatan metodologis penting (harus dibaca bersamaan dengan angka 32,0% di atas):** trial dipilih acak dari keseluruhan 178 trial valid, bukan dibatasi ke 36 trial *held-out test split* yang dipakai saat evaluasi pelatihan assembler S3-only. Karena 142 dari 178 trial itu adalah bagian *training set* assembler, sebagian besar dari 25 sampel di atas kemungkinan besar tumpang tindih dengan data yang sudah "dilihat" model saat dilatih. Angka 32,0% ini karena itu **bias ke atas (optimis)** dan **bukan estimasi akurasi yang representatif** — estimasi yang tidak bias tetap angka *test split* resmi dari pelatihan: **11,11%** (lihat bagian S3-only vs Pooled). Angka 32,0% dilaporkan apa adanya sebagai hasil sampel pengujian integrasi, dengan caveat ini secara eksplisit dicatat.

---

## B.3 Latensi Sistem Nyata (Distribusi 25 Sampel)

| Tahap | Median (ms) | P95 (ms) | Rata-rata (ms) |
|---|---|---|---|
| **Total end-to-end** | **1058,07** | **1104,68** | 1064,97 |
| Baca sinyal + filter bandpass | 0,23 | 0,82 | 0,32 |
| Inferensi SVM slot 1 (`predict_proba`) | 529,69 | 600,38 | 535,47 |
| Inferensi SVM slot 2 (`predict_proba`) | 523,82 | 589,04 | 528,63 |
| Word assembly (LogReg) | 0,18 | 0,28 | 0,20 |
| Sentence refinement (rule-based) | 0,00 | 0,00 | 0,00 |

(Dihitung mandiri dari `backend/reports/poin7_raw_results.csv`; cocok dengan `/api/metrics`: median 1058,1 ms, P95 1104,7 ms — selisih desimal hanya karena metode interpolasi kuantil yang sedikit berbeda.)

### Perbandingan dengan Angka 0,43 ms di Draft Analisis

Angka 0,43 ms pada Draft Analisis (`Main Analysis 5: System Readiness and Inference Latency`, notebook `BCI Analysis and Results.ipynb`) diukur dengan metodologi yang **secara sengaja berbeda cakupannya** dari pengukuran end-to-end di atas, sehingga kedua angka **tidak bisa dibandingkan langsung sebagai "sebelum vs sesudah"**:

1. **Input yang diukur berbeda.** Notebook mengukur `scaler.transform()` + `svm.predict()` pada **fitur yang sudah diekstrak sebelumnya** (`Xtest_SVM_{fg}_{edir}_{subj}.npy`, vektor 28 dimensi siap pakai). Pengukuran sistem nyata di atas mengukur dari **sinyal EEG mentah** (14 kanal × 256 sampel) — termasuk pembacaan trial, filter bandpass, DAN ekstraksi fitur Barlow, dua kali (slot 1 dan slot 2).
2. **Method classifier yang berbeda.** Notebook memanggil `svm.predict()` (klasifikasi langsung, argmax dari decision function, tidak perlu kalibrasi probabilitas). Sistem nyata memanggil `svm.predict_proba()` **dua kali** per siklus, karena Word Assembler butuh distribusi probabilitas penuh 19 kelas untuk kedua slot, bukan hanya label prediksi. `predict_proba()` pada `SVC` scikit-learn menjalankan kalibrasi Platt-scaling (metode Wu-Lin-Weng) yang mengevaluasi fungsi keputusan terhadap **seluruh support vector** dari 171 sub-classifier one-vs-one (untuk 19 kelas) — jauh lebih mahal secara komputasi daripada `predict()` biasa, terutama karena akurasi champion yang rendah (18,10%) mengindikasikan margin antar kelas yang tumpang tindih dan kemungkinan jumlah support vector yang besar.
3. **Kondisi pengukuran.** Notebook mengulang 100 kali pada **sampel yang sama** (warm-up + reuse), sedangkan sistem nyata mengukur siklus penuh (baca-filter-ekstrak-infer-assembly-refine) pada **trial acak berbeda** setiap kali.

Kesimpulan: selisih ~2.400x (0,43 ms vs ~1058 ms) **sebagian besar dijelaskan oleh dua panggilan `predict_proba()` yang mahal secara kalibrasi**, bukan oleh regresi performa atau bug. Sesuai instruksi, **latensi ini tidak dioptimasi** dan dilaporkan apa adanya untuk didiskusikan/direvisi targetnya secara terpisah oleh pengguna di Bab 3.

---

## B.4 Analisis Kesalahan (Observasi Faktual dari 25 Sampel)

**Distribusi kata target (ground truth) dalam sampel:** SAKIT (5), MANDI (3), LELAH (3), TIDUR (3), BERAK (3), MAKAN (2), MINUM (2), BOSAN (2), SAYANG (1), PIPIS (1).

**Distribusi kata hasil prediksi:** SAYANG (5), PIPIS (5), MAKAN (4), MANDI (3), TIDUR (3), LELAH (1), BERAK (1), MINUM (1), SAKIT (1), BOSAN (1).

**Observasi 1 — bias prediksi ke "SAYANG" dan "PIPIS".** Kedua kata ini diprediksi 5 kali masing-masing, padahal sebagai kata target sebenarnya masing-masing hanya muncul 1 kali dalam sampel. Model tampak condong memprediksi kedua kelas ini terlepas dari input, pola khas classifier yang dilatih dari data kecil (178 trial, ~17-18 sampel/kelas) dengan sinyal masukan yang sangat noisy.

**Observasi 2 — recall per kata (dari kata target yang benar diprediksi / muncul):**

| Kata Target | Benar / Muncul | Recall |
|---|---|---|
| MAKAN | 2/2 | 100% |
| SAYANG | 1/1 | 100% |
| TIDUR | 1/3 | 33% |
| BOSAN | 1/2 | 50% |
| BERAK | 1/3 | 33% |
| MANDI | 1/3 | 33% |
| SAKIT | 1/5 | 20% |
| LELAH | 0/3 | 0% |
| MINUM | 0/2 | 0% |
| PIPIS | 0/1 | 0% |

**Observasi 3 — pasangan kesalahan (confusion) yang muncul lebih dari sekali:** SAKIT→MANDI (2x), MINUM→MAKAN (2x), LELAH→PIPIS (2x). Tidak ada pola confusion yang mendominasi secara jelas selain bias umum ke SAYANG/PIPIS pada Observasi 1.

**Observasi 4 — pengulangan trial.** Trial #43, #76, dan #172 masing-masing terpilih dua kali dalam 25 percobaan acak (dari pool 178 trial) — konsisten dengan pemilihan acak dengan pengembalian, bukan anomali sistem.

Analisis lebih dalam mengenai penyebab pola-pola ini (misalnya kaitan dengan fitur Barlow atau distribusi suku kata champion SVM) berada di luar cakupan laporan ini dan diserahkan ke pembahasan Bab 6.

---

## Akurasi Word Assembler: S3-Only vs Pooled 12-Subjek (Berdampingan)

| Varian | Data Latih | Jumlah Trial | Akurasi Training | Akurasi Uji (Test Split) | File Model |
|---|---|---|---|---|---|
| **S3-only** (aktif di demo) | Hanya subjek S3 | 178 (142 latih / 36 uji) | 37,32% | **11,11%** | `logreg_assembler_svm_S3only_barlow_E5_Data_Augmentation.pkl` |
| **Pooled 12-subjek** (disimpan, untuk diskusi keterbatasan generalisasi) | Seluruh 12 subjek | 1961 (1568 latih / 393 uji) | 17,41% | 7,38% | `logreg_assembler_svm_pooled12subj_barlow_E5_Data_Augmentation.pkl` |

**Kenapa keduanya relevan dilaporkan:**
- **S3-only** merepresentasikan performa realistis demo utama, karena inilah model yang benar-benar dimuat di jalur inferensi (`main.py`) dan konsisten secara metodologis dengan sifat *subject-dependent* champion SVM — probabilitas input assembler ini hanya berasal dari sinyal subjek yang model-nya benar-benar kenali.
- **Pooled 12-subjek** tetap relevan sebagai bukti pendukung kuantitatif untuk pembahasan Bab 6 mengenai keterbatasan generalisasi lintas-subjek: menunjukkan bahwa mencampur sinyal dari subjek yang "asing" bagi champion (11 dari 12 subjek) menghasilkan akurasi yang bahkan lebih rendah lagi (7,38% vs 11,11%), mendukung argumen bahwa pendekatan subject-dependent — meski akurasinya sendiri rendah — masih lebih baik daripada menggeneralisasi lintas-subjek secara naif untuk arsitektur ini.

**Observasi jujur:** peningkatan S3-only (11,11%) dibandingkan pooled (7,38%) **tidak sedramatis** yang mungkin diharapkan dari akurasi champion sendiri (18,10% pada tingkat suku kata). Kemungkinan penyebab: (a) ukuran sampel uji yang kecil (36 sampel) membuat estimasi akurasi bervariansi tinggi; (b) kesenjangan besar antara akurasi training (37,32%) dan uji (11,11%) mengindikasikan *overfitting* pada 142 sampel latih yang tersedia; (c) akurasi champion SVM sendiri yang rendah (18,10%) pada tingkat suku kata membatasi *ceiling* performa assembler apa pun yang dibangun di atasnya, terlepas dari komposisi data latihnya.

---

## B.5 Evaluasi terhadap Tujuan Penelitian

**Catatan cakupan:** rumusan masalah/tujuan penelitian eksplisit dari Bab 1 tidak ditemukan di repository ini (tidak ada `INSIGHT_id.docx` atau dokumen serupa; notebook analisis tidak memuat bagian rumusan masalah). Tabel berikut memakai rumusan umum yang diberikan sebagai fallback, sesuai instruksi.

| Tujuan Penelitian (Rumusan Umum) | Bukti dari B.1–B.4 | Status | Alasan Singkat |
|---|---|---|---|
| Bagaimana mendekode imagined speech dari EEG 14-kanal? | Pipeline lengkap (filter → fitur Barlow → SVM → assembler) berjalan end-to-end pada data EEG 14-kanal nyata subjek S3 (fase imagined maupun overt, lihat `phase` pada `OfflineTrialReader`); 25/25 siklus berhasil tanpa error teknis | **Tercapai sebagian** | Pipeline dekode berfungsi secara teknis (tidak ada kegagalan sistem), tetapi akurasi keluaran (11,11% test / 32,0% sampel bias) jauh dari andal untuk penggunaan praktis |
| Bagaimana membangun model ML (SVM + LogReg assembler) untuk sistem AAC 10-kata? | Champion SVM (18,10% akurasi suku kata, 18/19 cakupan kelas) + word assembler S3-only (11,11% akurasi uji) berhasil dibangun, dilatih dari data nyata, dan diintegrasikan ke backend produksi | **Tercapai sebagian** | Arsitektur model dan integrasinya ke sistem lengkap (termasuk TTS dan rule-based refinement) berhasil dibangun secara utuh; performa prediktifnya sendiri rendah |
| Berapa tingkat akurasi suku kata dan kata? | Suku kata (champion SVM, dari notebook analisis): 18,10%, cakupan 18/19 kelas. Kata (word assembler S3-only, sesi ini): training 37,32%, uji 11,11% (unbiased); sampel integrasi 32,0% (bias, lihat catatan B.2) | **Tercapai** | Kedua angka terukur dan terdokumentasi secara eksplisit dan jujur, tanpa rekayasa, sebagaimana disyaratkan |
| Apakah sistem memenuhi kebutuhan latensi real-time untuk AAC? | Latensi end-to-end median 1058,07 ms, P95 1104,68 ms, didominasi kalibrasi probabilitas SVM (`predict_proba` x2, lihat B.3) | **Tidak tercapai** (terhadap ambang lama 350 ms di Bab 3 draf lama) | Sudah diputuskan diterima apa adanya; target non-fungsional akan direvisi terpisah oleh pengguna, bukan dioptimasi oleh implementasi ini |
| Apakah kalimat keluaran sesuai konteks AAC (bukan kata mentah)? | Seluruh 25 percobaan B.2 menghasilkan kalimat lengkap dari `REFINEMENT_TABLE` (rule-based), bukan kata mentah; TTS mengucapkan `refined_sentence` (diverifikasi statis) | **Tercapai** (refinement) / **perlu verifikasi manual** (TTS di browser) | Refinement rule-based bekerja penuh dan diuji nyata; TTS sudah benar secara kode tapi belum dikonfirmasi langsung di browser |

---

## Berkas Pendukung

- `backend/reports/poin7_raw_results.csv` — data mentah 25 percobaan (sumber tabel B.2–B.4).
- `backend/logs/latency_history.csv` — log latensi nyata per siklus (direset sebelum pengujian batch ini agar statistik B.3 murni dari 25 sampel di atas).

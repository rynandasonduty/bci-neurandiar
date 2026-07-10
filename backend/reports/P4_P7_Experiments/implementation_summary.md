# P4-P7 Implementation Summary

Ringkasan seluruh keputusan desain untuk implementasi paradigma P4-P7, ditulis agar peneliti bisa
membaca ulang logika lengkapnya sebelum menjalankan training di komputer lab. Sesuai batasan tugas,
implementasi ini **hanya kode** -- tidak ada training skala penuh (12 subjek) yang dijalankan;
hanya smoke test ringan (lihat bagian "Smoke Test" di bawah).

## 1. Isolasi dari P1/P2/P3

- Tidak ada file yang terdaftar di Aturan Prioritas #1/#2 (`run_subject_dependent.py`,
  `run_e8_classical.py`, `run_master_experiments.py`, `train_pipeline.py`,
  `train_word_assembler.py`, `train_word_assembler_s3.py`, `signal_processor.py`,
  `build_dataset.py`, `extract_eeg_features.py`, `classical_models.py`, `data_utils.py`,
  `transfer_learning.py`, `main.py`, atau isi `P1_Global/`, `P2_EEGNet/`, `P3_SVM/`,
  `P4_TransferLearning/Calibrated/`) yang dimodifikasi. Semua file tersebut hanya dibaca
  (langsung via Read tool, atau dipanggil sebagai import/subclass read-only) untuk memahami
  signature dan perilaku sebelum menulis kode baru.
- Seluruh kode baru berada di `backend/src/experiments_p4_p7/` (paket baru), ditambah satu
  `git mv` (lihat bagian 2) dan edit `README.md`.
- Variasi P4/P5 dicapai lewat **subclassing** murni: `FullEpochSignalProcessor(SignalProcessor)`,
  `ShiftedBandSignalProcessor(SignalProcessor)` (di `signal_processors_ext.py`), dan
  `NoWindowDatasetBuilder(DatasetBuilder)`, `ShiftedBandDatasetBuilder(DatasetBuilder)` (di
  `dataset_builders_ext.py`) -- masing-masing hanya meng-override `__init__`/`windowing_slot`,
  seluruh logika lain (parsing CSV, log, phase filter, channel selection) tetap warisan langsung
  dari kelas asli.
- P6 dan P7 tidak butuh subclass sama sekali -- keduanya memanggil `DatasetBuilder` asli
  langsung dengan parameter berbeda (`phase_filter='overt'/'imagined'` untuk P6,
  `phase_filter='all'` untuk P7, identik metodologi P1-P3), karena variabel yang diuji bukan pada
  level pemrosesan sinyal.

## 2. Arsip P4 Transfer Learning Lama (Langkah 0.1)

`backend/src/models/run_p4_transfer_learning.py` dipindah (`git mv`, riwayat git terjaga) ke
`backend/src/models/legacy/run_p4_transfer_learning_DEPRECATED.py`, dengan komentar penjelas di
baris paling atas. Diverifikasi sebelum pemindahan: tidak ada file lain (`main.py`,
`transfer_learning.py`, `run_system_diagnostics.py`) yang mereferensikan path lama script ini
secara fungsional -- `transfer_learning.py` punya satu baris komentar yang menyebut nama file lama
untuk konteks historis, dibiarkan apa adanya (bukan referensi impor/path, dan file itu sendiri
tidak boleh disentuh sama sekali per Aturan Prioritas #3). Folder
`backend/models/weights/P4_TransferLearning/` (dipakai `/api/v1/calibrate`) tidak disentuh.

## 3. Struktur Folder

Mengikuti persis struktur yang diwajibkan di prompt tugas (lihat `README.md` Bagian 12 untuk
salinan terkini). Titik penting: `setup_experiment(exp_id, pilar)` (tidak diubah) selalu
menghasilkan `backend/models/weights/{pilar}/{exp_id}/` -- karena itu path Stage A/B setiap
paradigma dibangun dengan memanggil fungsi ini dengan `exp_id` = nama sub-folder tahap
(`"Spotcheck_S3_E0"`, `"Fullscale_12Subj_E0"`, dst.) dan `pilar` = nama paradigma
(`"P4_NoWindowing"`, dst.), bukan dengan nama eksperimen literal seperti pada P1-P3.

## 4. Aturan Seleksi Fitur Otomatis

Diimplementasikan sebagai `dataset_builders_ext.select_winning_feature_group()`, dipakai oleh
P4, P5, dan tahap kasar P7 (dengan `n_classes` berbeda: 19 untuk P4/P5, 4 untuk P7 tahap kasar,
supaya ambang chance-level yang dihitung benar). Diverifikasi lewat smoke test unit (bukan hanya
dibaca kodenya) untuk tiga skenario: pemenang langsung, tie-break -> barlow, dan
tie-break -> fallback coverage (barlow tidak ikut seri) -- lihat bagian Smoke Test.

Ditempatkan di `dataset_builders_ext.py` (bukan file terpisah) karena struktur folder yang
diwajibkan tidak mencantumkan file utilitas bersama tersendiri untuk logika ini; menaruhnya di
sini menghindari duplikasi 3x di `run_p4_nowindowing.py`/`run_p5_shifted_bandpass.py`/
`run_p7_coarse_to_fine.py` tanpa menambah file baru di luar daftar wajib.

## 5. P4 -- No-Windowing

**Temuan penting sebelum menulis kode:** repository sudah punya pilot P4 dari sesi kerja
sebelumnya (tanggal yang sama, terlihat dari `backend/reports/p4_no_windowing_pilot_report.md`
dan `p4_control_subsampled_report.md`), yaitu `preprocessing/full_epoch_processor.py`
(`FullEpochDatasetBuilder`, standalone, sengaja menduplikasi loop parsing alih-alih men-subclass
`DatasetBuilder`, karena saat itu `DatasetBuilder.__init__` tidak punya parameter `pilar`),
`preprocessing/windowed_reference_processor.py` (kontrolnya), `models/run_p4_no_windowing.py`
(driver pilot: S3 saja, Barlow saja -- hasil: **0% akurasi uji** pada n_train=106), dan
`models/run_p4_control_subsampled.py` (kontrol subsampling 5-seed pada dataset windowed penuh
1394 sampel -- rata-rata akurasi uji ~6.09%, menyimpulkan penurunan akurasi pilot terutama akibat
ukuran sampel kecil, bukan struktur no-windowing itu sendiri, z~-1.57 tidak signifikan).

**Keputusan:** menulis arsitektur BARU sesuai spesifikasi prompt ini persis (subclass murni
`FullEpochSignalProcessor`/`NoWindowDatasetBuilder`), yang secara arsitektur lebih bersih
daripada pendekatan lama (duplikasi loop) -- dan kini memungkinkan karena
`DatasetBuilder.process_subject()` sudah memanggil `self.processor.windowing_slot()` secara
generik. Struktur skrip driver (`run_p4_nowindowing.py`), skema hasil JSON, dan konvensi
penamaan file diadaptasi langsung dari `run_p4_no_windowing.py` (skrip pilot) dan
`run_e8_classical.py` (pola auto-resume), diperluas ke 5 fitur x 12 subjek. File pilot/kontrol
lama TIDAK disentuh/dihapus -- tetap sebagai referensi historis, ditulis di
`P4_NoWindowing_report.md`, dan menulis ke sub-folder BERBEDA (`Spotcheck_S3_E0/`,
`Fullscale_12Subj_E0/`, vs. pilot lama di `E0_Baseline/`), sehingga tidak ada tabrakan.

**Catatan teknis:** `NoWindowDatasetBuilder.__init__` memanggil `super().__init__()` (yang secara
internal memanggil `setup_experiment(exp_id)` tanpa `pilar`, selalu default ke `"P1_Global"`),
lalu segera menimpa `self.paths`/`self.raw_data_dir`/`self.output_dir` dengan
`setup_experiment(exp_id, pilar="P4_NoWindowing")`. Efek samping `os.makedirs(exist_ok=True)`
pada folder `P1_Global/{exp_id}/` bersifat no-op (folder itu, jika memakai nama exp_id yang
sudah dipakai P1 asli, sudah ada; jika belum, hanya folder kosong yang dibuat, tidak ada file
yang pernah ditulis ke sana). Dalam praktiknya, skrip P4 tidak pernah memanggil
`build_full_dataset()` (satu-satunya method yang memakai `self.output_dir`), jadi efek samping
ini murni kosmetik.

## 6. P5 -- Shifted Bandpass Filter

Tidak ada artefak sesi sebelumnya untuk P5 -- ditulis baru, simetris persis dengan struktur P4.

**Bug yang ditemukan lewat smoke test (lihat bagian Smoke Test):** signature
`ShiftedBandSignalProcessor.__init__` (persis seperti diberikan di prompt) TIDAK menerima
parameter `band`, karena rentang filter 15-65Hz memang dikunci secara internal. Namun
`E0_PROCESSOR_PARAMS` standar yang dipakai di seluruh codebase (termasuk rencana awal untuk P5)
menyertakan key `"band"`. Solusi: `run_p5_shifted_bandpass.py` mendefinisikan
`P5_PROCESSOR_PARAMS` sendiri (`{"apply_ica": False, "target_fs": 256}`, tanpa key `"band"`)
alih-alih memakai `E0_PROCESSOR_PARAMS` P4/P1-P3 apa adanya -- bukan mengubah signature kelas
yang sudah diberikan di prompt.

## 7. P6 -- Transfer Overt->Imagined

Baseline **tidak dilatih ulang** -- memuat langsung model/scaler/Xtest/ytest
`P3_SVM/E6_CrossModality_ImaginedOnly/barlow` yang sudah ada untuk 12 subjek (diverifikasi
lengkap sebelum diproses; subjek dengan artefak hilang dilaporkan, dilewati, tidak dilatih ulang
diam-diam). Split imagined-only direkonstruksi via `DatasetBuilder` asli + `three_way_split`
(seed 42), lalu disanity-check terhadap Xtest yang sudah tersimpan dengan cara: ekstrak ulang
fitur Barlow dari epoch mentah yang direkonstruksi, terapkan scaler LAMA (E6, sudah di-load, bukan
di-fit ulang) via `.transform()`, bandingkan hasilnya (shape + `np.allclose`) dengan Xtest yang
sudah tersimpan. Untuk model enriched yang baru, scaler BARU di-fit ulang khusus dari
`X_train_enriched` (imagined-train + seluruh overt), lalu diterapkan ke split val/test yang sama
(bukan Xtest lama) -- karena mengevaluasi model baru dengan fitur yang diskalakan scaler lama
akan salah secara matematis.

## 8. P7 -- Coarse-to-Fine Hierarchical Decoding

Setiap 5 sub-model (`coarse`, `fine_A`, `fine_I`, `fine_E`, `sa_branch`) diturunkan dari SATU
`three_way_split(seed=42)` bersama per subjek (dataset E0/19-kelas standar, dibangun via
`DatasetBuilder` asli) lewat filter label (`filter_split_by_labels`), masing-masing dengan scaler
sendiri yang di-fit hanya dari subset hasil filter tersebut (bukan scaler bersama) -- konsisten
dengan konvensi "Golden Standard" P1-P3 di mana setiap direktori model/subjek/fitur punya scaler
sendiri yang mandiri.

**Metrik end-to-end suku kata pertama** dihitung langsung dari `X_test`/`y_test` hasil split
bersama (bebas leakage, tidak perlu pemasangan trial) -- inilah metrik utama yang dipakai untuk
uji Wilcoxon di notebook. **Metrik end-to-end kata penuh** memerlukan pemasangan epoch slot-1 dan
slot-2 dari trial asli yang sama, yang tidak tersimpan di split level-window; dipecahkan dengan
memakai kembali `pipeline/offline_trial_reader.py` (`OfflineTrialReader`, dipakai juga oleh
`train_word_assembler.py`, read-only, bukan salah satu dari 5 file terlarang) untuk merekonstruksi
trial nyata langsung dari CSV+log mentah, lalu split 80/20 di level trial
(`test_size=0.2, random_state=42`) meniru persis metodologi `train_word_assembler_s3.py`. Karena
split trial ini independen dari split window-level yang melatih sub-model, ini BUKAN estimasi
bebas leakage murni -- caveat ini didokumentasikan eksplisit di kode, laporan, dan notebook
(caveat yang sama juga berlaku pada akurasi word assembler yang sudah ada, yang bahkan tidak
memisahkan diri dari data latih champion SVM sama sekali).

Sub-model `SubModelBundle` (helper generik baru, di `run_p7_coarse_to_fine.py`) dipakai untuk
inferensi single-epoch per sub-model, mengikuti pola `pipeline/svm_champion.py`'s `SVMChampion`
tapi dibuat generik untuk ruang kelas apa pun (bukan reuse langsung -- `SVMChampion`
mengasumsikan output tetap 19 kelas, tidak cocok untuk sub-model 2/3/4-kelas).

## 9. Orkestrator

Tiap tahap dijalankan sebagai **subprocess terpisah** (`subprocess.run([sys.executable, ...])`),
bukan pemanggilan fungsi in-process, khusus untuk kebersihan memori antar tahap -- mesin ini
memory-constrained, dan pola ini meniru alasan yang sama yang sudah dipakai
`train_word_assembler.py` (subprocess per subjek) untuk CSV mentah berukuran ratusan MB. Granularitas
subprocess di level SKRIP per PANGGILAN (bukan per subjek di dalam tiap skrip) karena
`run_e8_classical.py` -- preseden grid skala penuh paling mirip (480 model) -- terbukti selesai
tanpa isolasi subprocess per subjek; pola itu diikuti untuk kesederhanaan di dalam tiap
`run_p*.py`.

Checkpoint ditulis ke `orchestrator_run_log.md` (markdown, append-only) sebelum lanjut ke tahap
berikutnya; kegagalan satu tahap dicatat tapi TIDAK menghentikan pipeline. `--resume-from` murni
jaring pengaman (bukan jeda manual) karena seluruh keputusan (termasuk seleksi fitur) sudah
otomatis di dalam tiap skrip tahap.

## 10. Notebook Analisis

`notebooks/P4_P7_Analysis.ipynb`, terpisah total dari notebook champion. Baseline per-subjek
untuk P4/P5/P7 first-syllable dihitung dengan **mengevaluasi ulang** (bukan melatih ulang) model
P1-P3 yang sudah ada langsung dari model+Xtest+ytest tersimpan -- diperlukan karena ekspor CSV
P1-P3 yang sudah ada (`T14`/`T18`) hanya agregat (mean/std lintas subjek), tidak per-subjek,
sehingga tidak bisa dipasangkan untuk uji Wilcoxon berpasangan 12 subjek. P6 tidak butuh ini
karena baseline-nya sudah tersimpan langsung per-subjek di JSON hasil P6 sendiri. Untuk metrik
kata penuh P7, tidak ditemukan baseline 12-subjek yang bisa dipasangkan di codebase yang ada
(`train_word_assembler_s3.py` hanya S3, `train_word_assembler.py` menggabungkan semua subjek
jadi satu model) -- dilaporkan sebagai statistik deskriptif saja, sesuai "jika memungkinkan" di
spesifikasi tugas, tanpa uji Wilcoxon dipaksakan.

## 11. Status Verifikasi (Smoke Test Skala Kecil Ditunda)

Atas permintaan peneliti, eksekusi smoke test skala kecil untuk kelima `run_p*.py`/orkestrator
(yang melibatkan pembacaan CSV mentah 1 subjek dan pelatihan SVM sungguhan, meski dipersempit ke
1 subjek) **ditunda** -- akan dijalankan terpisah atas instruksi eksplisit peneliti berikutnya,
bukan bagian dari sesi implementasi ini. Sebagai gantinya, dilakukan verifikasi menyeluruh yang
**tidak menjalankan skrip eksperimen P4-P7 mana pun** namun tetap menemukan dan memperbaiki bug
nyata sebelum diserahkan:

1. **Unit test modul bersama** (`signal_processors_ext.py`, `dataset_builders_ext.py`) dengan
   data sintetis kecil: `FullEpochSignalProcessor.windowing_slot` (epoch penuh + rejeksi artefak),
   `ShiftedBandSignalProcessor` (lowcut/highcut ter-override benar), `NoWindowDatasetBuilder`/
   `ShiftedBandDatasetBuilder` (instansiasi, `self.processor` benar, `self.channel_indices` tetap
   utuh setelah swap processor), `select_winning_feature_group` (3 skenario: pemenang langsung,
   tie-break->barlow, tie-break->fallback coverage, plus peringatan below-chance),
   `filter_split_by_labels`, `map_labels_to_vowel_group_ids`.
   - **Bug ditemukan & diperbaiki lewat unit test ini:** `ShiftedBandSignalProcessor.__init__`
     (persis seperti diberikan di prompt) tidak menerima parameter `band`, sementara
     `E0_PROCESSOR_PARAMS` standar menyertakannya -- lihat Bagian 6 untuk detail perbaikan.
2. **Eksekusi penuh `verify_p6_phase_labels.py` dan `verify_p7_label_scheme.py`** terhadap data
   mentah sungguhan seluruh 12 subjek (bukan sekadar dibaca kodenya) -- keduanya PASS bersih:
   12/12 subjek punya split overt/imagined 100/100 seimbang; skema label P7 cocok 100% (2400/2400
   trial nyata, 0 mismatch) dengan `SYLLABLE_CLASSES`, hierarki vokal, kamus kata deterministik,
   dan offset ID marker.
3. **Notebook `P4_P7_Analysis.ipynb` dijalankan penuh, dua kali**, sel demi sel, secara terpisah
   dari skrip eksperimen P4-P7 (notebook hanya membaca hasil JSON + mengevaluasi ulang model P1-P3
   yang sudah ada, tidak pernah melatih ulang apa pun):
   - Sekali terhadap repo dalam keadaan kosong (belum ada hasil P4-P7 sama sekali, kondisi nyata
     repo saat ini) -- menemukan dan memperbaiki **KeyError nyata** (`p7_df['subject']` pada
     `pd.DataFrame` kosong tanpa kolom) di dua sel notebook P7.
   - Sekali lagi dengan data sintetis (5 subjek palsu per paradigma, ditulis lalu dihapus lagi
     setelah tes, termasuk satu nilai akurasi persis 0.0 secara sengaja) untuk melatih jalur uji
     Wilcoxon, plotting, dan kesimpulan go/no-go -- sekaligus menemukan dan memperbaiki **bug
     falsy-zero** (`X or np.nan` salah memperlakukan akurasi 0.0 yang valid sebagai data hilang,
     padahal pilot P4 sebelumnya sudah pernah mencatat hasil 0% asli). Evaluasi model P3 baseline
     dalam tes ini memakai model P3 sungguhan (inferensi read-only, bukan pelatihan ulang). Seluruh
     file/folder sintetis dan figur PNG hasil tes dihapus kembali setelah verifikasi -- tidak ada
     data palsu yang tersisa di repository.
4. **Mekanisme inti orkestrator** (`run_stage`, logging checkpoint, semantik lanjut-meski-gagal)
   diuji dengan perintah palsu (`python -c "..."`, tanpa pernah memanggil skrip P4-P7 sungguhan)
   terhadap file log sementara -- terverifikasi mencatat SUCCESS/FAILED dengan benar dan tidak
   pernah melempar exception saat satu tahap gagal. `--list-stages`/`--help` juga dijalankan
   langsung (aman, tidak meluncurkan subprocess apa pun) dan cocok dengan dokumentasi di README.
5. **Tinjauan manual baris-demi-baris** seluruh isi `run_p4_nowindowing.py`,
   `run_p5_shifted_bandpass.py`, `run_p6_transfer_overt_imagined.py`, `run_p7_coarse_to_fine.py`
   (skrip yang benar-benar melatih SVM pada data mentah, sengaja tidak dieksekusi sesuai
   permintaan peneliti) -- memverifikasi konsistensi nama file model/scaler antara logika
   auto-resume dan logika penyimpanan, kebenaran urutan transpose bentuk array, konsistensi ruang
   label antar sub-model P7, dan penanganan kasus tepi (data hilang, subjek tanpa artefak
   baseline, dll).

**Belum diverifikasi lewat eksekusi nyata** (menunggu smoke test lanjutan atas instruksi
peneliti): apakah `three_way_split` benar-benar menghasilkan reproduksi identik pasangan
train/val/test P3 saat direkonstruksi P6/P7 (logika sudah ditelusuri manual dan diyakini benar --
lihat Bagian 7 -- tapi belum dibuktikan lewat run sungguhan), dan performa memori/waktu nyata pada
mesin lab untuk memuat CSV mentah 12 subjek secara berurutan.

## 12. Verifikasi Isolasi

`git status`/`git diff` di titik verifikasi tidak menunjukkan file baru sebagai "belum di-stage"
seperti biasanya -- repository ini punya mekanisme auto-commit + auto-push ke `origin/main` yang
aktif selama sesi ini berjalan (dikonfirmasi peneliti sebagai perilaku yang disengaja, bukan aksi
Claude Code -- tidak pernah ada pemanggilan `git commit`/`git push` eksplisit dalam sesi ini).
Akibatnya, sebagian besar pekerjaan sesi ini sudah ter-commit otomatis ke `main` dan ter-push ke
GitHub sebelum laporan ini selesai ditulis. Lihat `git log --oneline` untuk komit-komit terkait
(mis. "feat: add code for P4 P5 P6 P7 paradigm") dan `git show --stat <hash>` untuk daftar file
per komit sebagai bukti isolasi -- daftar file di setiap komit hanya berisi file-file yang memang
dimaksudkan prompt ini (`experiments_p4_p7/`, `README.md`, `legacy/`,
`notebooks/P4_P7_Analysis.ipynb`, `backend/reports/P4_P7_Experiments/`), tidak ada file P1/P2/P3
atau `transfer_learning.py`/`main.py` yang tersentuh.

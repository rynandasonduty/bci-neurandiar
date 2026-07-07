"""
Train Word Assembler — Data Nyata
====================================
Melatih ulang model LogisticRegression `WordAssembler` menggunakan
probabilitas prediksi NYATA dari champion SVM (P3_SVM / E5_Data_Augmentation
/ subjek S3 / grup fitur Barlow), menggantikan pendekatan lama yang melatih
pada data acak (`np.random.rand`, lihat `logreg_model.py` blok `__main__`).

Untuk setiap subjek dan setiap trial pada dataset mentah asli, skrip ini:
    1. Mengekstrak epoch slot 1 dan slot 2 sungguhan (OfflineTrialReader),
       memakai parameter preprocessing identik dengan resep E5.
    2. Menjalankan kedua epoch tersebut lewat champion SVM (SVMChampion)
       untuk mendapatkan prob_slot1 dan prob_slot2 nyata (19 dimensi).
    3. Memasangkan pasangan probabilitas ini dengan label kata sebenarnya
       dari log eksperimen (bukan label yang direkayasa).
    4. Melatih LogisticRegression pada seluruh data nyata yang terkumpul.

Setiap subjek diproses di SUBPROCESS terpisah (bukan hanya fungsi dalam satu
proses panjang): CSV mentah tiap subjek berukuran ratusan MB, dan pada mesin
dengan memori bebas terbatas, `gc.collect()` di dalam proses yang sama tidak
selalu benar-benar mengembalikan memori ke OS akibat fragmentasi allocator.
Subprocess yang keluar sepenuhnya menjamin memori dikembalikan ke OS sebelum
subjek berikutnya diproses.

PENTING — LARANGAN TEGAS: Skrip ini tidak menggunakan data dummy, data
sintetis, maupun rekayasa angka akurasi dalam bentuk apa pun. Akurasi yang
dilaporkan di akhir eksekusi adalah akurasi murni dari data nyata dan wajib
dilaporkan apa adanya, sekalipun rendah. Jika satu atau lebih subjek gagal
diproses (misalnya karena keterbatasan memori sistem saat runtime), hal ini
dilaporkan secara eksplisit sebagai subjek yang dilewati, bukan disembunyikan.

Catatan: Model SVM champion bersifat subject-dependent (dilatih hanya dari
data subjek S3). Skrip ini tetap menjalankan model tersebut terhadap epoch
milik SELURUH subjek (S1-S12) untuk memaksimalkan jumlah data latih
assembler, sesuai spesifikasi tugas. Akibatnya, probabilitas slot untuk
subjek selain S3 kemungkinan besar kurang akurat (model tidak pernah
melihat sinyal subjek tersebut saat dilatih) — ini adalah keterbatasan yang
diketahui dan disengaja, bukan bug.

Model hasil skrip ini ("pooled", 12 subjek) disimpan sebagai
`logreg_assembler_svm_pooled12subj_barlow_E5_Data_Augmentation.pkl` dan
TIDAK dipakai sebagai model aktif jalur inferensi demo (lihat
`train_word_assembler_s3.py` untuk model demo). Model ini tetap
dipertahankan sebagai bukti pendukung keterbatasan generalisasi
lintas-subjek untuk pembahasan hasil.

Usage:
    cd backend/src/models
    python train_word_assembler.py
"""

import os
import sys
import gc
import glob
import argparse
import tempfile
import subprocess

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import RAW_DATA_DIR
from models.logreg_model import WordAssembler, WORD_CLASSES

CHAMPION_PARADIGM   = "P3_SVM"
CHAMPION_EXP        = "E5_Data_Augmentation"
CHAMPION_SUBJECT    = "S3"
CHAMPION_FEAT_GROUP = "barlow"

E5_PROCESSOR_PARAMS = {"band": "broadband", "apply_ica": False, "target_fs": 256}

WORD_ASSEMBLER_FILENAME = (
    f"logreg_assembler_svm_pooled12subj_{CHAMPION_FEAT_GROUP}_{CHAMPION_EXP}.pkl"
)


def resolve_champion_paths():
    """Temukan path model dan scaler champion SVM di disk (tidak menebak nama file)."""
    from config import MODELS_DIR

    weights_dir = os.path.join(MODELS_DIR, "weights", CHAMPION_PARADIGM, CHAMPION_EXP)
    model_path = os.path.join(
        weights_dir, f"SVM_{CHAMPION_FEAT_GROUP}_{CHAMPION_EXP}_{CHAMPION_SUBJECT}.pkl"
    )
    scaler_path = os.path.join(
        weights_dir, f"scaler_SVM_{CHAMPION_FEAT_GROUP}_{CHAMPION_EXP}_{CHAMPION_SUBJECT}.pkl"
    )
    return model_path, scaler_path


def build_training_data_for_subject(subject_id, model_path, scaler_path):
    """
    Ekstrak (prob_slot1, prob_slot2, label_kata) nyata untuk setiap trial valid
    milik satu subjek. Dijalankan di dalam proses worker (lihat --subject).
    """
    from pipeline.offline_trial_reader import OfflineTrialReader
    from pipeline.svm_champion import SVMChampion

    svm_champion = SVMChampion(
        model_path=model_path,
        scaler_path=scaler_path,
        feat_group=CHAMPION_FEAT_GROUP,
        fs=E5_PROCESSOR_PARAMS["target_fs"],
    )
    reader = OfflineTrialReader(RAW_DATA_DIR, E5_PROCESSOR_PARAMS)

    try:
        trials = reader.list_valid_trials(subject_id)
    except Exception as e:
        print(f"      [WARNING] Tidak bisa membaca data mentah subjek {subject_id}: {e}")
        return None, None

    X_rows, y_rows = [], []
    n_skipped = 0

    for trial_idx, trial_meta in enumerate(trials):
        word = trial_meta["word"].strip().upper()
        if word not in WORD_CLASSES:
            n_skipped += 1
            continue

        try:
            trial = reader.read_trial(subject_id, trial_index=trial_idx)
        except Exception:
            n_skipped += 1
            continue

        prob_slot1 = svm_champion.predict_proba_full(trial["epoch_slot1"])
        prob_slot2 = svm_champion.predict_proba_full(trial["epoch_slot2"])

        X_rows.append(np.concatenate([prob_slot1, prob_slot2]))
        y_rows.append(WORD_CLASSES[word])

    if n_skipped:
        print(f"      [INFO] {n_skipped} trial dilewati (label tidak dikenal / window tidak bersih).")

    del reader, svm_champion
    gc.collect()

    if not X_rows:
        return None, None
    return np.array(X_rows), np.array(y_rows)


def run_subject_worker(subject_id, output_path):
    """Mode worker: proses satu subjek dan simpan hasilnya ke .npz, lalu keluar."""
    model_path, scaler_path = resolve_champion_paths()
    X_subj, y_subj = build_training_data_for_subject(subject_id, model_path, scaler_path)
    if X_subj is None:
        return 1
    np.savez(output_path, X=X_subj, y=y_subj)
    return 0


def main():
    model_path, scaler_path = resolve_champion_paths()
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print(f"[ERROR] Artefak champion SVM tidak ditemukan:")
        print(f"        model  : {model_path}")
        print(f"        scaler : {scaler_path}")
        print("[ERROR] Pastikan run_e8_classical.py sudah dijalankan sebelumnya. Dibatalkan.")
        return

    log_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "logs", "*_experiment_log.txt")))
    subject_ids = [os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files]
    print(f"[INFO] Ditemukan {len(subject_ids)} subjek pada dataset mentah: {subject_ids}")

    all_X, all_y = [], []
    per_subject_counts = {}
    failed_subjects = []

    with tempfile.TemporaryDirectory(prefix="neurandiar_word_assembler_") as tmp_dir:
        for subject_id in subject_ids:
            print(f"\n[INFO] Memproses subjek {subject_id} (subprocess terisolasi)...")
            out_path = os.path.join(tmp_dir, f"{subject_id}.npz")

            result = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--subject", subject_id, "--output", out_path],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )

            if result.returncode != 0 or not os.path.exists(out_path):
                print(f"      [WARNING] Tidak ada data valid untuk subjek {subject_id}. Dilewati.")
                failed_subjects.append(subject_id)
                continue

            with np.load(out_path) as data:
                X_subj, y_subj = data["X"], data["y"]

            print(f"      [INFO] {len(y_subj)} trial nyata berhasil diekstraksi untuk {subject_id}.")
            per_subject_counts[subject_id] = len(y_subj)
            all_X.append(X_subj)
            all_y.append(y_subj)

    if not all_X:
        print("\n[ERROR] Tidak ada data nyata yang berhasil diekstraksi dari subjek mana pun. Pelatihan dibatalkan.")
        return

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print("\n" + "=" * 60)
    print(" DATA NYATA TERKUMPUL")
    print("=" * 60)
    for subj, n in per_subject_counts.items():
        print(f"  {subj:<6}: {n} trial")
    if failed_subjects:
        print(f"  DILEWATI (gagal diproses): {failed_subjects}")
    print(f"  TOTAL : {len(y)} trial dari {len(per_subject_counts)} subjek")
    print("=" * 60)

    unique_labels, label_counts = np.unique(y, return_counts=True)
    can_stratify = len(unique_labels) > 1 and min(label_counts) >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
    )

    assembler = WordAssembler(
        exp_id=CHAMPION_EXP,
        pilar=CHAMPION_PARADIGM,
        filename=WORD_ASSEMBLER_FILENAME,
    )
    assembler.train(X_train, y_train)

    y_pred = assembler.model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(" HASIL PELATIHAN WORD ASSEMBLER DARI DATA NYATA")
    print(" (Akurasi dilaporkan apa adanya, tanpa rekayasa)")
    print("=" * 60)
    print(f" Jumlah total sampel latih+uji : {len(y)}")
    print(f" Jumlah subjek berhasil        : {len(per_subject_counts)} / {len(subject_ids)}")
    if failed_subjects:
        print(f" Subjek dilewati               : {failed_subjects}")
    print(f" Akurasi set uji (test split)  : {test_acc * 100:.2f}%")
    print("=" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    assembler.save_model()
    print(f"[INFO] Word assembler tersimpan di: {assembler.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=None, help="Internal: worker mode for one subject.")
    parser.add_argument("--output", type=str, default=None, help="Internal: .npz output path for worker mode.")
    args = parser.parse_args()

    if args.subject:
        sys.exit(run_subject_worker(args.subject, args.output))
    else:
        main()

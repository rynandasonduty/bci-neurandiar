"""
Train Word Assembler — Khusus Subjek S3 (Model Demo Aktif)
==============================================================
Varian `train_word_assembler.py` yang melatih dan mengevaluasi
LogisticRegression `WordAssembler` HANYA dari trial nyata subjek S3,
bukan pooled 12 subjek.

Alasan (keputusan desain yang disengaja, bukan perbaikan bug): champion
SVM (P3_SVM / E5_Data_Augmentation / S3 / Barlow) bersifat subject-dependent
— dilatih murni dari sinyal S3. Mencampur trial dari 11 subjek lain yang
sinyalnya tidak pernah dilihat model ke dalam data latih assembler tidak
representatif untuk performa demo, yang juga dikunci hanya untuk subjek S3
(lihat `api/main.py`). Model ini adalah yang dimuat aktif di jalur
inferensi demo.

Model hasil skrip pooled 12 subjek (`train_word_assembler.py`) TETAP
disimpan terpisah (`logreg_assembler_svm_pooled12subj_barlow_E5_Data_Augmentation.pkl`)
sebagai bukti pendukung keterbatasan generalisasi lintas-subjek, bukan
dihapus atau ditimpa.

PENTING — LARANGAN TEGAS: Skrip ini tidak menggunakan data dummy, data
sintetis, maupun rekayasa angka akurasi dalam bentuk apa pun. Akurasi yang
dilaporkan di akhir eksekusi adalah akurasi murni dari data nyata subjek S3
dan wajib dilaporkan apa adanya.

Usage:
    cd backend/src/models
    python train_word_assembler_s3.py
"""

import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.logreg_model import WordAssembler
from models.train_word_assembler import (
    CHAMPION_PARADIGM,
    CHAMPION_EXP,
    CHAMPION_SUBJECT,
    CHAMPION_FEAT_GROUP,
    resolve_champion_paths,
    build_training_data_for_subject,
)

WORD_ASSEMBLER_FILENAME = (
    f"logreg_assembler_svm_S3only_{CHAMPION_FEAT_GROUP}_{CHAMPION_EXP}.pkl"
)


def main():
    model_path, scaler_path = resolve_champion_paths()
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("[ERROR] Artefak champion SVM tidak ditemukan:")
        print(f"        model  : {model_path}")
        print(f"        scaler : {scaler_path}")
        print("[ERROR] Pastikan run_e8_classical.py sudah dijalankan sebelumnya. Dibatalkan.")
        return

    print(f"[INFO] Memuat champion SVM: {os.path.basename(model_path)}")
    print(f"[INFO] Mengekstrak trial nyata khusus subjek {CHAMPION_SUBJECT}...")

    X, y = build_training_data_for_subject(CHAMPION_SUBJECT, model_path, scaler_path)
    if X is None:
        print(f"[ERROR] Tidak ada data valid untuk subjek {CHAMPION_SUBJECT}. Pelatihan dibatalkan.")
        return

    print(f"[INFO] {len(y)} trial nyata berhasil diekstraksi untuk subjek {CHAMPION_SUBJECT}.")

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
    train_acc = assembler.train(X_train, y_train)

    y_pred = assembler.model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f" HASIL PELATIHAN WORD ASSEMBLER — SUBJEK {CHAMPION_SUBJECT} SAJA")
    print(" (Akurasi dilaporkan apa adanya, tanpa rekayasa)")
    print("=" * 60)
    print(f" Jumlah total sampel latih+uji : {len(y)}")
    print(f" Akurasi training              : {train_acc * 100:.2f}%")
    print(f" Akurasi set uji (test split)  : {test_acc * 100:.2f}%")
    print("=" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    assembler.save_model()
    print(f"[INFO] Word assembler (S3-only) tersimpan di: {assembler.model_path}")


if __name__ == "__main__":
    main()

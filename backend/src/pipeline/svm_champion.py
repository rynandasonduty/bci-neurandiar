"""
SVM Champion Inference Helper
================================
Memuat pipeline classical-ML champion (P3_SVM / E5_Data_Augmentation /
subjek S3 / grup fitur Barlow) dan menyediakan satu titik akses untuk
mengubah satu epoch EEG mentah (channels, time) menjadi vektor probabilitas
19 kelas suku kata. Modul ini dipakai bersama oleh endpoint inferensi
langsung (`api/main.py`) dan skrip pelatihan word assembler
(`models/train_word_assembler.py`) agar keduanya memakai prosedur ekstraksi
fitur dan prediksi yang identik.
"""

import pickle

import numpy as np

from features.extract_eeg_features import EEGFeatureExtractor

NUM_SYLLABLE_CLASSES = 19


class SVMChampion:
    def __init__(self, model_path, scaler_path, feat_group="barlow", fs=256):
        """Memuat pipeline SVM dan scaler champion yang sudah dilatih dari disk."""
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.feat_group = feat_group
        self.extractor = EEGFeatureExtractor(fs=fs)

    def predict_proba_full(self, epoch_2d):
        """
        Jalankan ekstraksi fitur dan prediksi SVM pada satu epoch EEG nyata.

        Args:
            epoch_2d (np.ndarray): Epoch EEG tunggal, bentuk (channels, time).

        Returns:
            np.ndarray: Vektor probabilitas berbentuk (19,). Kelas suku kata
            yang tidak pernah muncul pada label data latih diisi nol, agar
            dimensi keluaran selalu konsisten 19 terlepas dari
            `self.pipeline.classes_`.
        """
        X_3d = epoch_2d[np.newaxis, :, :]
        features = self.extractor.transform(X_3d, groups=[self.feat_group])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = self.scaler.transform(features)

        raw_proba = self.pipeline.predict_proba(features_scaled)[0]
        classes = self.pipeline.classes_

        full_proba = np.zeros(NUM_SYLLABLE_CLASSES, dtype=np.float64)
        for cls, p in zip(classes, raw_proba):
            full_proba[int(cls)] = p
        return full_proba

# File: backend/src/models/transfer_learning.py
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Menghubungkan path agar bisa memanggil modul lain
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.extract_eeg_features import EEGFeatureExtractor
from models.classical_models import ClassicalClassifier
from sklearn.preprocessing import StandardScaler

def calibrate_new_user(base_model_path, X_new_3d, y_new, new_subject_id, save_dir, champion_type="eegnet", feat_group="all"):
    """
    Modul Cerdas Kalibrasi Pengguna Baru (Mendukung EEGNet dan SVM).
    
    Parameters:
    - base_model_path: Path ke file model juara (.h5 atau .pkl).
    - X_new_3d: Data mentah pengguna baru. Shape: (Samples, 14 Channels, 256 Time, 1 Depth).
    - y_new: Label/Target dari pengguna baru.
    - new_subject_id: ID pengguna (misal: "S_Baru_01").
    - save_dir: Lokasi penyimpanan model hasil kalibrasi.
    - champion_type: "eegnet" atau "svm".
    - feat_group: Konfigurasi fitur jika SVM (misal: "time", "all").
    """
    print(f"[*] Memulai Proses Kalibrasi untuk Pengguna: {new_subject_id}...")
    os.makedirs(save_dir, exist_ok=True)
    
    # =====================================================================
    # CABANG 1: JIKA CHAMPION MODEL ADALAH DEEP LEARNING (EEGNet)
    # =====================================================================
    if champion_type.lower() == "eegnet":
        print("[INFO] Arsitektur: EEGNet. Menjalankan Protokol Transfer Learning Sejati.")
        
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"[!] Base model EEGNet tidak ditemukan di {base_model_path}")
            
        model = load_model(base_model_path)
        
        # 1. Bekukan (Freeze) semua lapisan kecuali lapisan ujung (Klasifikasi)
        for layer in model.layers:
            if 'dense' not in layer.name.lower() and 'softmax' not in layer.name.lower():
                layer.trainable = False
                
        # 2. Compile ulang dengan Learning Rate sangat kecil agar ilmu aslinya tidak rusak
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # 3. Fine-tuning dengan data pengguna baru
        print(f"[-] Menyesuaikan lapisan kognitif dengan {len(X_new_3d)} sampel baru...")
        model.fit(X_new_3d, y_new, epochs=15, batch_size=2, verbose=1) # Epoch kecil, batch kecil karena data sedikit
        
        # 4. Simpan Model Personal
        save_path = os.path.join(save_dir, f"calibrated_EEGNet_{new_subject_id}.h5")
        model.save(save_path)
        print(f"[+] Model personal EEGNet berhasil disimpan di: {save_path}")
        
        return save_path, "eegnet"

    # =====================================================================
    # CABANG 2: JIKA CHAMPION MODEL ADALAH CLASSICAL ML (SVM)
    # =====================================================================
    elif champion_type.lower() == "svm":
        print("[INFO] Arsitektur: SVM. Menjalankan Protokol Fast Retraining.")
        
        # 1. Hapus dimensi kedalaman (Depth) karena SVM tidak butuh
        # Dari (Samples, 14, 256, 1) menjadi (Samples, 14, 256)
        X_new_2d_raw = np.squeeze(X_new_3d, axis=-1)
        
        # 2. Ekstraksi Fitur Matematis
        print(f"[-] Mengekstrak fitur '{feat_group}' dari {len(X_new_2d_raw)} sampel...")
        extractor = EEGFeatureExtractor(fs=256) # fs=256 sesuai standar Emotiv EPOC X
        groups = None if feat_group == 'all' else [feat_group]
        
        X_features = extractor.transform(X_new_2d_raw, groups=groups)
        
        # Pembersihan NaN (jika terjadi error ICA pada sampel baru)
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Scaling Fitur (Fit & Transform baru khusus user ini)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        
        # Simpan scaler khusus pengguna ini
        scaler_path = os.path.join(save_dir, f"calibrated_scaler_{new_subject_id}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        # 4. Latih SVM Baru dari Nol (Sangat cepat, < 1 detik)
        print("[-] Melatih model SVM personal...")
        svm_model = ClassicalClassifier(model_type='svm', C=10)
        svm_model.train(X_scaled, y_new)
        
        # 5. Simpan Model Personal
        save_path = os.path.join(save_dir, f"calibrated_SVM_{new_subject_id}.pkl")
        svm_model.save_model(save_path)
        
        print(f"[+] Model personal SVM berhasil disimpan di: {save_path}")
        print(f"[+] Scaler personal disimpan di: {scaler_path}")
        
        return save_path, "svm"
        
    else:
        raise ValueError("Champion type harus 'eegnet' atau 'svm'")
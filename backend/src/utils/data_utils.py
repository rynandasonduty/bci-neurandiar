"""
utils/data_utils.py
Fungsi split terpusat dan normalisasi aman untuk semua paradigma eksperimen BCI.
Menjamin tidak ada data leakage dan test set konsisten antar eksperimen.
"""
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

def three_way_split(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Memecah data menjadi Train, Validation, dan Test secara aman (Anti-Leakage).
    Dilengkapi Fallback jika stratifikasi gagal akibat data imbalanced.
    """
    test_ratio = test_size
    val_relative_ratio = val_size / (1.0 - test_ratio)
    
    try:
        # Percobaan 1: Stratified Split (Pembagian Proporsional - Best Practice)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_ratio, random_state=random_state, stratify=y_temp
        )
        
    except ValueError:
        # Percobaan 2: Fallback ke Random Split biasa jika ada kelas yang cuma berjumlah 1
        print("      [!] Peringatan: Data terlalu imbalanced untuk stratifikasi. Menggunakan Random Split.")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state, stratify=None
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_ratio, random_state=random_state, stratify=None
        )

    return X_train, X_val, X_test, y_train, y_val, y_test

def fit_and_apply_scaler(X_train, X_val, X_test, save_path=None):
    """
    HUKUM EMAS: StandardScaler HANYA boleh di-fit pada X_train!
    Setelah itu, baru transform X_train, X_val, dan X_test.
    Fungsi ini aman untuk dimensi 3D (Machine Learning Klasik) maupun 4D (EEGNet).
    """
    original_shape_train = X_train.shape
    original_shape_val = X_val.shape
    original_shape_test = X_test.shape

    scaler = StandardScaler()
    
    # Reshape ke 2D (Sample, Features) agar StandardScaler bisa bekerja
    X_train_2d = X_train.reshape(len(X_train), -1)
    X_val_2d = X_val.reshape(len(X_val), -1)
    X_test_2d = X_test.reshape(len(X_test), -1)

    # FIT HANYA PADA TRAIN, LALU TRANSFORM KETIGANYA
    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(original_shape_train)
    X_val_scaled = scaler.transform(X_val_2d).reshape(original_shape_val)
    X_test_scaled = scaler.transform(X_test_2d).reshape(original_shape_test)

    # Simpan scaler ke disk untuk tahap inferensi online atau evaluasi nanti
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(scaler, f)
            
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
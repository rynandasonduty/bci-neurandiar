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

def three_way_split(X, y, val_ratio=0.15, test_ratio=0.15):
    """
    Membagi data menjadi 3 kantong: Train (70%), Validation (15%), dan Test (15%).
    Menggunakan stratify untuk menjaga keseimbangan kelas suku kata.
    """
    # Split pertama: Pisahkan Test set (15% dari total)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=RANDOM_STATE, stratify=y
    )
    
    # Hitung proporsi validation terhadap sisa data (X_temp)
    # Jika test_ratio = 0.15, sisa data (temp) adalah 0.85. 
    # Maka 0.15 / 0.85 = 0.17647...
    val_of_temp = val_ratio / (1.0 - test_ratio)
    
    # Split kedua: Pisahkan Train dan Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_of_temp, 
        random_state=RANDOM_STATE, stratify=y_temp
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
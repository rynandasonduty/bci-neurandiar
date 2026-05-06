import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import shap
import tensorflow as tf

# Memaksa TensorFlow untuk berperilaku seperti Numpy agar SHAP tidak crash
tf.experimental.numpy.experimental_enable_numpy_behavior()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment

def run_explainability(exp_id="E0_Baseline"):
    print("\n" + "="*50)
    print(f" MEMULAI ANALISIS SHAP (EXPLAINABLE AI) UNTUK: {exp_id} ")
    print("="*50)

    paths = setup_experiment(exp_id)
    weights_dir = paths["weights"]
    reports_dir = paths["reports"]
    processed_dir = paths["processed_data"]

    print("[*] Memuat Model EEGNet dan Data Tensor...")
    model_path = os.path.join(weights_dir, f"eegnet_trained_{exp_id}.h5")
    if not os.path.exists(model_path):
        print("[X] Model tidak ditemukan.")
        return
        
    model = load_model(model_path)
    
    # [PERBAIKAN AUDIT] Menggunakan X_test.npy yang sudah di-scale
    data_path = os.path.join(processed_dir, "X_test.npy")
    if not os.path.exists(data_path):
        print("[X] Data X_test.npy tidak ditemukan.")
        return
        
    X = np.load(data_path)
    
    # Ambil sampel (50 untuk background, 3 untuk diuji)
    background = X[:50]
    test_samples = X[50:53]
    
    print("[*] Menyiapkan Background Data untuk Explainer...")
    explainer = shap.GradientExplainer(model, background)
    
    print("[*] Menjalankan Algoritma SHAP (GradientExplainer)...")
    try:
        shap_values = explainer.shap_values(test_samples)
        
        print("[*] Membuat Visualisasi Heatmap...")
        shap_array = np.array(shap_values) 
        
        # [PERBAIKAN KRITIS] Merata-rata dimensi agar menjadi 2D (Channel x Time)
        if isinstance(shap_values, list):
            # Bentuk: (19 kelas, 3 sampel, 14 channel, 256 time, 1 depth)
            shap_mean = np.mean(np.abs(shap_array), axis=(0, 1, 4)) 
        else:
            shap_abs = np.abs(shap_array)
            # Ratakan semua axis kecuali Channel (axis=1) dan Time (axis=2)
            shap_mean = np.mean(shap_abs, axis=tuple([i for i in range(shap_abs.ndim) if i not in [1, 2]]))
        
        # Pastikan orientasi matrix benar: 14 baris (Channel), 256 kolom (Waktu)
        n_channels = X.shape[1] 
        if shap_mean.shape[0] != n_channels:
            shap_mean = shap_mean.T

        plt.figure(figsize=(12, 6))
        sns.heatmap(shap_mean, cmap="viridis", cbar_kws={'label': 'Mean |SHAP value|'})
        plt.title(f"SHAP Feature Importance (Channels x Time) - {exp_id}")
        plt.ylabel("EEG Channels")
        plt.xlabel("Time Samples")
        
        out_file = os.path.join(reports_dir, f"shap_heatmap_{exp_id}.png")
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print(f"[+] Visualisasi SHAP berhasil disimpan di: {out_file}")
        
    except Exception as e:
        print(f"[!] Gagal mengeksekusi kalkulasi SHAP: {e}")

    print("\n" + "="*50)
    print(f" ANALISIS SHAP SELESAI. Cek folder: {reports_dir}/")
    print("="*50)

if __name__ == "__main__":
    run_explainability()
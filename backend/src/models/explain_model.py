import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf
from tensorflow.keras.models import load_model

# Agar SHAP DeepExplainer berjalan lancar di TensorFlow 2.x
tf.compat.v1.disable_v2_behavior()

# Konfigurasi Path
PROCESSED_DIR = "../../dataset/processed"
MODEL_DIR = "../../dataset/models"
EXPLAIN_DIR = "../../dataset/explainability"

# Saluran Emotiv EPOC X sesuai urutan di signal_processor.py
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

def ensure_dirs():
    os.makedirs(EXPLAIN_DIR, exist_ok=True)

def plot_shap_heatmap(shap_values_2d, class_name, sample_idx):
    """
    Menggambar Heatmap SHAP (14 Channels x 256 Time Steps)
    Warna merah = Mendorong tebakan ke kelas tersebut
    Warna biru = Menolak tebakan ke kelas tersebut
    """
    plt.figure(figsize=(12, 6))
    
    # Ambil nilai absolut rata-rata jika ingin melihat intensitas murni, 
    # tapi nilai asli (positif/negatif) lebih informatif untuk SHAP.
    sns.heatmap(shap_values_2d, cmap='coolwarm', center=0, 
                yticklabels=CHANNELS, cbar_kws={'label': 'SHAP Value (Dampak Fitur)'})
    
    plt.title(f'SHAP Feature Relevance - Prediksi Suku Kata: {class_name} (Sample #{sample_idx})')
    plt.xlabel('Waktu (Titik Sampel 0 - 256)')
    plt.ylabel('Sensor EEG (14 Channels)')
    plt.tight_layout()
    
    filename = f"shap_heatmap_class_{class_name}_sample_{sample_idx}.png"
    plt.savefig(os.path.join(EXPLAIN_DIR, filename))
    plt.close()
    print(f"[+] Heatmap disimpan: {filename}")

def main():
    ensure_dirs()
    print("="*50)
    print(" MEMULAI ANALISIS SHAP (DEEPLIFT) UNTUK EEGNET ")
    print("="*50)

    # 1. Muat Model dan Data
    print("[*] Memuat Model EEGNet dan Data Tensor...")
    try:
        model = load_model(os.path.join(MODEL_DIR, "eegnet_trained.h5"))
        X = np.load(os.path.join(PROCESSED_DIR, "X_features.npy"))
        # Ingat: Bentuk asli dari build_dataset adalah (N, 256, 14)
        # Kita harus transposisi ke bentuk yang dikenali model: (N, 14, 256, 1)
        X = np.transpose(X, (0, 2, 1))
        X = np.expand_dims(X, axis=3)
    except Exception as e:
        print(f"[X] Gagal memuat model/data: {e}")
        return

    # 2. Siapkan Data Latar Belakang (Background Data)
    # SHAP membutuhkan data acuan untuk membandingkan "tidak ada sinyal" dengan "ada sinyal"
    # Kita ambil 100 sampel acak sebagai representasi baseline
    print("[*] Menyiapkan Background Data untuk DeepExplainer...")
    background_indices = np.random.choice(X.shape[0], 100, replace=False)
    background_data = X[background_indices]

    # 3. Inisialisasi SHAP DeepExplainer (Berbasis DeepLIFT)
    print("[*] Menjalankan Algoritma SHAP (Ini mungkin memakan waktu beberapa menit)...")
    explainer = shap.DeepExplainer(model, background_data)

    # 4. Pilih beberapa sampel uji untuk dibedah (misal: 3 sampel pertama)
    num_test_samples = 3
    test_data = X[:num_test_samples]
    
    # Dapatkan prediksi aktual dari model untuk pelabelan grafik
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)

    # Hitung SHAP Values
    # shap_values akan berbentuk list [19 array], satu array untuk setiap kelas suku kata
    shap_values = explainer.shap_values(test_data)

    print("\n[*] Membuat Visualisasi Heatmap...")
    for i in range(num_test_samples):
        pred_class = predicted_classes[i]
        
        # Ambil matriks SHAP untuk kelas yang berhasil ditebak oleh model pada sampel ke-i
        # Bentuk awal: (14 channels, 256 time steps, 1 depth)
        sample_shap = shap_values[pred_class][i]
        
        # Hilangkan dimensi depth yang tidak perlu menjadi (14, 256)
        sample_shap_2d = np.squeeze(sample_shap)
        
        # Gambar peta panasnya
        plot_shap_heatmap(sample_shap_2d, pred_class, i)

    print("\n" + "="*50)
    print(f" ANALISIS SHAP SELESAI. Cek folder: {EXPLAIN_DIR}/")
    print("="*50)

if __name__ == "__main__":
    main()
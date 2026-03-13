import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from eegnet_model import EEGNetClassifier

# Konfigurasi Path Folder
PROCESSED_DIR = "../../dataset/processed"
MODEL_DIR = "../../dataset/models"

def load_and_prepare_data():
    """Memuat data tensor dan menyesuaikan dimensinya untuk TensorFlow/Keras"""
    print("[*] Memuat dataset tensor dari folder processed...")
    X = np.load(os.path.join(PROCESSED_DIR, "X_features.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "y_labels.npy"))
    
    print(f"[+] Bentuk data awal X: {X.shape}")
    
    # Keras Conv2D membutuhkan format: (Samples, Channels, Time, Depth)
    # Saat ini data kita dari build_dataset.py adalah: (Samples, 256, 14)
    # Kita harus melakukan transposisi menjadi: (Samples, 14, 256)
    X = np.transpose(X, (0, 2, 1))
    
    # Tambahkan dimensi 'Depth' (menjadi 1 karena ini sinyal 1D, bukan gambar RGB)
    X = np.expand_dims(X, axis=3)
    
    print(f"[+] Bentuk data siap latih untuk EEGNet: {X.shape}")
    return X, y

def plot_history(history, save_path):
    """Menggambar grafik Akurasi dan Loss selama proses pelatihan"""
    plt.figure(figsize=(12, 5))
    
    # Grafik Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Akurasi Pelatihan (Train)')
    plt.plot(history.history['val_accuracy'], label='Akurasi Validasi (Val)')
    plt.title('Perkembangan Akurasi Model')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend()
    
    # Grafik Loss (Tingkat Kesalahan)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss Pelatihan (Train)')
    plt.plot(history.history['val_loss'], label='Loss Validasi (Val)')
    plt.title('Penurunan Tingkat Kesalahan (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[+] Grafik evaluasi pelatihan disimpan di: {save_path}")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Siapkan Data
    X, y = load_and_prepare_data()
    
    # 2. Bagi menjadi Data Latih (80%) dan Data Uji/Validasi (20%)
    # stratify=y memastikan proporsi kelas 19 suku kata terbagi rata
    print("\n[*] Membagi data menjadi 80% Training dan 20% Validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Inisialisasi Arsitektur EEGNet
    print("\n[*] Membangun Arsitektur Jaringan Saraf Tiruan (EEGNet-8,2)...")
    eegnet = EEGNetClassifier(nb_classes=19, channels=14, samples=256)
    
    # 4. Mulai Proses Pelatihan (Training)
    print("[*] Memulai pelatihan mesin... (Bisa memakan waktu beberapa menit)")
    # Menggunakan batch_size 64 agar proses lebih cepat dan stabil di RAM
    history = eegnet.train(X_train, y_train, X_val, y_val, epochs=500, batch_size=64)
    
    # 5. Simpan Model Digital yang Sudah Pintar
    model_path = os.path.join(MODEL_DIR, "eegnet_trained.h5")
    eegnet.save_model(model_path)
    print(f"\n[SUCCESS] Model berhasil dilatih dan disimpan di: {model_path}")
    
    # 6. Buat Laporan Visual (Grafik)
    plot_path = os.path.join(MODEL_DIR, "training_history.png")
    plot_history(history, plot_path)
    print("="*50)
    print(" SELURUH PROSES PELATIHAN EEGNET SELESAI ")
    print("="*50)

if __name__ == "__main__":
    main()
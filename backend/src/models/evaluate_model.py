import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from eegnet_model import EEGNetClassifier # Mengimpor class untuk mendapatkan struktur jika perlu
from word_assembler import WORD_CLASSES, REVERSE_WORD_CLASSES

# Konfigurasi Path
PROCESSED_DIR = "../../dataset/processed"
MODEL_DIR = "../../dataset/models"
EVAL_DIR = "../../dataset/evaluation"

# Daftar suku kata (Index 0-18)
SYLLABLE_NAMES = [
    "MA", "KAN", "MI", "NUM", "BE", "RAK", "PI", "PIS", "MAN", "DI", 
    "BO", "SAN", "LE", "LAH", "SA", "KIT", "TI", "DUR", "YANG"
]
WORD_NAMES = list(WORD_CLASSES.keys())

def ensure_dirs():
    os.makedirs(EVAL_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    """Fungsi helper untuk menggambar dan menyimpan Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Label Asli (True)')
    plt.xlabel('Tebakan Sistem (Predicted)')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, filename))
    plt.close()
    print(f"[+] {title} disimpan di: {filename}")

def evaluate_system():
    ensure_dirs()
    print("="*50)
    print(" MEMULAI EVALUASI SISTEM BCI (AKURASI & LATENSI) ")
    print("="*50)

    # 1. Muat Model yang Sudah Dilatih
    print("[*] Memuat Model EEGNet dan Word Assembler...")
    try:
        eegnet = load_model(os.path.join(MODEL_DIR, "eegnet_trained.h5"))
        with open(os.path.join(MODEL_DIR, "logistic_regression_assembler.pkl"), 'rb') as f:
            word_assembler = pickle.load(f)
    except Exception as e:
        print(f"[X] Gagal memuat model. Pastikan train_pipeline dan word_assembler sudah dijalankan. Error: {e}")
        return

    # 2. Simulasi Data Uji (Karena data asli belum ada, kita gunakan dummy yang terstruktur)
    # Dalam skenario nyata, Anda akan meload X_test.npy dan y_test.npy
    print("[*] Menyiapkan 1000 data simulasi untuk Uji Kinerja...")
    num_samples = 1000
    # Input tensor untuk EEGNet: (Samples, 14 Channels, 256 Time, 1 Depth)
    X_test_eeg = np.random.randn(num_samples, 14, 256, 1)
    y_test_syllables = np.random.randint(0, 19, num_samples)

    # ---------------------------------------------------------
    # EVALUASI TAHAP 1: EEGNET (AKURASI SUKU KATA & LATENSI)
    # ---------------------------------------------------------
    print("\n--- EVALUASI TAHAP 1: SUKU KATA (EEGNET) ---")
    
    inference_times_eeg = []
    y_pred_syllables = []

    # Kita uji satu per satu untuk mengukur latensi persis seperti saat live-streaming
    for i in range(num_samples):
        sample = np.expand_dims(X_test_eeg[i], axis=0) # Ambil 1 epoch (1 detik)
        
        start_time = time.perf_counter()
        prob = eegnet.predict(sample, verbose=0)
        end_time = time.perf_counter()
        
        inference_times_eeg.append((end_time - start_time) * 1000) # Konversi ke milidetik
        y_pred_syllables.append(np.argmax(prob))

    # Kalkulasi Metrik Suku Kata
    acc_syllable = accuracy_score(y_test_syllables, y_pred_syllables)
    median_lat_eeg = np.median(inference_times_eeg)
    p95_lat_eeg = np.percentile(inference_times_eeg, 95)

    print(f"[+] Akurasi Suku Kata    : {acc_syllable * 100:.2f}%")
    print(f"[+] Median Latensi       : {median_lat_eeg:.2f} ms")
    print(f"[+] 95th Percentile Lat  : {p95_lat_eeg:.2f} ms")
    
    plot_confusion_matrix(y_test_syllables, y_pred_syllables, SYLLABLE_NAMES, 
                          "Confusion Matrix Suku Kata", "cm_syllables.png")

    # ---------------------------------------------------------
    # EVALUASI TAHAP 2: WORD ASSEMBLER (AKURASI KATA & LATENSI TOTAL)
    # ---------------------------------------------------------
    print("\n--- EVALUASI TAHAP 2: KATA UTUH (SISTEM END-TO-END) ---")
    
    # Membuat simulasi pasangan kata: 500 Kata Utuh (1 Kata = 2 Suku Kata)
    num_words = 500
    y_test_words = np.random.randint(0, 10, num_words)
    
    total_latencies = []
    y_pred_words = []

    for i in range(num_words):
        # Ambil 2 sampel EEG acak seolah-olah ini Slot 1 dan Slot 2
        slot1_eeg = np.expand_dims(X_test_eeg[i], axis=0)
        slot2_eeg = np.expand_dims(X_test_eeg[i+1], axis=0)
        
        # MENGHITUNG TOTAL WAKTU END-TO-END (Dari Sinyal -> Teks Kata)
        start_total = time.perf_counter()
        
        # Prediksi Slot 1 & 2
        prob1 = eegnet.predict(slot1_eeg, verbose=0)[0]
        prob2 = eegnet.predict(slot2_eeg, verbose=0)[0]
        
        # Rakit Kata
        combined_probs = np.concatenate((prob1, prob2)).reshape(1, -1)
        pred_word_idx = word_assembler.predict(combined_probs)[0]
        
        end_total = time.perf_counter()
        
        total_latencies.append((end_total - start_total) * 1000)
        y_pred_words.append(pred_word_idx)

    # Kalkulasi Metrik Kata
    acc_word = accuracy_score(y_test_words, y_pred_words)
    median_lat_total = np.median(total_latencies)
    p95_lat_total = np.percentile(total_latencies, 95)

    print(f"[+] Akurasi Kata Utuh    : {acc_word * 100:.2f}%")
    print(f"[+] Median Latensi Total : {median_lat_total:.2f} ms")
    print(f"[+] 95th Percentile Total: {p95_lat_total:.2f} ms")

    plot_confusion_matrix(y_test_words, y_pred_words, WORD_NAMES, 
                          "Confusion Matrix Kata", "cm_words.png")

    print("\n" + "="*50)
    print(f" EVALUASI SELESAI. Semua laporan disimpan di: {EVAL_DIR}/")
    print("="*50)

if __name__ == "__main__":
    evaluate_system()
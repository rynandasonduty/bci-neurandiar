import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment
from models.eegnet_model import EEGNetClassifier 

SYLLABLE_NAMES = [
    "MA", "KAN", "MI", "NUM", "BE", "RAK", "PI", "PIS", "MAN", "DI", 
    "BO", "SAN", "LE", "LAH", "SA", "KIT", "TI", "DUR", "YANG"
]

WORD_NAMES = [
    "MAKAN", "MINUM", "BERAK", "PIPIS", "MANDI", 
    "BOSAN", "LELAH", "SAKIT", "TIDUR", "SAYANG"
]

def plot_confusion_matrix(y_true, y_pred, classes, title, filepath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Label Asli (True)')
    plt.xlabel('Tebakan Sistem (Predicted)')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"[+] {title} disimpan di: {filepath}")

def evaluate_system(exp_id="E0_Baseline"):
    print("\n" + "="*50)
    print(f" MEMULAI EVALUASI SISTEM BCI (TEST DATA) UNTUK: {exp_id} ")
    print("="*50)

    paths = setup_experiment(exp_id)
    weights_dir = paths["weights"]
    reports_dir = paths["reports"]
    processed_dir = paths["processed_data"]

    # 1. MUAT MODEL
    print("[*] Memuat Model EEGNet dan Word Assembler...")
    try:
        eegnet_path = os.path.join(weights_dir, f"eegnet_trained_{exp_id}.h5")
        eegnet = load_model(eegnet_path)
        
        logreg_path = os.path.join(weights_dir, f"logreg_assembler_{exp_id}.pkl")
        with open(logreg_path, 'rb') as f:
            word_assembler = pickle.load(f)
    except Exception as e:
        print(f"[X] Gagal memuat model untuk '{exp_id}'. Error: {e}")
        return

    # 2. MUAT TEST DATA
    print("[*] Memuat data TEST dari direktori eksperimen...")
    try:
        # Data Suku Kata untuk Evaluasi EEGNet (Sudah dalam bentuk yang benar dari train_pipeline.py)
        X_eeg = np.load(os.path.join(processed_dir, "X_test.npy"))
        y_syl = np.load(os.path.join(processed_dir, "y_test.npy"))

        # Data Kata Utuh untuk Evaluasi Logistic Regression
        X_word = np.load(os.path.join(processed_dir, "X_word_test.npy"))
        y_word = np.load(os.path.join(processed_dir, "y_word_test.npy"))
    except Exception as e:
        print(f"[X] Gagal memuat data test. Pastikan train_pipeline dan build_logreg_dataset dijalankan. Error: {e}")
        return

    # ---------------------------------------------------------
    # EVALUASI TAHAP 1: EEGNET (AKURASI SUKU KATA)
    # ---------------------------------------------------------
    print("\n--- EVALUASI TAHAP 1: SUKU KATA (EEGNET) ---")
    
    start_time = time.perf_counter()
    prob_syl = eegnet.predict(X_eeg, verbose=0)
    end_time = time.perf_counter()
    
    y_pred_syl = np.argmax(prob_syl, axis=1)
    acc_syllable = accuracy_score(y_syl, y_pred_syl)
    avg_lat_eeg = ((end_time - start_time) / len(X_eeg)) * 1000

    print(f"[+] Total Sampel Suku Kata : {len(X_eeg)}")
    print(f"[+] Akurasi Suku Kata      : {acc_syllable * 100:.2f}%")
    print(f"[+] Rata-rata Latensi/Item : {avg_lat_eeg:.2f} ms")
    
    cm_syl_path = os.path.join(reports_dir, f"cm_syllables_{exp_id}.png")
    plot_confusion_matrix(y_syl, y_pred_syl, SYLLABLE_NAMES, 
                          f"Confusion Matrix Suku Kata ({exp_id})", cm_syl_path)

    # ---------------------------------------------------------
    # EVALUASI TAHAP 2: WORD ASSEMBLER (AKURASI KATA)
    # ---------------------------------------------------------
    print("\n--- EVALUASI TAHAP 2: KATA UTUH (LOGISTIC REGRESSION) ---")
    
    start_total = time.perf_counter()
    y_pred_word = word_assembler.predict(X_word)
    end_total = time.perf_counter()

    acc_word = accuracy_score(y_word, y_pred_word)
    avg_lat_word = ((end_total - start_total) / len(X_word)) * 1000

    print(f"[+] Total Sampel Kata      : {len(X_word)}")
    print(f"[+] Akurasi Kata Utuh      : {acc_word * 100:.2f}%")
    print(f"[+] Rata-rata Latensi/Item : {avg_lat_word:.2f} ms")

    cm_word_path = os.path.join(reports_dir, f"cm_words_{exp_id}.png")
    plot_confusion_matrix(y_word, y_pred_word, WORD_NAMES, 
                          f"Confusion Matrix Kata ({exp_id})", cm_word_path)

    print("\n" + "="*50)
    print(f" EVALUASI SELESAI. Semua laporan disimpan di: {reports_dir}/")
    print("="*50)

if __name__ == "__main__":
    evaluate_system(exp_id="E0_Baseline")
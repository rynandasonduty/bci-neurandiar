import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf
from tensorflow.keras.models import load_model

# Impor dari root backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment

# Saluran Emotiv EPOC X sesuai urutan di signal_processor.py
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

def plot_shap_heatmap(shap_values_2d, class_name, sample_idx, exp_id, reports_dir):
    """
    Menggambar Heatmap SHAP (14 Channels x 256 Time Steps)
    """
    plt.figure(figsize=(12, 6))
    
    sns.heatmap(shap_values_2d, cmap='coolwarm', center=0, 
                yticklabels=CHANNELS, cbar_kws={'label': 'SHAP Value (Dampak Fitur)'})
    
    plt.title(f'SHAP Feature Relevance - Suku Kata ID: {class_name} (Sample #{sample_idx} | {exp_id})')
    plt.xlabel('Waktu (Titik Sampel 0 - 256)')
    plt.ylabel('Sensor EEG (14 Channels)')
    plt.tight_layout()
    
    filename = f"shap_heatmap_class_{class_name}_sample_{sample_idx}_{exp_id}.png"
    plt.savefig(os.path.join(reports_dir, filename))
    plt.close()
    print(f"[+] Heatmap disimpan: {filename}")

def run_explainability(exp_id="E0_Baseline"):
    print("\n" + "="*50)
    print(f" MEMULAI ANALISIS SHAP (EXPLAINABLE AI) UNTUK: {exp_id} ")
    print("="*50)

    paths = setup_experiment(exp_id)
    processed_dir = paths["processed_data"]
    weights_dir = paths["weights"]
    reports_dir = paths["reports"]

    print("[*] Memuat Model EEGNet dan Data Tensor...")
    try:
        model_path = os.path.join(weights_dir, f"eegnet_trained_{exp_id}.h5")
        model = load_model(model_path)
        
        data_path = os.path.join(processed_dir, "X_test.npy")
        X = np.load(data_path)
        
    except Exception as e:
        print(f"[X] Gagal memuat model/data untuk {exp_id}. Error: {e}")
        return

    print("[*] Menyiapkan Background Data untuk Explainer...")
    bg_size = min(100, X.shape[0])
    background_indices = np.random.choice(X.shape[0], bg_size, replace=False)
    background_data = X[background_indices]
    
    # Konversi ke format tensor secara eksplisit untuk GradientExplainer
    background_tensor = tf.cast(background_data, tf.float32)

    print("[*] Menjalankan Algoritma SHAP (GradientExplainer)...")
    try:
        # Menggunakan GradientExplainer yang aman untuk TensorFlow 2 Eager Execution
        explainer = shap.GradientExplainer(model, background_tensor)

        num_test_samples = min(3, X.shape[0])
        test_data = X[:num_test_samples]
        test_tensor = tf.cast(test_data, tf.float32)
        
        predictions = model.predict(test_data, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        shap_values = explainer.shap_values(test_tensor)

        print("\n[*] Membuat Visualisasi Heatmap...")
        for i in range(num_test_samples):
            pred_class = predicted_classes[i]
            
            # Memisahkan list array output dari GradientExplainer
            if isinstance(shap_values, list):
                sample_shap_2d = np.squeeze(shap_values[pred_class][i])
            else:
                sample_shap_2d = np.squeeze(shap_values[i])
            
            plot_shap_heatmap(sample_shap_2d, pred_class, i, exp_id, reports_dir)
            
    except Exception as e:
        print(f"[!] Gagal mengeksekusi kalkulasi SHAP: {e}. Modul Explainability dilewati.")

    print("\n" + "="*50)
    print(f" ANALISIS SHAP SELESAI. Cek folder: {reports_dir}/")
    print("="*50)

if __name__ == "__main__":
    run_explainability(exp_id="E0_Baseline")
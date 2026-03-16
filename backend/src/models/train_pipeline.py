import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from eegnet_model import EEGNetClassifier

# Import MLOps Tools
import optuna
import mlflow
import mlflow.tensorflow

# Konfigurasi Path
PROCESSED_DIR = "../../dataset/processed"
MODEL_DIR = "../../dataset/models"

def load_and_prepare_data():
    print("[*] Memuat dataset tensor...")
    X = np.load(os.path.join(PROCESSED_DIR, "X_features.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "y_labels.npy"))
    
    # Transposisi dari (Samples, Time, Channels) menjadi (Samples, Channels, Time)
    X = np.transpose(X, (0, 2, 1))
    X = np.expand_dims(X, axis=3) # Menambahkan depth untuk CNN
    return X, y

def plot_history(history, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Akurasi Latih')
    plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
    plt.title('Perkembangan Akurasi')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss Latih')
    plt.plot(history.history['val_loss'], label='Loss Validasi')
    plt.title('Penurunan Tingkat Kesalahan (Loss)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Siapkan Data
    X, y = load_and_prepare_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =======================================================
    # FASE MLOPS 1: OPTUNA HYPERPARAMETER TUNING
    # =======================================================
    print("\n" + "="*50)
    print(" MEMULAI OPTUNA HYPERPARAMETER TUNING ")
    print("="*50)

    # Konfigurasi MLflow Experiment
    mlflow.set_tracking_uri("sqlite:///mlruns.db") # Menggunakan SQLite ringan untuk log
    mlflow.set_experiment("EEGNet_Tuning")

    def objective(trial):
        """Fungsi objektif yang akan dicoba berulang kali oleh Optuna"""
        
        # Optuna akan menebak kombinasi parameter ini
        dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7, step=0.1)
        f1 = trial.suggest_categorical("F1", [4, 8, 16]) # Jumlah filter temporal
        d = trial.suggest_int("D", 2, 4)                 # Kedalaman filter spasial
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        
        with mlflow.start_run(nested=True):
            # Log parameter ke MLflow
            mlflow.log_params(trial.params)
            
            # Bangun arsitektur dengan parameter tebakan Optuna
            # F2 selalu di-set F1 * D sesuai anjuran jurnal asli
            eegnet = EEGNetClassifier(
                nb_classes=19, channels=14, samples=256,
                dropout_rate=dropout_rate, F1=f1, D=d, F2=f1*d
            )
            
            # Kita batasi epochs menjadi 100 saat tuning agar tidak terlalu lama
            history = eegnet.train(
                X_train, y_train, X_val, y_val, 
                epochs=100, batch_size=batch_size
            )
            
            # Ambil akurasi validasi terbaik dari pelatihan ini
            best_val_acc = max(history.history['val_accuracy'])
            
            # Log metrik ke MLflow
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            
            return best_val_acc

    # Buat study Optuna dan cari nilai maksimal (maximize)
    study = optuna.create_study(direction="maximize", study_name="EEGNet_Optimization")
    
    # Jalankan 10 percobaan (trials). Bisa dinaikkan ke 30-50 nanti jika Anda ada waktu luang.
    print("[*] Optuna akan melakukan 10 kali percobaan iterasi arsitektur...")
    study.optimize(objective, n_trials=10)

    print("\n[+] TUNING SELESAI!")
    print(f"    Akurasi Terbaik: {study.best_value * 100:.2f}%")
    print(f"    Parameter Terbaik: {study.best_params}")

    # =======================================================
    # FASE MLOPS 2: PELATIHAN MODEL TERBAIK & MODEL REGISTRY
    # =======================================================
    print("\n" + "="*50)
    print(" MELATIH MODEL FINAL DENGAN PARAMETER TERBAIK ")
    print("="*50)
    
    best_params = study.best_params
    f1 = best_params["F1"]
    d = best_params["D"]
    
    # Memulai MLflow run utama untuk Production
    with mlflow.start_run(run_name="Production_EEGNet"):
        mlflow.log_params(best_params)
        
        final_model = EEGNetClassifier(
            nb_classes=19, channels=14, samples=256,
            dropout_rate=best_params["dropout_rate"], 
            F1=f1, D=d, F2=f1*d
        )
        
        # Pelatihan penuh dengan 500 epochs (Early Stopping tetap aktif)
        history = final_model.train(
            X_train, y_train, X_val, y_val, 
            epochs=500, batch_size=best_params["batch_size"]
        )
        
        # Menyimpan model H5 secara lokal
        model_path = os.path.join(MODEL_DIR, "eegnet_trained.h5")
        final_model.save_model(model_path)
        
        # Menyimpan model ke dalam MLflow Model Registry
        mlflow.tensorflow.log_model(final_model.model, "eegnet_model_registry")
        mlflow.log_metric("final_val_accuracy", max(history.history['val_accuracy']))
        
        # Simpan dan log grafik
        plot_path = os.path.join(MODEL_DIR, "training_history.png")
        plot_history(history, plot_path)
        mlflow.log_artifact(plot_path)

    print(f"\n[SUCCESS] Model final dilatih dan disimpan di: {model_path}")
    print("[INFO] Ketik 'mlflow ui --backend-store-uri sqlite:///mlruns.db' di terminal untuk melihat dashboard pelacakan.")

if __name__ == "__main__":
    main()
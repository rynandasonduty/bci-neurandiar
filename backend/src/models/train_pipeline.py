import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from eegnet_model import EEGNetClassifier
import optuna
import mlflow
import mlflow.tensorflow

# Menghubungkan ke Mesin Direktori di config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment, MLFLOW_DB_PATH

def load_and_prepare_data(processed_dir):
    print(f"[*] Memuat dataset tensor dari: {processed_dir}")
    X = np.load(os.path.join(processed_dir, "X_features.npy"))
    y = np.load(os.path.join(processed_dir, "y_labels.npy"))
    
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

def run_training_pipeline(exp_id="E0_Baseline", n_trials=10, max_epochs=500):
    """
    Fungsi master untuk melatih EEGNet berdasarkan eksperimen tertentu.
    """
    print("\n" + "="*50)
    print(f" MEMULAI PIPELINE PELATIHAN UNTUK EKSPERIMEN: {exp_id} ")
    print("="*50)
    
    # 1. Panggil Mesin Pembangun Direktori
    paths = setup_experiment(exp_id)
    processed_dir = paths["processed_data"]
    weights_dir = paths["weights"]
    reports_dir = paths["reports"]
    
    # 2. Siapkan Data
    try:
        X, y = load_and_prepare_data(processed_dir)
    except FileNotFoundError:
        print(f"[X] Data tidak ditemukan di {processed_dir}. Jalankan build_dataset terlebih dahulu.")
        return
        
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Konfigurasi MLflow Experiment
    mlflow.set_tracking_uri(MLFLOW_DB_PATH) 
    mlflow.set_experiment(f"BCI_{exp_id}")

    def objective(trial):
        dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7, step=0.1)
        f1 = trial.suggest_categorical("F1", [4, 8, 16]) 
        d = trial.suggest_int("D", 2, 4)                 
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        
        with mlflow.start_run(nested=True):
            mlflow.log_params(trial.params)
            
            eegnet = EEGNetClassifier(
                nb_classes=19, channels=14, samples=X_train.shape[2],
                dropout_rate=dropout_rate, F1=f1, D=d, F2=f1*d
            )
            
            # Epochs lebih kecil saat tuning untuk menghemat waktu
            history = eegnet.train(
                X_train, y_train, X_val, y_val, 
                epochs=100, batch_size=batch_size
            )
            
            best_val_acc = max(history.history['val_accuracy'])
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            return best_val_acc

    # 4. Optuna Hyperparameter Tuning
    print(f"[*] Melakukan {n_trials} percobaan iterasi arsitektur...")
    study = optuna.create_study(direction="maximize", study_name=f"EEGNet_Opt_{exp_id}")
    study.optimize(objective, n_trials=n_trials)

    print("\n[+] TUNING SELESAI!")
    print(f"    Akurasi Terbaik: {study.best_value * 100:.2f}%")
    print(f"    Parameter Terbaik: {study.best_params}")

    # 5. Pelatihan Model Terbaik (Production Run)
    best_params = study.best_params
    f1 = best_params["F1"]
    d = best_params["D"]
    
    with mlflow.start_run(run_name=f"Production_EEGNet_{exp_id}"):
        mlflow.log_params(best_params)
        
        final_model = EEGNetClassifier(
            nb_classes=19, channels=14, samples=X_train.shape[2],
            dropout_rate=best_params["dropout_rate"], 
            F1=f1, D=d, F2=f1*d
        )
        
        history = final_model.train(
            X_train, y_train, X_val, y_val, 
            epochs=max_epochs, batch_size=best_params["batch_size"]
        )
        
        # Simpan model spesifik untuk eksperimen ini
        model_path = os.path.join(weights_dir, f"eegnet_trained_{exp_id}.h5")
        final_model.save_model(model_path)
        
        mlflow.tensorflow.log_model(final_model.model, f"eegnet_registry_{exp_id}")
        mlflow.log_metric("final_val_accuracy", max(history.history['val_accuracy']))
        
        plot_path = os.path.join(reports_dir, f"training_history_{exp_id}.png")
        plot_history(history, plot_path)
        mlflow.log_artifact(plot_path)
        
        print(f"\n[SUCCESS] Model final dilatih dan disimpan di: {model_path}")

if __name__ == "__main__":
    # Ini akan dieksekusi jika file dijalankan secara mandiri untuk testing
    run_training_pipeline(exp_id="E0_Baseline", n_trials=3, max_epochs=50)
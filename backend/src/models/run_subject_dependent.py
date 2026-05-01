import os
import sys
import time
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.tensorflow

# Pastikan bisa memanggil modul dari direktori root backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment, MLFLOW_DB_PATH
from preprocessing.build_dataset import DatasetBuilder
from models.eegnet_model import EEGNetClassifier

def load_data_for_subject(exp_id, subject_id, raw_data_dir):
    """
    Fungsi pembantu untuk mengekstrak data murni milik 1 subjek saja.
    """
    builder = DatasetBuilder(exp_id=exp_id)
    log_file = os.path.join(raw_data_dir, "logs", f"{subject_id}_experiment_log.txt")
    csv_files = glob.glob(os.path.join(raw_data_dir, f"{subject_id}*.csv"))
    
    if not os.path.exists(log_file) or not csv_files:
        print(f"[!] Data untuk {subject_id} tidak lengkap. Dilewati.")
        return None, None
        
    X_list, y_list = builder.process_subject(subject_id, csv_files[0], log_file)
    
    if len(X_list) == 0:
        return None, None
        
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Transposisi untuk EEGNet: (Samples, Channels, Time, Depth)
    X = np.transpose(X, (0, 2, 1))
    X = np.expand_dims(X, axis=3)
    
    return X, y

def execute_experiment_9_subject_dependent(max_epochs=200):
    """
    [EKSPERIMEN 9] Pelatihan Subject-Dependent.
    Membangun 1 model EEGNet eksklusif untuk setiap subjek yang ada di dataset.
    """
    exp_id = "E9_Subject_Dependent"
    print("\n" + "="*80)
    print(f"🚀 MEMULAI EKSEKUSI: {exp_id}")
    print("="*80)
    
    paths = setup_experiment(exp_id)
    raw_dir = paths["raw_data"]
    weights_dir = paths["weights"]
    
    mlflow.set_tracking_uri(MLFLOW_DB_PATH)
    mlflow.set_experiment(f"BCI_{exp_id}")
    
    # Cari semua subjek yang ada di folder raw
    log_files = glob.glob(os.path.join(raw_dir, "logs", "*_experiment_log.txt"))
    subject_ids = [os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files]
    
    for subject_id in subject_ids:
        print(f"\n[*] Memulai Pelatihan Individu untuk: {subject_id}")
        
        X, y = load_data_for_subject(exp_id, subject_id, raw_dir)
        if X is None: continue
            
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        with mlflow.start_run(run_name=f"E9_{subject_id}"):
            mlflow.log_param("subject_id", subject_id)
            
            # Instansiasi model dengan parameter standar
            model = EEGNetClassifier(
                nb_classes=19, channels=14, samples=X.shape[2],
                dropout_rate=0.5, F1=8, D=2, F2=16
            )
            
            history = model.train(X_train, y_train, X_val, y_val, epochs=max_epochs, batch_size=32)
            
            # Simpan model khusus subjek ini
            model_path = os.path.join(weights_dir, f"eegnet_{subject_id}.h5")
            model.save_model(model_path)
            
            best_acc = max(history.history['val_accuracy'])
            mlflow.log_metric("best_val_accuracy", best_acc)
            print(f"[SUCCESS] Model {subject_id} Selesai. Akurasi: {best_acc*100:.2f}%")

def execute_experiment_10_transfer_learning(target_subject_id, max_epochs_base=200, max_epochs_fine_tune=50):
    """
    [EKSPERIMEN 10] Transfer Learning.
    Melatih model dasar dari 11 subjek, membekukan layernya, lalu melatih ulang layer akhir dengan data target subjek.
    """
    exp_id = "E10_Transfer_Learning"
    print("\n" + "="*80)
    print(f"🚀 MEMULAI EKSEKUSI: {exp_id} (Target: {target_subject_id})")
    print("="*80)
    
    paths = setup_experiment(exp_id)
    raw_dir = paths["raw_data"]
    weights_dir = paths["weights"]
    
    mlflow.set_tracking_uri(MLFLOW_DB_PATH)
    mlflow.set_experiment(f"BCI_{exp_id}")
    
    log_files = glob.glob(os.path.join(raw_dir, "logs", "*_experiment_log.txt"))
    all_subjects = [os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files]
    
    if target_subject_id not in all_subjects:
        print(f"[X] Subjek target {target_subject_id} tidak ditemukan di dataset.")
        return
        
    source_subjects = [s for s in all_subjects if s != target_subject_id]
    
    # 1. Kumpulkan Data Base (Source Subjects)
    print(f"\n[STEP 1] Mengumpulkan data dari {len(source_subjects)} subjek sumber...")
    X_base, y_base = [], []
    for subj in source_subjects:
        X_s, y_s = load_data_for_subject(exp_id, subj, raw_dir)
        if X_s is not None:
            X_base.extend(X_s)
            y_base.extend(y_s)
            
    X_base = np.array(X_base)
    y_base = np.array(y_base)
    
    X_base_train, X_base_val, y_base_train, y_base_val = train_test_split(
        X_base, y_base, test_size=0.2, random_state=42, stratify=y_base
    )
    
    # 2. Kumpulkan Data Target
    print(f"\n[STEP 2] Memuat data target: {target_subject_id}...")
    X_target, y_target = load_data_for_subject(exp_id, target_subject_id, raw_dir)
    X_tgt_train, X_tgt_val, y_tgt_train, y_tgt_val = train_test_split(
        X_target, y_target, test_size=0.5, random_state=42, stratify=y_target # 50% latih, 50% uji untuk fine-tuning
    )
    
    with mlflow.start_run(run_name=f"E10_Target_{target_subject_id}"):
        mlflow.log_param("target_subject", target_subject_id)
        
        # 3. Latih Base Model
        print("\n[STEP 3] Melatih Base Model (Pre-training)...")
        base_model = EEGNetClassifier(
            nb_classes=19, channels=14, samples=X_base.shape[2],
            dropout_rate=0.5, F1=8, D=2, F2=16
        )
        base_model.train(X_base_train, y_base_train, X_base_val, y_base_val, epochs=max_epochs_base, batch_size=64)
        
        base_acc = max(base_model.model.history.history['val_accuracy'])
        mlflow.log_metric("base_val_accuracy", base_acc)
        
        # 4. Pembekuan Layer (Freezing) & Fine Tuning
        print("\n[STEP 4] Membekukan Layer Spasial & Temporal untuk Fine-Tuning...")
        for layer in base_model.model.layers:
            # Hanya biarkan layer klasifikasi (Dense/Softmax) yang bisa dilatih ulang
            if layer.name not in ['dense', 'softmax']:
                layer.trainable = False
                
        # Kompilasi ulang setelah pembekuan
        base_model.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        print("\n[STEP 5] Melakukan Fine-Tuning pada Target Subjek...")
        history_ft = base_model.model.fit(
            X_tgt_train, y_tgt_train,
            validation_data=(X_tgt_val, y_tgt_val),
            epochs=max_epochs_fine_tune,
            batch_size=16, # Batch size kecil untuk fine-tuning
            verbose=1
        )
        
        # 5. Simpan Model Akhir
        ft_acc = max(history_ft.history['val_accuracy'])
        mlflow.log_metric("finetune_val_accuracy", ft_acc)
        
        model_path = os.path.join(weights_dir, f"transfer_model_target_{target_subject_id}.h5")
        base_model.save_model(model_path)
        
        print(f"\n[SUCCESS] Transfer Learning Selesai.")
        print(f"Akurasi Base Model   : {base_acc*100:.2f}%")
        print(f"Akurasi Fine-Tuned   : {ft_acc*100:.2f}%")


if __name__ == "__main__":
    print("MENGAKTIFKAN PROTOKOL PEMODELAN INDIVIDU...")
    
    # Jalankan Eksperimen 9 (Train 12 model terpisah untuk 12 orang)
    # execute_experiment_9_subject_dependent(max_epochs=200)
    
    # Jalankan Eksperimen 10 (Simulasi Transfer Learning, misal subjek targetnya adalah "SUBJ12")
    # Ganti "SUBJ12" dengan ID subjek asli Anda nanti
    # execute_experiment_10_transfer_learning(target_subject_id="SUBJ12", max_epochs_base=200, max_epochs_fine_tune=50)
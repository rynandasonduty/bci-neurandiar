import os
import sys
import time
import glob
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment, MLFLOW_DB_PATH
from preprocessing.build_dataset import DatasetBuilder
from models.eegnet_model import EEGNetClassifier

# Utilitas pelindung kebocoran data
from utils.data_utils import three_way_split, fit_and_apply_scaler

EXPERIMENT_RECIPES = {
    "E0_Baseline": {
        "processor_params": {"band": "broadband", "apply_ica": False, "target_fs": 256},
        "crop_time": None, "phase_filter": "all", "channels_to_use": "all", "use_augmentation": False
    },
    "E1_ICA_Filtering": {
        "processor_params": {"band": "broadband", "apply_ica": True, "target_fs": 256},
        "crop_time": None, "phase_filter": "all", "channels_to_use": "all", "use_augmentation": False
    },
    "E2_Resampling_512Hz": {
        "processor_params": {"band": "broadband", "apply_ica": False, "target_fs": 512},
        "crop_time": None, "phase_filter": "all", "channels_to_use": "all", "use_augmentation": False
    },
    "E3_ERP_N400": {
        "processor_params": {"band": "broadband", "apply_ica": False, "target_fs": 256},
        "crop_time": (200, 600), "phase_filter": "all", "channels_to_use": "all", "use_augmentation": False
    },
    "E4_Channel_Language": {
        "processor_params": {"band": "broadband", "apply_ica": False, "target_fs": 256},
        "crop_time": None, "phase_filter": "all", "channels_to_use": ["EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7", "EEG.P7"], "use_augmentation": False
    },
    "E5_Data_Augmentation": {
        "processor_params": {"band": "broadband", "apply_ica": False, "target_fs": 256},
        "crop_time": None, "phase_filter": "all", "channels_to_use": "all", "use_augmentation": True,
        "augmentation_params": {"add_noise": True, "noise_factor": 0.05, "apply_jitter": True, "jitter_ms": 10}
    },
    "E6_CrossModality_ImaginedOnly": {
        "processor_params": {"band": "broadband", "apply_ica": False, "target_fs": 256},
        "crop_time": None, "phase_filter": "imagined", "channels_to_use": "all", "use_augmentation": False
    },
    "E7_Band_Alpha": {
        "processor_params": {"band": "alpha", "apply_ica": False, "target_fs": 256},
        "crop_time": None, "phase_filter": "all", "channels_to_use": "all", "use_augmentation": False
    }
}

def load_data_for_subject_grid(exp_id, subject_id, raw_data_dir, recipe):
    builder = DatasetBuilder(
        exp_id=f"GRID_{exp_id}", 
        processor_params=recipe.get("processor_params"),
        crop_time=recipe.get("crop_time"),
        use_augmentation=recipe.get("use_augmentation", False),
        augmentation_params=recipe.get("augmentation_params"),
        phase_filter=recipe.get("phase_filter", "all"),
        channels_to_use=recipe.get("channels_to_use", "all")
    )
    
    log_file = os.path.join(raw_data_dir, "logs", f"{subject_id}_experiment_log.txt")
    csv_files = glob.glob(os.path.join(raw_data_dir, f"{subject_id}*.csv"))
    
    if not os.path.exists(log_file) or not csv_files:
        return None, None
        
    X_list, y_list = builder.process_subject(subject_id, csv_files[0], log_file)
    
    if len(X_list) == 0:
        return None, None
        
    X = np.array(X_list)
    y = np.array(y_list)
    
    X = np.transpose(X, (0, 2, 1))
    X = np.expand_dims(X, axis=3)
    
    return X, y

def execute_grid_subject_dependent(max_epochs=150):
    grid_id = "E9_Grid_SubjectDependent"
    print("\n" + "="*80)
    print(f"🚀 MEMULAI GRID EKSPERIMEN: {grid_id}")
    print("="*80)
    
    paths = setup_experiment(grid_id)
    raw_dir = paths["raw_data"]
    weights_dir = paths["weights"]
    
    mlflow.set_tracking_uri(MLFLOW_DB_PATH)
    
    log_files = glob.glob(os.path.join(raw_dir, "logs", "*_experiment_log.txt"))
    subject_ids = [os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files]
    
    for exp_name, recipe in EXPERIMENT_RECIPES.items():
        print(f"\n[{exp_name}] Menginisialisasi Resep...")
        mlflow.set_experiment(f"BCI_Grid_{exp_name}")
        
        for subject_id in subject_ids:
            print(f"  👉 Pelatihan {subject_id} menggunakan resep {exp_name}...")
            
            X, y = load_data_for_subject_grid(exp_name, subject_id, raw_dir, recipe)
            if X is None: 
                print(f"      [!] Data tidak valid, dilewati.")
                continue
                
            X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y)
            
            scaler_path = os.path.join(weights_dir, f"scaler_{exp_name}_{subject_id}.pkl")
            X_train, X_val, X_test, scaler = fit_and_apply_scaler(X_train, X_val, X_test, save_path=scaler_path)
            
            use_augmentation = recipe.get("use_augmentation", False)
            if use_augmentation:
                from preprocessing.signal_processor import SignalProcessor
                aug_params = recipe.get("augmentation_params", {})
                proc = SignalProcessor(target_fs=recipe.get("processor_params", {}).get("target_fs", 256))  
                          
                aug_list = []
                # Asumsi X_train berbentuk (N, 14, 256, 1) atau sejenisnya
                # Sesuaikan np.squeeze(sample) agar menjadi (Time, Channels) untuk proc
                for sample in X_train:               
                    s2d = np.squeeze(sample).T       
                    aug = proc.apply_augmentation(s2d, **aug_params)
                    aug_list.append(np.expand_dims(aug.T, -1))
                    
                X_aug = np.array(aug_list)
                X_train = np.concatenate([X_train, X_aug], axis=0)
                y_train = np.concatenate([y_train, y_train], axis=0)
                
                # Acak ulang (shuffle) agar data asli dan augmentasi bercampur
                shuffle_idx = np.random.permutation(len(y_train))
                X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]
                print(f"      [+] E5 Augmentasi berhasil: Total X_train menjadi {len(X_train)} sampel")
            
            np.save(os.path.join(weights_dir, f"Xtest_{exp_name}_{subject_id}.npy"), X_test)
            np.save(os.path.join(weights_dir, f"ytest_{exp_name}_{subject_id}.npy"), y_test)
            
            with mlflow.start_run(run_name=f"{exp_name}_{subject_id}"):
                mlflow.log_param("subject_id", subject_id)
                mlflow.log_param("experiment_type", exp_name)
                
                model = EEGNetClassifier(
                    nb_classes=19, channels=X_train.shape[1], samples=X_train.shape[2],
                    dropout_rate=0.5, F1=8, D=2, F2=16
                )
                
                # [PERBAIKAN AUDIT] Memanggil model.train() agar mewarisi ReduceLROnPlateau dan kesabaran dinamis
                history = model.train(
                    X_train, y_train, X_val, y_val,
                    epochs=max_epochs, batch_size=32
                )
                
                best_acc = max(history.history['val_accuracy'])
                mlflow.log_metric("best_val_accuracy", best_acc)
                
                model_path = os.path.join(weights_dir, f"{exp_name}_{subject_id}.h5")
                model.save_model(model_path)
                
                print(f"      [SUCCESS] Akurasi Validasi: {best_acc*100:.2f}%")

def execute_grid_transfer_learning(target_subject_id, max_epochs_base=150, max_epochs_ft=50):
    """
    [EKSPERIMEN 10 EXTENDED] Menguji skenario Transfer Learning di berbagai resep (E0-E7).
    """
    grid_id = "E10_Grid_TransferLearning"
    print("\n" + "="*80)
    print(f"🚀 MEMULAI GRID TRANSFER LEARNING: Target ({target_subject_id})")
    print("="*80)
    
    paths = setup_experiment(grid_id)
    raw_dir = paths["raw_data"]
    weights_dir = paths["weights"]
    
    mlflow.set_tracking_uri(MLFLOW_DB_PATH)
    
    log_files = glob.glob(os.path.join(raw_dir, "logs", "*_experiment_log.txt"))
    all_subjects = [os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files]
    source_subjects = [s for s in all_subjects if s != target_subject_id]
    
    for exp_name, recipe in EXPERIMENT_RECIPES.items():
        print(f"\n[{exp_name}] Membangun Base Model dari {len(source_subjects)} Subjek...")
        mlflow.set_experiment(f"BCI_Grid_TL_{exp_name}")
        
        X_base, y_base = [], []
        for subj in source_subjects:
            X_s, y_s = load_data_for_subject_grid(exp_name, subj, raw_dir, recipe)
            if X_s is not None:
                X_base.extend(X_s)
                y_base.extend(y_s)
                
        if not X_base: continue
        X_base, y_base = np.array(X_base), np.array(y_base)
        
        # Split Data Base Model (Subjek Sumber)
        X_b_train, X_b_val, X_b_test, y_b_train, y_b_val, y_b_test = three_way_split(X_base, y_base)
        X_b_train, X_b_val, X_b_test, _ = fit_and_apply_scaler(X_b_train, X_b_val, X_b_test)
        
        # Split Data Target (Fine Tuning)
        X_tgt, y_tgt = load_data_for_subject_grid(exp_name, target_subject_id, raw_dir, recipe)
        if X_tgt is None: continue
        X_t_train, X_t_val, X_t_test, y_t_train, y_t_val, y_t_test = three_way_split(X_tgt, y_tgt)
        X_t_train, X_t_val, X_t_test, _ = fit_and_apply_scaler(X_t_train, X_t_val, X_t_test)
        
        with mlflow.start_run(run_name=f"TL_{exp_name}_Target_{target_subject_id}"):
            mlflow.log_param("target_subject", target_subject_id)
            
            print(f"  👉 Pre-training Base Model...")
            base_model = EEGNetClassifier(
                nb_classes=19, channels=X_b_train.shape[1], samples=X_b_train.shape[2],
                dropout_rate=0.5, F1=8, D=2, F2=16
            )
            base_model.model.fit(X_b_train, y_b_train, validation_data=(X_b_val, y_b_val), epochs=max_epochs_base, batch_size=64, verbose=0)
            base_acc = max(base_model.model.history.history['val_accuracy'])
            mlflow.log_metric("base_val_accuracy", base_acc)
            
            print(f"  👉 Fine-tuning ke Subjek {target_subject_id}...")
            for layer in base_model.model.layers:
                if layer.name not in ['dense', 'softmax']:
                    layer.trainable = False
            base_model.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            history_ft = base_model.model.fit(
                X_t_train, y_t_train, validation_data=(X_t_val, y_t_val),
                epochs=max_epochs_ft, batch_size=16, verbose=0
            )
            
            ft_acc = max(history_ft.history['val_accuracy'])
            mlflow.log_metric("finetune_val_accuracy", ft_acc)
            
            model_path = os.path.join(weights_dir, f"TL_{exp_name}_target_{target_subject_id}.h5")
            base_model.save_model(model_path)
            
            print(f"      [SUCCESS] Base Acc: {base_acc*100:.2f}% | Fine-Tuned Acc: {ft_acc*100:.2f}%")

if __name__ == "__main__":
    print("MENGAKTIFKAN PROTOKOL 2D GRID EXPERIMENT...")
    
    execute_grid_subject_dependent(max_epochs=150)
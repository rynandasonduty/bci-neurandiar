import os
import sys
import time
import glob
import numpy as np
import mlflow
import mlflow.tensorflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment, MLFLOW_DB_PATH
from preprocessing.build_dataset import DatasetBuilder
from models.eegnet_model import EEGNetClassifier

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
    print("\n[INFO] Initiating subject-dependent grid experiment (P2_EEGNet, E0-E7).")

    # Resolve raw data directory via any valid experiment path
    raw_dir = setup_experiment("E0_Baseline", pilar="P2_EEGNet")["raw_data"]

    mlflow.set_tracking_uri(MLFLOW_DB_PATH)

    log_files = glob.glob(os.path.join(raw_dir, "logs", "*_experiment_log.txt"))
    subject_ids = [os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files]

    for exp_name, recipe in EXPERIMENT_RECIPES.items():
        # Each experiment resolves its own Golden Standard path under P2_EEGNet
        paths = setup_experiment(exp_name, pilar="P2_EEGNet")
        weights_dir = paths["weights"]

        print(f"\n[{exp_name}] Initializing recipe for P2_EEGNet...")
        mlflow.set_experiment(f"BCI_Grid_{exp_name}")
        
        for subject_id in subject_ids:
            print(f"  [INFO] Training {subject_id} with recipe {exp_name}...")

            X, y = load_data_for_subject_grid(exp_name, subject_id, raw_dir, recipe)
            if X is None:
                print(f"      [WARNING] No valid data for {subject_id}. Skipping.")
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
                for sample in X_train:
                    s2d = np.squeeze(sample).T
                    aug = proc.apply_augmentation(s2d, **aug_params)
                    aug_list.append(np.expand_dims(aug.T, -1))

                X_aug = np.array(aug_list)
                X_train = np.concatenate([X_train, X_aug], axis=0)
                y_train = np.concatenate([y_train, y_train], axis=0)

                shuffle_idx = np.random.permutation(len(y_train))
                X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]
                print(f"      [INFO] E5 augmentation applied. Training set expanded to {len(X_train)} samples.")
            
            np.save(os.path.join(weights_dir, f"Xtest_{exp_name}_{subject_id}.npy"), X_test)
            np.save(os.path.join(weights_dir, f"ytest_{exp_name}_{subject_id}.npy"), y_test)
            
            with mlflow.start_run(run_name=f"{exp_name}_{subject_id}"):
                mlflow.log_param("subject_id", subject_id)
                mlflow.log_param("experiment_type", exp_name)
                
                model = EEGNetClassifier(
                    nb_classes=19, channels=X_train.shape[1], samples=X_train.shape[2],
                    dropout_rate=0.5, F1=8, D=2, F2=16
                )
                
                history = model.train(
                    X_train, y_train, X_val, y_val,
                    epochs=max_epochs, batch_size=32
                )
                
                best_acc = max(history.history['val_accuracy'])
                mlflow.log_metric("best_val_accuracy", best_acc)
                
                model_path = os.path.join(weights_dir, f"{exp_name}_{subject_id}.h5")
                model.save_model(model_path)
                
                print(f"      [INFO] Validation accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    print("[INFO] Activating subject-dependent grid experiment protocol (P2_EEGNet)...")
    execute_grid_subject_dependent(max_epochs=150)
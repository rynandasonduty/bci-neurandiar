import os
import sys
import glob
import numpy as np
import mlflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment, MLFLOW_DB_PATH
from preprocessing.build_dataset import DatasetBuilder
from features.extract_eeg_features import EEGFeatureExtractor
from models.classical_models import ClassicalClassifier
from run_subject_dependent import EXPERIMENT_RECIPES

from utils.data_utils import three_way_split, fit_and_apply_scaler

# Five feature groups for the SVM ablation study (yields 480 models total)
SELECTED_FEAT_GROUPS = ['time', 'hjorth', 'barlow', 'band_ratio', 'all']

def load_3d_data(exp_id, subject_id, raw_data_dir, recipe):
    builder = DatasetBuilder(
        exp_id=f"GRID_{exp_id}",
        processor_params=recipe.get("processor_params"),
        crop_time=recipe.get("crop_time"),
        use_augmentation=False,  # Augmentation applied post-split; disabled here to prevent leakage
        phase_filter=recipe.get("phase_filter", "all"),
        channels_to_use=recipe.get("channels_to_use", "all")
    )
    
    log_file = os.path.join(raw_data_dir, "logs", f"{subject_id}_experiment_log.txt")
    csv_files = glob.glob(os.path.join(raw_data_dir, f"{subject_id}*.csv"))
    if not os.path.exists(log_file) or not csv_files: return None, None
    
    X_list, y_list = builder.process_subject(subject_id, csv_files[0], log_file)
    if len(X_list) == 0: return None, None
    X_3d = np.transpose(np.array(X_list), (0, 2, 1))
    X_3d = np.expand_dims(X_3d, axis=-1)
    return X_3d, np.array(y_list)

def execute_e8_classical_grid():
    print("\n[INFO] Initiating E8 classical ML grid experiment (SVM feature ablation -- 480 models).")

    # Resolve raw data directory via any valid experiment path
    raw_dir = setup_experiment("E0_Baseline", pilar="P3_SVM")["raw_data"]

    mlflow.set_tracking_uri(MLFLOW_DB_PATH)

    log_files = glob.glob(os.path.join(raw_dir, "logs", "*_experiment_log.txt"))
    subject_ids = [os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files]

    for exp_name, recipe in EXPERIMENT_RECIPES.items():
        # Each experiment resolves its own Golden Standard path under P3_SVM
        weights_dir = setup_experiment(exp_name, pilar="P3_SVM")["weights"]

        print(f"\n[{exp_name}] Entering SVM training loop...")
        
        for feat_group in SELECTED_FEAT_GROUPS:
            mlflow.set_experiment(f"BCI_Grid_E8_SVM_{feat_group}_{exp_name}")
            print(f"  [INFO] Extracting and training feature group: {feat_group.upper()}")

            for subject_id in subject_ids:

                # Auto-resume: skip subjects whose model artifact already exists
                model_path = os.path.join(weights_dir, f"SVM_{feat_group}_{exp_name}_{subject_id}.pkl")
                if os.path.exists(model_path):
                    print(f"      [SKIP] Model for {feat_group}/{subject_id} already exists.")
                    continue

                # 1. Load raw 3D EEG data for this subject
                X_3d, y = load_3d_data(exp_name, subject_id, raw_dir, recipe)
                if X_3d is None: continue

                # 2. Stratified three-way split (while still 3D)
                X_train_3d, X_val_3d, X_test_3d, y_train, y_val, y_test = three_way_split(X_3d, y)

                # 3. Apply augmentation to training split (3D, pre-feature extraction)
                use_augmentation = recipe.get("use_augmentation", False)
                if use_augmentation:
                    from preprocessing.signal_processor import SignalProcessor
                    aug_params = recipe.get("augmentation_params", {})
                    proc = SignalProcessor(target_fs=recipe.get("processor_params", {}).get("target_fs", 256))
                    aug_list = []
                    for sample in X_train_3d:
                        s2d = np.squeeze(sample).T
                        aug = proc.apply_augmentation(s2d, **aug_params)
                        aug_list.append(np.expand_dims(aug.T, -1))

                    X_aug = np.array(aug_list)
                    X_train_3d = np.concatenate([X_train_3d, X_aug], axis=0)
                    y_train = np.concatenate([y_train, y_train], axis=0)

                    shuffle_idx = np.random.permutation(len(y_train))
                    X_train_3d, y_train = X_train_3d[shuffle_idx], y_train[shuffle_idx]
                    print(f"      [INFO] E5 augmentation applied. Training set: {len(X_train_3d)} samples.")

                # 4. Extract handcrafted features (3D -> 2D)
                target_fs = recipe.get("processor_params", {}).get("target_fs", 256)
                extractor = EEGFeatureExtractor(fs=target_fs)
                groups = None if feat_group == 'all' else [feat_group]

                # Extractor expects shape (N, C, T) without the depth dimension
                X_train = extractor.transform(np.squeeze(X_train_3d, axis=-1), groups=groups)
                X_val = extractor.transform(np.squeeze(X_val_3d, axis=-1), groups=groups)
                X_test = extractor.transform(np.squeeze(X_test_3d, axis=-1), groups=groups)

                # Sanitize NaN/Inf values that may arise from ICA-affected feature extraction
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

                # 5. Anti-leakage feature scaling
                scaler_path = os.path.join(weights_dir, f"scaler_SVM_{feat_group}_{exp_name}_{subject_id}.pkl")
                X_train, X_val, X_test, scaler = fit_and_apply_scaler(X_train, X_val, X_test, save_path=scaler_path)

                # Persist test set to disk
                np.save(os.path.join(weights_dir, f"Xtest_SVM_{feat_group}_{exp_name}_{subject_id}.npy"), X_test)
                np.save(os.path.join(weights_dir, f"ytest_SVM_{feat_group}_{exp_name}_{subject_id}.npy"), y_test)

                with mlflow.start_run(run_name=f"SVM_{feat_group}_{exp_name}_{subject_id}"):
                    mlflow.log_param("subject_id", subject_id)
                    mlflow.log_param("preprocessing", exp_name)
                    mlflow.log_param("model_type", "SVM")
                    mlflow.log_param("feature_group", feat_group)

                    model = ClassicalClassifier(model_type='svm', C=10)
                    model.train(X_train, y_train)

                    val_acc = model.evaluate(X_val, y_val)
                    mlflow.log_metric("best_val_accuracy", val_acc)

                    model.save_model(model_path)

                    print(f"      [INFO] Subject: {subject_id} | Val accuracy: {val_acc*100:.2f}%")

if __name__ == "__main__":
    execute_e8_classical_grid()
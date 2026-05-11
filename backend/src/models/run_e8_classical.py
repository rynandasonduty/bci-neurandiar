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

# [PERBAIKAN] Impor utilitas pemecah data dan scaler anti-leakage
from utils.data_utils import three_way_split, fit_and_apply_scaler

# Definisi 5 Kelompok Fitur Taktis untuk Ablasi SVM (Total 480 Model)
SELECTED_FEAT_GROUPS = ['time', 'hjorth', 'barlow', 'band_ratio', 'all']

# 1. GANTI FUNGSI LAMA DENGAN INI (Hanya me-load 3D, tanpa ekstraksi)
def load_3d_data(exp_id, subject_id, raw_data_dir, recipe):
    builder = DatasetBuilder(
        exp_id=f"GRID_{exp_id}",
        processor_params=recipe.get("processor_params"),
        crop_time=recipe.get("crop_time"),
        use_augmentation=False, # Dimatikan agar tidak double
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
    grid_id = "E8_ML_Klasik_SubjectDependent"
    print("\n" + "="*80)
    print(f"🚀 MEMULAI GRID EKSPERIMEN E8: ABLASI 5 FITUR SVM (480 MODEL)")
    print("="*80)
    
    paths = setup_experiment(grid_id)
    raw_dir = paths["raw_data"]
    weights_dir = paths["weights"]
    
    mlflow.set_tracking_uri(MLFLOW_DB_PATH)
    
    log_files = glob.glob(os.path.join(raw_dir, "logs", "*_experiment_log.txt"))
    subject_ids = [os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files]
    
    for exp_name, recipe in EXPERIMENT_RECIPES.items():
        print(f"\n[{exp_name}] Memasuki Arena SVM...")
        
        for feat_group in SELECTED_FEAT_GROUPS:
            mlflow.set_experiment(f"BCI_Grid_E8_SVM_{feat_group}_{exp_name}")
            print(f"  ⚡ Ekstraksi & Pelatihan Grup Fitur: {feat_group.upper()}")
            
            for subject_id in subject_ids:
                
                # =======================================================
                # [PERBAIKAN 1] SISTEM AUTO-RESUME (SKIP MODEL YANG SUDAH ADA)
                model_path = os.path.join(weights_dir, f"SVM_{feat_group}_{exp_name}_{subject_id}.pkl")
                if os.path.exists(model_path):
                    print(f"      [SKIP] Model {feat_group} untuk {subject_id} sudah ada. Melanjutkan...")
                    continue
                # =======================================================

                # 1. Load data 3D Mentah
                X_3d, y = load_3d_data(exp_name, subject_id, raw_dir, recipe)
                if X_3d is None: continue
                
                # 2. Split (Saat masih 3D)
                X_train_3d, X_val_3d, X_test_3d, y_train, y_val, y_test = three_way_split(X_3d, y)
                
                # 3. AUGMENTASI X_TRAIN (Saat masih 3D)
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
                    print(f"      [+] E5 Augmentasi 3D berhasil: {len(X_train_3d)} sampel")

                # 4. EKSTRAKSI FITUR (Ubah 3D menjadi 2D)
                target_fs = recipe.get("processor_params", {}).get("target_fs", 256)
                extractor = EEGFeatureExtractor(fs=target_fs)
                groups = None if feat_group == 'all' else [feat_group]
                
                # Ingat, extractor mengharapkan (N, C, T) tanpa depth dimension (-1)
                X_train = extractor.transform(np.squeeze(X_train_3d, axis=-1), groups=groups)
                X_val = extractor.transform(np.squeeze(X_val_3d, axis=-1), groups=groups)
                X_test = extractor.transform(np.squeeze(X_test_3d, axis=-1), groups=groups)
                
                # =======================================================
                # [PERBAIKAN 2] PEMBERSIH NaN & Infinity akibat ICA
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
                # =======================================================

                # 5. SCALING FITUR 2D
                scaler_path = os.path.join(weights_dir, f"scaler_SVM_{feat_group}_{exp_name}_{subject_id}.pkl")
                X_train, X_val, X_test, scaler = fit_and_apply_scaler(X_train, X_val, X_test, save_path=scaler_path)
                
                # [PERBAIKAN BEST PRACTICE] Simpan Test Set permanen ke disk
                np.save(os.path.join(weights_dir, f"Xtest_SVM_{feat_group}_{exp_name}_{subject_id}.npy"), X_test)
                np.save(os.path.join(weights_dir, f"ytest_SVM_{feat_group}_{exp_name}_{subject_id}.npy"), y_test)
                
                with mlflow.start_run(run_name=f"SVM_{feat_group}_{exp_name}_{subject_id}"):
                    mlflow.log_param("subject_id", subject_id)
                    mlflow.log_param("preprocessing", exp_name)
                    mlflow.log_param("model_type", "SVM")
                    mlflow.log_param("feature_group", feat_group)
                    
                    # Inisialisasi dan latih model (Hanya SVM)
                    model = ClassicalClassifier(model_type='svm', C=10)
                    model.train(X_train, y_train)
                    
                    # Evaluasi pada Validation Set 
                    val_acc = model.evaluate(X_val, y_val)
                    mlflow.log_metric("best_val_accuracy", val_acc)
                    
                    # Simpan bobot (.pkl)
                    # model_path sudah didefinisikan di awal loop (Sistem Auto-Resume)
                    model.save_model(model_path)
                    
                    print(f"      👉 Subjek: {subject_id} | Val Acc: {val_acc*100:.2f}%")

if __name__ == "__main__":
    execute_e8_classical_grid()
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

def load_and_extract_features(exp_id, subject_id, raw_data_dir, recipe, feat_group):
    """
    Menarik data mentah 1 subjek, memprosesnya, lalu mengubahnya menjadi fitur matematis.
    Dilengkapi dengan perlindungan dimensi (Transpose) dan Ablasi Fitur Dinamis.
    """
    builder = DatasetBuilder(
        exp_id=f"GRID_{exp_id}",
        processor_params=recipe.get("processor_params"),
        crop_time=recipe.get("crop_time"),
        # Augmentasi diabaikan pada fase ini sesuai perbaikan anti-leakage DatasetBuilder
        use_augmentation=recipe.get("use_augmentation", False),
        augmentation_params=recipe.get("augmentation_params"),
        phase_filter=recipe.get("phase_filter", "all"),
        channels_to_use=recipe.get("channels_to_use", "all")
    )
    
    log_file = os.path.join(raw_data_dir, "logs", f"{subject_id}_experiment_log.txt")
    csv_files = glob.glob(os.path.join(raw_data_dir, f"{subject_id}*.csv"))
    
    if not os.path.exists(log_file) or not csv_files:
        return None, None
        
    # Tarik data Tensor 3D Mentah (Samples, Waktu, Channels)
    X_list, y_list = builder.process_subject(subject_id, csv_files[0], log_file)
    if len(X_list) == 0: return None, None
    
    X_3d = np.array(X_list)
    
    # [PERBAIKAN KRITIS] Transpose dimensi dari (N, T, C) menjadi (N, C, T) WAJIB untuk Ekstraktor
    X_3d = np.transpose(X_3d, (0, 2, 1))
    y = np.array(y_list)
    
    # Ekstraksi Fitur 2D (Samples, Features) dengan Ablasi Terpilih
    target_fs = recipe.get("processor_params", {}).get("target_fs", 256)
    extractor = EEGFeatureExtractor(fs=target_fs)
    
    groups = None if feat_group == 'all' else [feat_group]
    X_2d = extractor.transform(X_3d, groups=groups)
    
    return X_2d, y

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
                
                X, y = load_and_extract_features(exp_name, subject_id, raw_dir, recipe, feat_group)
                if X is None: continue
                
                # [PERBAIKAN KRITIS] 3-way Split Anti-Leakage
                X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y)
                
                # [PERBAIKAN KRITIS #2] Fit scaler HANYA pada data train.
                # Karena ClassicalClassifier sudah kita bersihkan dari StandardScaler,
                # data yang masuk ke model dari sini sudah merupakan data final yang benar secara statistik.
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
                    model_path = os.path.join(weights_dir, f"SVM_{feat_group}_{exp_name}_{subject_id}.pkl")
                    model.save_model(model_path)
                    
                    print(f"      👉 Subjek: {subject_id} | Val Acc: {val_acc*100:.2f}%")

if __name__ == "__main__":
    execute_e8_classical_grid()
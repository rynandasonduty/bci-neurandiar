import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.build_dataset import DatasetBuilder
from preprocessing.build_logreg_dataset import LogRegDatasetBuilder
from models.train_pipeline import run_training_pipeline
from models.evaluate_model import evaluate_system
from models.explain_model import run_explainability

def execute_experiment(exp_id, processor_params=None, crop_time=None, 
                       use_augmentation=False, augmentation_params=None,
                       phase_filter="all", channels_to_use="all",
                       n_trials_optuna=5, max_epochs=300):
    """
    Fungsi eksekusi tunggal untuk satu skenario eksperimen penuh (End-to-End).
    """
    print("\n" + "="*80)
    print(f"🚀 MEMULAI EKSEKUSI PENUH: {exp_id}")
    print("="*80)
    
    start_time = time.time()

    # 1. Bikin Dataset Utama (Suku Kata)
    print(f"\n[STEP 1/5] Membangun Dataset Utama untuk {exp_id}...")
    builder = DatasetBuilder(
        exp_id=exp_id, 
        processor_params=processor_params, 
        crop_time=crop_time,
        use_augmentation=use_augmentation,
        augmentation_params=augmentation_params,
        phase_filter=phase_filter,
        channels_to_use=channels_to_use
    )
    builder.build_full_dataset()

    # 2. Latih EEGNet (Termasuk Tuning Optuna)
    print(f"\n[STEP 2/5] Melatih Model Deep Learning (EEGNet) untuk {exp_id}...")
    run_training_pipeline(exp_id=exp_id, n_trials=n_trials_optuna, max_epochs=max_epochs)

    # 3. Bikin Dataset LogReg dan Latih Word Assembler
    print(f"\n[STEP 3/5] Melatih Word Assembler (Logistic Regression) untuk {exp_id}...")
    try:
        logreg_builder = LogRegDatasetBuilder(
            exp_id=exp_id,
            processor_params=processor_params,
            crop_time=crop_time,
            phase_filter=phase_filter,
            channels_to_use=channels_to_use
        )
        logreg_builder.build_dataset()
        
        # Latih model assembler (diatur oleh kelas WordAssembler itu sendiri)
        from logreg_model import WordAssembler
        assembler = WordAssembler(exp_id=exp_id)
        
        # Load X_word dan y_word
        import numpy as np
        from config import setup_experiment         # <--- Tambahkan baris ini
        paths = setup_experiment(exp_id)            # <--- Ganti builder.paths menjadi ini
        X_w = np.load(os.path.join(paths["processed_data"], "X_word_features.npy"))
        y_w = np.load(os.path.join(paths["processed_data"], "y_word_labels.npy"))
        
        assembler.train(X_w, y_w)
        assembler.save_model()
    except Exception as e:
        print(f"[!] Peringatan: Gagal melatih Word Assembler. Lanjut ke step berikutnya. Error: {e}")

    # 4. Evaluasi Kinerja (Confusion Matrix)
    print(f"\n[STEP 4/5] Evaluasi Metrik & Visualisasi untuk {exp_id}...")
    evaluate_system(exp_id=exp_id)

    # 5. Analisis SHAP (Explainable AI)
    print(f"\n[STEP 5/5] Analisis Explainability (SHAP) untuk {exp_id}...")
    run_explainability(exp_id=exp_id)
    
    end_time = time.time()
    mins = (end_time - start_time) // 60
    print("\n" + "="*80)
    print(f"✅ EKSEKUSI SELESAI: {exp_id} (Waktu Tempuh: {mins:.0f} Menit)")
    print("="*80)


if __name__ == "__main__":
    # =================================================================
    # DAFTAR 8 ANTREAN EKSPERIMEN (Hapus tanda '#' untuk menjalankan)
    # =================================================================
    
    print("MEMULAI ORKESTRASI EKSPERIMEN...")

    # # # --- 0. EXPERIMENT BASELINE (WAJIB JALAN PERTAMA) ---
    execute_experiment(
        exp_id="E0_Baseline",
        processor_params={"band": "broadband", "apply_ica": False, "target_fs": 256},
        crop_time=None, 
        phase_filter="all", # Baseline menggunakan data overt & imagined
        channels_to_use="all",
        n_trials_optuna=5,
        max_epochs=300
    )

    # # # --- 1. EXPERIMENT ICA (MEMBERSIHKAN ARTEFAK MATA) ---
    execute_experiment(
        exp_id="E1_ICA_Filtering",
        processor_params={"band": "broadband", "apply_ica": True, "target_fs": 256},
        crop_time=None
    )

    # # # --- 2. EXPERIMENT RESAMPLING (UPSAMPLING KE 512 Hz) ---
    execute_experiment(
        exp_id="E2_Resampling_512Hz",
        processor_params={"band": "broadband", "apply_ica": False, "target_fs": 512},
        crop_time=None
    )

    # # # --- 3. EXPERIMENT ERP CROPPING (N400: FASE SEMANTIK 200-600ms) ---
    execute_experiment(
        exp_id="E3_ERP_N400",
        processor_params={"band": "broadband", "apply_ica": False, "target_fs": 256},
        crop_time=(200, 600) 
    )

    # # # --- 4. EXPERIMENT CHANNEL ABLATION (AREA BAHASA: BROCA & WERNICKE) ---
    execute_experiment(
        exp_id="E4_Channel_Language",
        processor_params={"band": "broadband", "apply_ica": False, "target_fs": 256},
        channels_to_use=["EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7", "EEG.P7"],
        crop_time=None
    )

    # # # --- 5. EXPERIMENT AUGMENTASI (NOISE INJECTION & JITTERING) ---
    execute_experiment(
        exp_id="E5_Data_Augmentation",
        processor_params={"band": "broadband", "apply_ica": False, "target_fs": 256},
        use_augmentation=True,
        augmentation_params={"add_noise": True, "noise_factor": 0.05, "apply_jitter": True, "jitter_ms": 10},
        crop_time=None
    )

    # # # --- 6. EXPERIMENT CROSS-MODALITY (HANYA MENGGUNAKAN DATA IMAGINED) ---
    # # Membandingkan hasilnya dengan E0_Baseline (yang menggabungkan overt+imagined)
    execute_experiment(
        exp_id="E6_CrossModality_ImaginedOnly",
        processor_params={"band": "broadband", "apply_ica": False, "target_fs": 256},
        phase_filter="imagined",
        crop_time=None
    )

    # --- 7. EXPERIMENT PITA FREKUENSI (ISOLASI GELOMBANG ALPHA) ---
    execute_experiment(
        exp_id="E7_Band_Alpha",
        processor_params={"band": "alpha", "apply_ica": False, "target_fs": 256},
        crop_time=None
    )
    
    print("\n🎉 SEMUA ANTREAN EKSPERIMEN TELAH SELESAI DIEKSEKUSI!")
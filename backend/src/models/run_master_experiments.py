import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.build_dataset import DatasetBuilder
from preprocessing.build_logreg_dataset import LogRegDatasetBuilder
from models.train_pipeline import run_training_pipeline
from models.evaluate_model import evaluate_system
from models.explain_model import run_explainability
from run_subject_dependent import EXPERIMENT_RECIPES

def execute_experiment(exp_id, processor_params=None, crop_time=None,
                       use_augmentation=False, augmentation_params=None,
                       phase_filter="all", channels_to_use="all",
                       n_trials_optuna=5, max_epochs=300):
    """
    Execute a complete end-to-end experiment pipeline for a single ablation configuration.

    The pipeline proceeds through five sequential stages:
      1. Dataset construction from raw EEG recordings.
      2. EEGNet training with Optuna hyperparameter optimization.
      3. Word assembler (Logistic Regression) dataset construction and training.
      4. System evaluation on the held-out test set (syllable and word accuracy).
      5. SHAP explainability analysis.

    Args:
        exp_id (str): Experiment identifier (e.g., 'E0_Baseline').
        processor_params (dict): Signal processing parameters for SignalProcessor.
        crop_time (tuple): ERP time window in milliseconds (start_ms, end_ms), or None for baseline windowing.
        use_augmentation (bool): Whether to apply data augmentation during training.
        augmentation_params (dict): Parameters for SignalProcessor.apply_augmentation().
        phase_filter (str): Recording phase to include: 'all', 'overt', or 'imagined'.
        channels_to_use (str or list): Channel subset for E4 ablation, or 'all'.
        n_trials_optuna (int): Number of Optuna search trials.
        max_epochs (int): Maximum training epochs for the final production model.
    """
    print(f"\n[INFO] Initiating full experiment execution: {exp_id}")

    start_time = time.time()

    # Step 1: Build primary syllable dataset
    print(f"\n[STEP 1/5] Building primary dataset for {exp_id}...")
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

    # Step 2: Train EEGNet with Optuna hyperparameter search
    print(f"\n[STEP 2/5] Training EEGNet model for {exp_id}...")

    recipe = EXPERIMENT_RECIPES[exp_id]

    run_training_pipeline(
        exp_id=exp_id,
        n_trials=n_trials_optuna,
        max_epochs=max_epochs,
        use_augmentation=recipe.get("use_augmentation", False),
        augmentation_params=recipe.get("augmentation_params", {}),
        target_fs=recipe.get("processor_params", {}).get("target_fs", 256)
    )

    # Step 3: Build word assembler dataset and train Logistic Regression
    print(f"\n[STEP 3/5] Training word assembler (Logistic Regression) for {exp_id}...")
    try:
        logreg_builder = LogRegDatasetBuilder(
            exp_id=exp_id,
            processor_params=processor_params,
            crop_time=crop_time,
            phase_filter=phase_filter,
            channels_to_use=channels_to_use
        )
        logreg_builder.build_dataset()

        from logreg_model import WordAssembler
        assembler = WordAssembler(exp_id=exp_id)

        import numpy as np
        from config import setup_experiment
        paths = setup_experiment(exp_id)
        X_w = np.load(os.path.join(paths["processed_data"], "X_word_features.npy"))
        y_w = np.load(os.path.join(paths["processed_data"], "y_word_labels.npy"))

        assembler.train(X_w, y_w)
        assembler.save_model()
    except Exception as e:
        print(f"[WARNING] Word assembler training failed; continuing to evaluation. Error: {e}")

    # Step 4: Evaluate system performance on the held-out test set
    print(f"\n[STEP 4/5] Evaluating system metrics for {exp_id}...")
    evaluate_system(exp_id=exp_id)

    # Step 5: SHAP explainability analysis
    print(f"\n[STEP 5/5] Running SHAP explainability analysis for {exp_id}...")
    run_explainability(exp_id=exp_id)

    end_time = time.time()
    mins = (end_time - start_time) // 60
    print(f"\n[INFO] Experiment execution completed: {exp_id}. Elapsed: {mins:.0f} min.")


if __name__ == "__main__":
    print("[INFO] Starting experiment orchestration...")

    # E0: Baseline (must be executed first)
    execute_experiment(
        exp_id="E0_Baseline",
        processor_params={"band": "broadband", "apply_ica": False, "target_fs": 256},
        crop_time=None,
        phase_filter="all",
        channels_to_use="all",
        n_trials_optuna=5,
        max_epochs=300
    )

    # E1: ICA artifact removal
    execute_experiment(
        exp_id="E1_ICA_Filtering",
        processor_params={"band": "broadband", "apply_ica": True, "target_fs": 256},
        crop_time=None
    )

    # E2: Resampling to 512 Hz
    execute_experiment(
        exp_id="E2_Resampling_512Hz",
        processor_params={"band": "broadband", "apply_ica": False, "target_fs": 512},
        crop_time=None
    )

    # E3: ERP N400 window cropping (200-600 ms)
    execute_experiment(
        exp_id="E3_ERP_N400",
        processor_params={"band": "broadband", "apply_ica": False, "target_fs": 256},
        crop_time=(200, 600)
    )

    # E4: Channel ablation — language cortex (Broca and Wernicke regions)
    execute_experiment(
        exp_id="E4_Channel_Language",
        processor_params={"band": "broadband", "apply_ica": False, "target_fs": 256},
        channels_to_use=["EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7", "EEG.P7"],
        crop_time=None
    )

    # E5: Data augmentation (noise injection and temporal jittering)
    execute_experiment(
        exp_id="E5_Data_Augmentation",
        processor_params={"band": "broadband", "apply_ica": False, "target_fs": 256},
        use_augmentation=True,
        augmentation_params={"add_noise": True, "noise_factor": 0.05, "apply_jitter": True, "jitter_ms": 10},
        crop_time=None
    )

    # E6: Cross-modality — imagined speech only
    execute_experiment(
        exp_id="E6_CrossModality_ImaginedOnly",
        processor_params={"band": "broadband", "apply_ica": False, "target_fs": 256},
        phase_filter="imagined",
        crop_time=None
    )

    # E7: Frequency band isolation — alpha band
    execute_experiment(
        exp_id="E7_Band_Alpha",
        processor_params={"band": "alpha", "apply_ica": False, "target_fs": 256},
        crop_time=None
    )

    print("\n[INFO] All queued experiment runs completed successfully.")

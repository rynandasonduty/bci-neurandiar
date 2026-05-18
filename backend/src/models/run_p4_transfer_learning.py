"""
P4 Transfer Learning Grid — Champion Model Strategy (Option B)
==============================================================
Loads the P1 Global champion model (E0_Baseline by default) and fine-tunes it
per subject, writing outputs to:

    backend/models/weights/P4_TransferLearning/{exp_id}/{subj_id}/
        TL_{exp_id}_{subj_id}.h5
        scaler_TL_{exp_id}_{subj_id}.pkl
        Xtest_TL_{exp_id}_{subj_id}.npy
        ytest_TL_{exp_id}_{subj_id}.npy

The base P1 model is frozen down to its classification head, and only the
Dense + Softmax layers are fine-tuned on each subject's per-experiment data.
A low learning rate (1e-4) preserves the pretrained spatial-temporal filters.

RESTRICTION: This script does NOT retrain any P1, P2, or P3 model.
             It only reads from existing P1 artefacts and writes to P4.

Usage:
    cd backend/src/models
    python run_p4_transfer_learning.py                     # all experiments, all subjects
    python run_p4_transfer_learning.py --exp E0_Baseline   # single experiment
    python run_p4_transfer_learning.py --subj S1           # single subject
    python run_p4_transfer_learning.py --champion E2_Resampling_512Hz  # use alt champion
"""

import os
import sys
import glob
import argparse
import time

import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import setup_experiment, MLFLOW_DB_PATH, MODELS_DIR
from preprocessing.build_dataset import DatasetBuilder
from models.eegnet_model import EEGNetClassifier
from utils.data_utils import three_way_split, fit_and_apply_scaler
from run_subject_dependent import EXPERIMENT_RECIPES, load_data_for_subject_grid

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CHAMPION_EXP      = "E0_Baseline"    # Default P1 champion experiment
FINE_TUNE_EPOCHS  = 50
FINE_TUNE_LR      = 1e-4
FINE_TUNE_BATCH   = 16
P4_PILAR          = "P4_TransferLearning"

# ---------------------------------------------------------------------------
# P4 PATH HELPER
# ---------------------------------------------------------------------------
def setup_p4_experiment(exp_id: str) -> dict:
    """
    Construct the Golden Standard directory for a P4 experiment.

    P4 is not registered in config.setup_experiment() because it is a
    derivative paradigm; its path mirrors the P1/P2/P3 convention but is
    created here to keep config.py non-invasive.

    Returns:
        dict: {'weights': <abs-path>, 'processed_data': <abs-path>}
    """
    exp_dir = os.path.join(MODELS_DIR, "weights", P4_PILAR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    return {"weights": exp_dir, "processed_data": exp_dir}

# ---------------------------------------------------------------------------
# LAYER FREEZE UTILITY
# ---------------------------------------------------------------------------
def freeze_base_layers(model: tf.keras.Model) -> tf.keras.Model:
    """
    Freeze all layers except the final classification head.

    Uses isinstance() to identify trainable head layers so that Keras's
    auto-generated name suffixes (e.g., 'dense_1', 'softmax_2') are never
    compared as bare strings.

    Args:
        model: A compiled tf.keras.Model (EEGNet-8,2 architecture).

    Returns:
        The same model with non-head layers frozen.
    """
    trainable_types = (
        tf.keras.layers.Dense,
        tf.keras.layers.Softmax,
        tf.keras.layers.Activation,
    )
    frozen_count = 0
    for layer in model.layers:
        if isinstance(layer, trainable_types):
            layer.trainable = True
        else:
            layer.trainable = False
            frozen_count += 1

    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    total_params = sum(
        tf.size(w).numpy() for w in model.weights
    )
    print(f"    [INFO] Frozen {frozen_count} layers. "
          f"Trainable params: {trainable_params:,} / {total_params:,} total.")
    return model

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def execute_p4_transfer_learning(
    champion_exp: str = CHAMPION_EXP,
    target_experiments: list = None,
    target_subjects: list = None,
    fine_tune_epochs: int = FINE_TUNE_EPOCHS,
):
    """
    Execute the P4 Champion Model fine-tuning grid.

    Args:
        champion_exp (str): Experiment ID of the P1 champion model to load as base.
        target_experiments (list): Subset of EXPERIMENT_RECIPES to fine-tune on.
                                   If None, all 8 experiments are used.
        target_subjects (list): Subject IDs to fine-tune. If None, all log-file-
                                 detected subjects are used.
        fine_tune_epochs (int): Maximum fine-tuning epochs per subject.
    """
    print("\n" + "=" * 60)
    print(f"  P4 TRANSFER LEARNING — Champion Model Strategy")
    print(f"  Champion Source : P1_Global / {champion_exp}")
    print(f"  Fine-tune epochs: {fine_tune_epochs}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Load the P1 champion model
    # -----------------------------------------------------------------------
    p1_dir = os.path.join(MODELS_DIR, "weights", "P1_Global", champion_exp)
    champion_model_path = os.path.join(p1_dir, f"eegnet_trained_{champion_exp}.h5")

    if not os.path.exists(champion_model_path):
        print(f"[ERROR] Champion model not found: {champion_model_path}")
        print("[ERROR] Ensure P1 training has completed for the selected experiment.")
        sys.exit(1)

    print(f"\n[INFO] Loading P1 champion model from: {champion_model_path}")
    champion_model = tf.keras.models.load_model(champion_model_path)
    print(f"[INFO] Model loaded. Input shape: {champion_model.input_shape}")

    # -----------------------------------------------------------------------
    # 2. Resolve raw data directory and subject list
    # -----------------------------------------------------------------------
    raw_dir = setup_experiment("E0_Baseline", pilar="P1_Global")["raw_data"]
    log_files = glob.glob(os.path.join(raw_dir, "logs", "*_experiment_log.txt"))
    all_subjects = sorted(
        os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files
    )

    subjects_to_run = target_subjects if target_subjects else all_subjects

    if target_experiments:
        valid_experiments   = {k: v for k, v in EXPERIMENT_RECIPES.items() if k in target_experiments}
        invalid_experiments = [e for e in target_experiments if e not in EXPERIMENT_RECIPES]
        if invalid_experiments:
            print(f"[ERROR] The following --exp values are not valid experiment IDs: {invalid_experiments}")
            print(f"[ERROR] Valid experiment IDs: {list(EXPERIMENT_RECIPES.keys())}")
            sys.exit(1)
        experiments_to_run = valid_experiments
    else:
        experiments_to_run = EXPERIMENT_RECIPES

    if not experiments_to_run:
        print("[ERROR] No valid experiments to process. Exiting.")
        sys.exit(1)

    print(f"[INFO] Subjects to fine-tune : {subjects_to_run}")
    print(f"[INFO] Experiments to process: {list(experiments_to_run.keys())}")

    # -----------------------------------------------------------------------
    # 3. MLflow tracking setup
    # -----------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_DB_PATH)
    mlflow.set_experiment("BCI_P4_TransferLearning")

    overall_start = time.time()
    total_success = 0
    total_skip    = 0

    # -----------------------------------------------------------------------
    # 4. Fine-tuning loop: experiment × subject
    # -----------------------------------------------------------------------
    for exp_name, recipe in experiments_to_run.items():
        paths       = setup_p4_experiment(exp_name)
        weights_dir = paths["weights"]

        print(f"\n[{exp_name}] Fine-tuning {len(subjects_to_run)} subjects...")

        for subject_id in subjects_to_run:
            run_start = time.time()
            print(f"  [{subject_id}] Loading subject data...")

            # Load per-subject, per-experiment EEG epochs
            X, y = load_data_for_subject_grid(exp_name, subject_id, raw_dir, recipe)
            if X is None:
                print(f"  [{subject_id}] WARNING: No valid data. Skipping.")
                total_skip += 1
                continue

            # Anti-leakage 3-way split
            X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y)

            # Fit scaler on training data only
            scaler_path = os.path.join(weights_dir, f"scaler_TL_{exp_name}_{subject_id}.pkl")
            X_train, X_val, X_test, _ = fit_and_apply_scaler(
                X_train, X_val, X_test, save_path=scaler_path
            )

            # Persist held-out test set alongside the model
            np.save(os.path.join(weights_dir, f"Xtest_TL_{exp_name}_{subject_id}.npy"), X_test)
            np.save(os.path.join(weights_dir, f"ytest_TL_{exp_name}_{subject_id}.npy"), y_test)

            # Clone champion weights into a fresh model instance to avoid
            # cross-subject gradient contamination
            fine_tune_model = tf.keras.models.clone_model(champion_model)
            fine_tune_model.set_weights(champion_model.get_weights())

            # Freeze all layers below the classification head
            fine_tune_model = freeze_base_layers(fine_tune_model)

            # Compile with a conservative learning rate
            fine_tune_model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
                metrics=["accuracy"],
            )

            # Early stopping to prevent over-fitting on small per-subject sets
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                restore_best_weights=True,
            )

            # Fine-tune
            with mlflow.start_run(run_name=f"P4_{exp_name}_{subject_id}"):
                mlflow.log_param("paradigm", P4_PILAR)
                mlflow.log_param("champion_exp", champion_exp)
                mlflow.log_param("experiment", exp_name)
                mlflow.log_param("subject_id", subject_id)
                mlflow.log_param("fine_tune_epochs", fine_tune_epochs)
                mlflow.log_param("fine_tune_lr", FINE_TUNE_LR)
                mlflow.log_param("n_train", len(y_train))

                print(f"  [{subject_id}] Fine-tuning (max {fine_tune_epochs} epochs)...")
                history = fine_tune_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=fine_tune_epochs,
                    batch_size=FINE_TUNE_BATCH,
                    callbacks=[early_stop],
                    verbose=0,
                )

                best_val_acc = max(history.history["val_accuracy"])
                mlflow.log_metric("best_val_accuracy", best_val_acc)

                # Evaluate on held-out test set
                test_loss, test_acc = fine_tune_model.evaluate(X_test, y_test, verbose=0)
                mlflow.log_metric("test_accuracy", test_acc)
                mlflow.log_metric("test_loss", test_loss)

                # Save model
                model_path = os.path.join(weights_dir, f"TL_{exp_name}_{subject_id}.h5")
                fine_tune_model.save(model_path)
                mlflow.log_artifact(model_path)

            elapsed = time.time() - run_start
            print(f"  [{subject_id}] Val acc: {best_val_acc*100:.2f}% | "
                  f"Test acc: {test_acc*100:.2f}% | "
                  f"Elapsed: {elapsed:.1f}s")
            total_success += 1

    # -----------------------------------------------------------------------
    # 5. Summary
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - overall_start
    mins = total_elapsed // 60
    secs = total_elapsed % 60
    print(f"\n{'=' * 60}")
    print(f"  P4 TRANSFER LEARNING COMPLETE")
    print(f"  Successful runs : {total_success}")
    print(f"  Skipped         : {total_skip}")
    print(f"  Total elapsed   : {mins:.0f} min {secs:.0f}s")
    print(f"  Artefacts saved to : models/weights/{P4_PILAR}/")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P4 Transfer Learning — fine-tune P1 champion model per subject."
    )
    parser.add_argument(
        "--champion",
        type=str,
        default=CHAMPION_EXP,
        help=f"Experiment ID of the P1 champion model (default: {CHAMPION_EXP}).",
    )
    parser.add_argument(
        "--exp",
        type=str,
        nargs="+",
        default=None,
        help="One or more experiment IDs to process (default: all 8).",
    )
    parser.add_argument(
        "--subj",
        type=str,
        nargs="+",
        default=None,
        help="One or more subject IDs to fine-tune (default: all detected).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=FINE_TUNE_EPOCHS,
        help=f"Maximum fine-tuning epochs per subject (default: {FINE_TUNE_EPOCHS}).",
    )
    args = parser.parse_args()

    execute_p4_transfer_learning(
        champion_exp=args.champion,
        target_experiments=args.exp,
        target_subjects=args.subj,
        fine_tune_epochs=args.epochs,
    )

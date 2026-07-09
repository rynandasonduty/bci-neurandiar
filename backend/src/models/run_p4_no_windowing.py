"""
backend/src/models/run_p4_no_windowing.py

P4 (No-Windowing) pilot experiment.

Hypothesis under test: training on the full 5-second imagined-speech epoch,
instead of five 1-second windows (as used by P1/P2/P3), yields better
accuracy because it avoids mixing the readiness-potential preparation phase
with the peak-activity phase under a single window label.

Pilot scope (deliberately narrow, per experiment design):
  - Subject: S3 only
  - Configuration: E0_Baseline only (band-pass 0.5-50 Hz, no augmentation)
  - Feature group: Barlow only

This script is entirely new. It reuses existing public APIs without
modifying them: SignalProcessor (via full_epoch_processor), EEGFeatureExtractor,
ClassicalClassifier, and the shared anti-leakage split/scaling utilities in
utils.data_utils. All P4 artifacts are written under
backend/models/weights/P4_NoWindowing/, fully separate from P1/P2/P3.
"""
import os
import sys
import glob
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment
from preprocessing.full_epoch_processor import FullEpochDatasetBuilder
from features.extract_eeg_features import EEGFeatureExtractor
from models.classical_models import ClassicalClassifier
from utils.data_utils import three_way_split, fit_and_apply_scaler

PILAR = "P4_NoWindowing"
EXP_ID = "E0_Baseline"
SUBJECT_ID = "S3"
FEAT_GROUP = "barlow"

# Identical E0 baseline recipe used by P1/P2/P3 (0.5-50 Hz band-pass, no ICA,
# native 256 Hz, no ERP cropping, no channel ablation, no augmentation).
E0_PROCESSOR_PARAMS = {"band": "broadband", "apply_ica": False, "target_fs": 256}
PHASE_FILTER = "all"


def load_p4_epochs(subject_id, raw_data_dir):
    """Load full 5-second (unwindowed) epochs for a single subject."""
    builder = FullEpochDatasetBuilder(
        processor_params=E0_PROCESSOR_PARAMS,
        phase_filter=PHASE_FILTER,
    )

    log_file = os.path.join(raw_data_dir, "logs", f"{subject_id}_experiment_log.txt")
    csv_files = glob.glob(os.path.join(raw_data_dir, f"{subject_id}*.csv"))
    if not os.path.exists(log_file) or not csv_files:
        return None, None

    X_list, y_list = builder.process_subject(subject_id, csv_files[0], log_file)
    if len(X_list) == 0:
        return None, None

    # (N, time, channels) -> (N, channels, time), matching EEGFeatureExtractor's
    # expected (N, Channels, Time) input convention.
    X_3d = np.transpose(np.array(X_list), (0, 2, 1))
    return X_3d, np.array(y_list)


def run_pilot():
    print(f"\n[INFO][P4] Starting no-windowing pilot: {SUBJECT_ID}/{EXP_ID}/{FEAT_GROUP}")

    paths = setup_experiment(EXP_ID, pilar=PILAR)
    raw_dir = paths["raw_data"]
    weights_dir = paths["weights"]

    X_3d, y = load_p4_epochs(SUBJECT_ID, raw_dir)
    if X_3d is None:
        raise RuntimeError(f"[P4] No epochs extracted for subject {SUBJECT_ID}.")

    print(f"[INFO][P4] Full-epoch samples extracted: {X_3d.shape[0]} "
          f"(shape per sample: {X_3d.shape[1:]})")

    X_train_3d, X_val_3d, X_test_3d, y_train, y_val, y_test = three_way_split(X_3d, y)
    print(f"[INFO][P4] Split sizes -> train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

    extractor = EEGFeatureExtractor(fs=E0_PROCESSOR_PARAMS["target_fs"])
    X_train = extractor.transform(X_train_3d, groups=[FEAT_GROUP])
    X_val = extractor.transform(X_val_3d, groups=[FEAT_GROUP])
    X_test = extractor.transform(X_test_3d, groups=[FEAT_GROUP])

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    scaler_path = os.path.join(
        weights_dir, f"scaler_P4_NoWindowing_SVM_{FEAT_GROUP}_{EXP_ID}_{SUBJECT_ID}.pkl"
    )
    X_train, X_val, X_test, scaler = fit_and_apply_scaler(X_train, X_val, X_test, save_path=scaler_path)

    np.save(os.path.join(weights_dir, f"Xtest_P4_NoWindowing_SVM_{FEAT_GROUP}_{EXP_ID}_{SUBJECT_ID}.npy"), X_test)
    np.save(os.path.join(weights_dir, f"ytest_P4_NoWindowing_SVM_{FEAT_GROUP}_{EXP_ID}_{SUBJECT_ID}.npy"), y_test)

    model = ClassicalClassifier(model_type='svm', C=10)
    model.train(X_train, y_train)

    val_acc = model.evaluate(X_val, y_val)
    test_acc = model.evaluate(X_test, y_test)

    y_test_pred = model.pipeline.predict(X_test)
    # Class coverage: classes for which at least one test sample was correctly
    # predicted (matches the definition used in the P3 champion grand matrix),
    # not merely classes the classifier happened to assign.
    correct_mask = y_test_pred == y_test
    classes_covered = sorted(set(np.asarray(y_test)[correct_mask].tolist()))

    model_path = os.path.join(weights_dir, f"SVM_P4_NoWindowing_{FEAT_GROUP}_{EXP_ID}_{SUBJECT_ID}.pkl")
    model.save_model(model_path)

    results = {
        "paradigm": PILAR,
        "experiment": EXP_ID,
        "subject_id": SUBJECT_ID,
        "feature_group": FEAT_GROUP,
        "n_samples_total": int(X_3d.shape[0]),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "n_classes_covered": len(classes_covered),
        "n_classes_total": 19,
        "classes_covered": classes_covered,
        "model_path": model_path,
        "scaler_path": scaler_path,
    }

    results_path = os.path.join(weights_dir, f"results_P4_NoWindowing_{FEAT_GROUP}_{EXP_ID}_{SUBJECT_ID}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO][P4] Val accuracy:  {val_acc*100:.4f}%")
    print(f"[INFO][P4] Test accuracy: {test_acc*100:.4f}%")
    print(f"[INFO][P4] Class coverage (test predictions): {len(classes_covered)}/19")
    print(f"[INFO][P4] Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    run_pilot()

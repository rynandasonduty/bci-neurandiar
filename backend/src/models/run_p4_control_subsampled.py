"""
backend/src/models/run_p4_control_subsampled.py

P4 subsampling control experiment.

Purpose: isolate whether the P4 No-Windowing pilot's accuracy drop (0.00%
test accuracy at n_train=106, n_val=23, n_test=23) was caused by the small
sample size, or by the full-epoch (no-windowing) structure itself.

Method: rebuild the standard 1-second-windowed S3/E0_Baseline/Barlow
dataset (same pipeline structure as the P3 champion), split it once with
three_way_split(random_state=42) -- identical to the P3 pipeline -- extract
Barlow features on each full split, then draw five independent random
subsamples (train=106, val=23, test=23) using five different subsampling
seeds (42-46, distinct from the split seed) to average out the luck of any
single small subset. One SVM is trained per seed with the same
configuration as the P3 champion (ClassicalClassifier(model_type='svm', C=10),
StandardScaler fit only on that seed's train subset).

Only existing public APIs are reused, none are modified:
SignalProcessor.apply_filter()/windowing_slot() (via
preprocessing.windowed_reference_processor), EEGFeatureExtractor,
ClassicalClassifier, three_way_split, fit_and_apply_scaler.

All artifacts are written under backend/models/weights/P4_Control_Subsampled/,
fully separate from P1/P2/P3 and from the P4_NoWindowing pilot.
"""
import os
import sys
import glob
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment
from preprocessing.windowed_reference_processor import WindowedReferenceDatasetBuilder
from features.extract_eeg_features import EEGFeatureExtractor
from models.classical_models import ClassicalClassifier
from utils.data_utils import three_way_split, fit_and_apply_scaler

PILAR = "P4_Control_Subsampled"
EXP_ID = "E0_Baseline"
SUBJECT_ID = "S3"
FEAT_GROUP = "barlow"

# Identical E0 baseline recipe used by P1/P2/P3 (0.5-50 Hz band-pass, no ICA,
# native 256 Hz, no ERP cropping, no channel ablation, no augmentation).
E0_PROCESSOR_PARAMS = {"band": "broadband", "apply_ica": False, "target_fs": 256}
PHASE_FILTER = "all"

# Fixed split seed, identical to the P3 pipeline (utils.data_utils.three_way_split default).
SPLIT_RANDOM_STATE = 42

# Target subset sizes, matching the P4 No-Windowing pilot exactly.
N_TRAIN_TARGET = 106
N_VAL_TARGET = 23
N_TEST_TARGET = 23

# Independent subsampling seeds (distinct from SPLIT_RANDOM_STATE). Five
# repeats are used to avoid drawing conclusions from a single subset that
# happens to be easy or hard, given the small test size (23 samples).
SUBSAMPLE_SEEDS = [42, 43, 44, 45, 46]

NUM_CLASSES = 19


def load_windowed_reference_dataset(subject_id, raw_data_dir):
    """Build the full standard-windowed S3/E0_Baseline dataset (no subsampling)."""
    builder = WindowedReferenceDatasetBuilder(
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


def subsample_indices(rng, n_available, n_target, label):
    """Draw n_target indices without replacement from a pool of n_available."""
    if n_target > n_available:
        raise ValueError(
            f"Requested {label} subsample size {n_target} exceeds available {n_available} samples."
        )
    return rng.choice(n_available, size=n_target, replace=False)


def class_distribution(y, num_classes=NUM_CLASSES):
    """Return a per-class sample count list of length num_classes (0 for absent classes)."""
    counts = np.zeros(num_classes, dtype=int)
    unique, freq = np.unique(y, return_counts=True)
    for cls, count in zip(unique, freq):
        counts[int(cls)] = int(count)
    return counts.tolist()


def run_control_experiment():
    print(f"\n[INFO][P4-Control] Rebuilding standard windowed reference dataset: "
          f"{SUBJECT_ID}/{EXP_ID}/{FEAT_GROUP}")

    paths = setup_experiment(EXP_ID, pilar=PILAR)
    raw_dir = paths["raw_data"]
    weights_dir = paths["weights"]

    X_3d, y = load_windowed_reference_dataset(SUBJECT_ID, raw_dir)
    if X_3d is None:
        raise RuntimeError(f"[P4-Control] No windowed samples extracted for subject {SUBJECT_ID}.")

    print(f"[INFO][P4-Control] Full windowed reference dataset: {X_3d.shape[0]} samples "
          f"(shape per sample: {X_3d.shape[1:]})")

    # Split once, exactly as the P3 pipeline does (random_state=42), on the
    # raw 3D windows before feature extraction to preserve identical split
    # semantics. No augmentation is applied (E0_Baseline recipe).
    X_train_3d, X_val_3d, X_test_3d, y_train_full, y_val_full, y_test_full = three_way_split(
        X_3d, y, random_state=SPLIT_RANDOM_STATE
    )
    print(f"[INFO][P4-Control] Full split sizes -> train: {len(y_train_full)}, "
          f"val: {len(y_val_full)}, test: {len(y_test_full)}")

    # Extract Barlow features once per full split. Feature extraction is
    # per-sample and stateless, so extracting before subsampling and
    # extracting after subsampling yield identical values; doing it once
    # here avoids redundant computation across the five seeds.
    extractor = EEGFeatureExtractor(fs=E0_PROCESSOR_PARAMS["target_fs"])
    X_train_feat_full = extractor.transform(X_train_3d, groups=[FEAT_GROUP])
    X_val_feat_full = extractor.transform(X_val_3d, groups=[FEAT_GROUP])
    X_test_feat_full = extractor.transform(X_test_3d, groups=[FEAT_GROUP])

    X_train_feat_full = np.nan_to_num(X_train_feat_full, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_feat_full = np.nan_to_num(X_val_feat_full, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_feat_full = np.nan_to_num(X_test_feat_full, nan=0.0, posinf=0.0, neginf=0.0)

    all_results = []

    for seed in SUBSAMPLE_SEEDS:
        print(f"\n[INFO][P4-Control] Running subsample seed {seed}...")

        # A single RandomState per seed draws train, then val, then test
        # indices in sequence -- deterministic and reproducible given the seed.
        rng = np.random.RandomState(seed)
        train_idx = subsample_indices(rng, len(y_train_full), N_TRAIN_TARGET, "train")
        val_idx = subsample_indices(rng, len(y_val_full), N_VAL_TARGET, "val")
        test_idx = subsample_indices(rng, len(y_test_full), N_TEST_TARGET, "test")

        X_train_sub = X_train_feat_full[train_idx]
        y_train_sub = y_train_full[train_idx]
        X_val_sub = X_val_feat_full[val_idx]
        y_val_sub = y_val_full[val_idx]
        X_test_sub = X_test_feat_full[test_idx]
        y_test_sub = y_test_full[test_idx]

        # StandardScaler fit only on this seed's train subset, matching the
        # P3 champion methodology.
        scaler_path = os.path.join(
            weights_dir, f"scaler_P4_Control_SVM_{FEAT_GROUP}_{EXP_ID}_{SUBJECT_ID}_seed{seed}.pkl"
        )
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = fit_and_apply_scaler(
            X_train_sub, X_val_sub, X_test_sub, save_path=scaler_path
        )

        np.save(
            os.path.join(weights_dir, f"Xtest_P4_Control_SVM_{FEAT_GROUP}_{EXP_ID}_{SUBJECT_ID}_seed{seed}.npy"),
            X_test_scaled,
        )
        np.save(
            os.path.join(weights_dir, f"ytest_P4_Control_SVM_{FEAT_GROUP}_{EXP_ID}_{SUBJECT_ID}_seed{seed}.npy"),
            y_test_sub,
        )

        model = ClassicalClassifier(model_type='svm', C=10)
        model.train(X_train_scaled, y_train_sub)

        val_acc = model.evaluate(X_val_scaled, y_val_sub)
        test_acc = model.evaluate(X_test_scaled, y_test_sub)

        y_test_pred = model.pipeline.predict(X_test_scaled)
        correct_mask = y_test_pred == y_test_sub
        classes_covered = sorted(set(np.asarray(y_test_sub)[correct_mask].tolist()))

        train_class_dist = class_distribution(y_train_sub)
        n_classes_in_train = int(np.sum(np.array(train_class_dist) > 0))

        model_path = os.path.join(
            weights_dir, f"SVM_P4_Control_{FEAT_GROUP}_{EXP_ID}_{SUBJECT_ID}_seed{seed}.pkl"
        )
        model.save_model(model_path)

        seed_result = {
            "seed": seed,
            "n_train": int(len(y_train_sub)),
            "n_val": int(len(y_val_sub)),
            "n_test": int(len(y_test_sub)),
            "val_accuracy": float(val_acc),
            "test_accuracy": float(test_acc),
            "n_classes_covered": len(classes_covered),
            "n_classes_total": NUM_CLASSES,
            "classes_covered": classes_covered,
            "n_classes_present_in_train": n_classes_in_train,
            "train_class_distribution": train_class_dist,
            "model_path": model_path,
            "scaler_path": scaler_path,
        }
        all_results.append(seed_result)

        print(f"[INFO][P4-Control] Seed {seed}: test accuracy {test_acc*100:.4f}%, "
              f"val accuracy {val_acc*100:.4f}%, class coverage {len(classes_covered)}/19, "
              f"classes present in train subset: {n_classes_in_train}/19")

    test_accs = np.array([r["test_accuracy"] for r in all_results])
    aggregate = {
        "seeds": SUBSAMPLE_SEEDS,
        "test_accuracies": test_accs.tolist(),
        "mean_test_accuracy": float(np.mean(test_accs)),
        "std_test_accuracy_sample": float(np.std(test_accs, ddof=1)),
        "std_test_accuracy_population": float(np.std(test_accs, ddof=0)),
    }

    summary = {
        "paradigm": PILAR,
        "experiment": EXP_ID,
        "subject_id": SUBJECT_ID,
        "feature_group": FEAT_GROUP,
        "split_random_state": SPLIT_RANDOM_STATE,
        "full_dataset_n_total": int(X_3d.shape[0]),
        "full_split_n_train": int(len(y_train_full)),
        "full_split_n_val": int(len(y_val_full)),
        "full_split_n_test": int(len(y_test_full)),
        "target_n_train": N_TRAIN_TARGET,
        "target_n_val": N_VAL_TARGET,
        "target_n_test": N_TEST_TARGET,
        "per_seed_results": all_results,
        "aggregate": aggregate,
    }

    summary_path = os.path.join(
        weights_dir, f"results_P4_Control_Subsampled_{FEAT_GROUP}_{EXP_ID}_{SUBJECT_ID}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[INFO][P4-Control] Mean test accuracy over {len(SUBSAMPLE_SEEDS)} seeds: "
          f"{aggregate['mean_test_accuracy']*100:.4f}% "
          f"(sample std: {aggregate['std_test_accuracy_sample']*100:.4f} pp)")
    print(f"[INFO][P4-Control] Summary saved to: {summary_path}")

    return summary


if __name__ == "__main__":
    run_control_experiment()

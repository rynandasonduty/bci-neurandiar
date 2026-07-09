"""
backend/src/experiments_p4_p7/run_p6_transfer_overt_imagined.py

P6 -- Transfer Overt->Imagined. Variable tested: training-data composition
(imagined-only vs. imagined+overt combined). Locked: standard windowing,
standard 0.5-50Hz filter, SVM, Barlow features (no spot-check -- the signal
itself is unchanged by this experiment). Test set: always pure imagined,
identical between the baseline and enriched conditions.

The baseline condition is NOT retrained. It reuses the already-validated
P3_SVM/E6_CrossModality_ImaginedOnly/Barlow artifacts (model, scaler, sealed
Xtest/ytest) for all 12 subjects as both the fixed official test set and the
baseline accuracy figure. Only the ENRICHED condition (imagined+overt
training data) is newly trained here, using the real, unmodified
DatasetBuilder directly (phase_filter='imagined' / 'overt' -- no subclass
needed, since DatasetBuilder already supports phase filtering natively).

Per subject:
  1. Load the existing E6/barlow model + scaler + Xtest/ytest (read-only).
  2. Rebuild the imagined-only train/val/test split via DatasetBuilder +
     three_way_split(random_state=42), and sanity-check the rebuilt test
     split against the loaded Xtest/ytest by re-extracting Barlow features
     and re-applying the OLD (loaded) scaler -- proving the reconstruction
     is faithful before it's used for anything.
  3. Build the extra overt-phase data (no further split -- all of it is
     additional training data).
  4. Concatenate imagined-train + overt into the enriched training set.
  5. Extract Barlow features, fit a NEW scaler on the enriched train split
     only, and use that new scaler (not the old E6 one) to transform the
     rebuilt val/test splits.
  6. Train a new SVM on the enriched data, evaluate on the (freshly
     rescaled) imagined-only test split.

Usage:
    cd backend/src/experiments_p4_p7
    python run_p6_transfer_overt_imagined.py                 # all 12 subjects
    python run_p6_transfer_overt_imagined.py --subjects S1 S2 S3
"""
import os
import sys
import glob
import json
import pickle
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment, RAW_DATA_DIR
from preprocessing.build_dataset import DatasetBuilder
from features.extract_eeg_features import EEGFeatureExtractor
from models.classical_models import ClassicalClassifier
from utils.data_utils import three_way_split, fit_and_apply_scaler

PILAR = "P6_TransferOvertImagined"
FULLSCALE_STAGE_DIR = "Fullscale_12Subj_E0"
FEAT_GROUP = "barlow"  # locked -- no spot-check, signal itself is unchanged

BASELINE_PARADIGM = "P3_SVM"
BASELINE_EXP = "E6_CrossModality_ImaginedOnly"

# Identical to E6_CrossModality_ImaginedOnly's processor_params in
# EXPERIMENT_RECIPES (models/run_subject_dependent.py, read-only reference).
E0_PROCESSOR_PARAMS = {"band": "broadband", "apply_ica": False, "target_fs": 256}
SPLIT_RANDOM_STATE = 42
NUM_CLASSES = 19

REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'P4_P7_Experiments'))
REPORT_PATH = os.path.join(REPORTS_DIR, "P6_TransferOvertImagined_report.md")


def discover_subject_ids():
    log_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "logs", "*_experiment_log.txt")))
    return [os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files]


def resolve_baseline_paths(subject_id):
    """Path to the existing, already-validated P3/E6/barlow artifacts.
    Read-only -- this experiment never writes into P3_SVM/."""
    weights_dir = setup_experiment(BASELINE_EXP, pilar=BASELINE_PARADIGM)["weights"]
    return {
        "model": os.path.join(weights_dir, f"SVM_{FEAT_GROUP}_{BASELINE_EXP}_{subject_id}.pkl"),
        "scaler": os.path.join(weights_dir, f"scaler_SVM_{FEAT_GROUP}_{BASELINE_EXP}_{subject_id}.pkl"),
        "Xtest": os.path.join(weights_dir, f"Xtest_SVM_{FEAT_GROUP}_{BASELINE_EXP}_{subject_id}.npy"),
        "ytest": os.path.join(weights_dir, f"ytest_SVM_{FEAT_GROUP}_{BASELINE_EXP}_{subject_id}.npy"),
    }


def verify_baseline_artifacts_exist(subject_ids):
    """Verify presence for all subjects up front. Missing artifacts are
    reported, never silently patched over by retraining a baseline --
    retraining would produce a different test set than the one already
    validated in Bab 6."""
    missing = {}
    for subject_id in subject_ids:
        paths = resolve_baseline_paths(subject_id)
        missing_files = [k for k, p in paths.items() if not os.path.exists(p)]
        if missing_files:
            missing[subject_id] = missing_files
    return missing


def rebuild_phase_split(subject_id, phase_filter, exp_id_tag):
    """Build one subject's standard-windowed dataset for a given phase via
    the real, unmodified DatasetBuilder. Returns (X_3d, y) or (None, None)."""
    builder = DatasetBuilder(
        exp_id=f"{exp_id_tag}_{subject_id}", phase_filter=phase_filter,
        processor_params=E0_PROCESSOR_PARAMS,
    )
    log_file = os.path.join(RAW_DATA_DIR, "logs", f"{subject_id}_experiment_log.txt")
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, f"{subject_id}*.csv"))
    if not os.path.exists(log_file) or not csv_files:
        return None, None

    X_list, y_list = builder.process_subject(subject_id, csv_files[0], log_file)
    if len(X_list) == 0:
        return None, None

    X_3d = np.transpose(np.array(X_list), (0, 2, 1))
    return X_3d, np.array(y_list)


def sanity_check_reconstruction(X_test_3d_rebuilt, y_test_rebuilt, baseline_scaler,
                                 X_test_loaded, y_test_loaded):
    """Re-extract Barlow features from the rebuilt raw test epochs and
    re-apply the OLD (already-fitted) E6 scaler; compare against the loaded,
    already-scaled Xtest/ytest npy files. This is the proof that
    rebuild_phase_split(phase_filter='imagined') reproduces the exact same
    test set P3 already validated -- not just a shape check."""
    extractor = EEGFeatureExtractor(fs=E0_PROCESSOR_PARAMS["target_fs"])
    X_test_feat = extractor.transform(X_test_3d_rebuilt, groups=[FEAT_GROUP])
    X_test_feat = np.nan_to_num(X_test_feat, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_rescaled = baseline_scaler.transform(X_test_feat)

    shape_match = X_test_rescaled.shape == X_test_loaded.shape
    y_match = shape_match and np.array_equal(y_test_rebuilt, y_test_loaded)
    value_match = False
    max_abs_diff = None
    if shape_match:
        max_abs_diff = float(np.max(np.abs(X_test_rescaled - X_test_loaded)))
        value_match = bool(np.allclose(X_test_rescaled, X_test_loaded, atol=1e-6))

    return {
        "shape_match": bool(shape_match),
        "y_match": bool(y_match),
        "value_match": value_match,
        "max_abs_diff": max_abs_diff,
        "rebuilt_shape": list(X_test_rescaled.shape),
        "loaded_shape": list(X_test_loaded.shape),
    }


def run_one_subject(subject_id, weights_dir):
    paths = resolve_baseline_paths(subject_id)

    X_test_loaded = np.load(paths["Xtest"])
    y_test_loaded = np.load(paths["ytest"])

    baseline_model = ClassicalClassifier(model_type='svm', C=10)
    baseline_model.load_model(paths["model"])
    baseline_test_acc = baseline_model.evaluate(X_test_loaded, y_test_loaded)

    with open(paths["scaler"], "rb") as f:
        baseline_scaler = pickle.load(f)

    X_3d_imagined, y_imagined = rebuild_phase_split(subject_id, "imagined", "P6_Imagined")
    if X_3d_imagined is None:
        return {"subject_id": subject_id, "status": "no_imagined_data"}

    X_train_3d, X_val_3d, X_test_3d, y_train, y_val, y_test = three_way_split(
        X_3d_imagined, y_imagined, random_state=SPLIT_RANDOM_STATE
    )

    sanity = sanity_check_reconstruction(X_test_3d, y_test, baseline_scaler, X_test_loaded, y_test_loaded)
    if not sanity["shape_match"]:
        print(f"[WARNING][P6] {subject_id}: rebuilt test set shape {sanity['rebuilt_shape']} != "
              f"loaded baseline shape {sanity['loaded_shape']}. Proceeding, but treat this subject's "
              f"result with caution -- see sanity_check in its results JSON.")

    X_overt_3d, y_overt = rebuild_phase_split(subject_id, "overt", "P6_Overt")
    if X_overt_3d is None:
        X_overt_3d = np.empty((0,) + X_train_3d.shape[1:], dtype=X_train_3d.dtype)
        y_overt = np.empty((0,), dtype=y_train.dtype)

    X_train_enriched_3d = np.concatenate([X_train_3d, X_overt_3d], axis=0)
    y_train_enriched = np.concatenate([y_train, y_overt], axis=0)

    extractor = EEGFeatureExtractor(fs=E0_PROCESSOR_PARAMS["target_fs"])
    X_train_feat = extractor.transform(X_train_enriched_3d, groups=[FEAT_GROUP])
    X_val_feat = extractor.transform(X_val_3d, groups=[FEAT_GROUP])
    X_test_feat = extractor.transform(X_test_3d, groups=[FEAT_GROUP])

    X_train_feat = np.nan_to_num(X_train_feat, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_feat = np.nan_to_num(X_val_feat, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_feat = np.nan_to_num(X_test_feat, nan=0.0, posinf=0.0, neginf=0.0)

    scaler_path = os.path.join(weights_dir, f"scaler_P6_Fullscale_{FEAT_GROUP}_{subject_id}.pkl")
    X_train_feat, X_val_feat, X_test_feat, new_scaler = fit_and_apply_scaler(
        X_train_feat, X_val_feat, X_test_feat, save_path=scaler_path
    )

    np.save(os.path.join(weights_dir, f"Xtest_P6_Fullscale_{FEAT_GROUP}_{subject_id}.npy"), X_test_feat)
    np.save(os.path.join(weights_dir, f"ytest_P6_Fullscale_{FEAT_GROUP}_{subject_id}.npy"), y_test)

    model = ClassicalClassifier(model_type='svm', C=10)
    model.train(X_train_feat, y_train_enriched)

    enriched_val_acc = model.evaluate(X_val_feat, y_val)
    enriched_test_acc = model.evaluate(X_test_feat, y_test)

    y_test_pred = model.pipeline.predict(X_test_feat)
    correct_mask = y_test_pred == y_test
    classes_covered = sorted(set(np.asarray(y_test)[correct_mask].tolist()))

    model_path = os.path.join(weights_dir, f"SVM_P6_Fullscale_{FEAT_GROUP}_{subject_id}.pkl")
    model.save_model(model_path)

    return {
        "paradigm": PILAR, "subject_id": subject_id, "feature_group": FEAT_GROUP,
        "baseline_paradigm": BASELINE_PARADIGM, "baseline_experiment": BASELINE_EXP,
        "baseline_test_accuracy": float(baseline_test_acc),
        "enriched_val_accuracy": float(enriched_val_acc),
        "enriched_test_accuracy": float(enriched_test_acc),
        "delta_pp": float((enriched_test_acc - baseline_test_acc) * 100.0),
        "n_train_imagined_only": int(len(y_train)),
        "n_overt_added": int(len(y_overt)),
        "n_train_enriched_total": int(len(y_train_enriched)),
        "n_val": int(len(y_val)), "n_test": int(len(y_test)),
        "n_classes_covered": len(classes_covered), "n_classes_total": NUM_CLASSES,
        "classes_covered": classes_covered,
        "sanity_check": sanity,
        "model_path": model_path, "scaler_path": scaler_path,
    }


def run_fullscale(subject_ids=None):
    print(f"\n{'=' * 70}\n P6 Transfer Overt->Imagined -- Full-scale ({BASELINE_EXP} baseline, {FEAT_GROUP})\n{'=' * 70}")

    weights_dir = setup_experiment(FULLSCALE_STAGE_DIR, pilar=PILAR)["weights"]
    if subject_ids is None:
        subject_ids = discover_subject_ids()

    print(f"[INFO][P6] Verifying baseline {BASELINE_EXP}/{FEAT_GROUP} artifacts for {len(subject_ids)} subject(s)...")
    missing = verify_baseline_artifacts_exist(subject_ids)
    if missing:
        print(f"[WARNING][P6] Missing baseline artifacts, these subjects will be SKIPPED (not retrained): {missing}")
    else:
        print(f"[INFO][P6] All baseline artifacts present for all {len(subject_ids)} subject(s).")

    results = {}
    for subject_id in subject_ids:
        if subject_id in missing:
            continue

        model_path = os.path.join(weights_dir, f"SVM_P6_Fullscale_{FEAT_GROUP}_{subject_id}.pkl")
        if os.path.exists(model_path):
            print(f"[SKIP][P6] Model for {subject_id} already exists.")
            existing_json = os.path.join(weights_dir, f"results_{subject_id}.json")
            if os.path.exists(existing_json):
                with open(existing_json) as f:
                    results[subject_id] = json.load(f)
            continue

        print(f"\n[INFO][P6] Processing subject {subject_id}...")
        result = run_one_subject(subject_id, weights_dir)
        if result.get("status") == "no_imagined_data":
            print(f"[WARNING][P6] No imagined-phase data rebuilt for {subject_id}; skipping.")
            continue

        results[subject_id] = result
        with open(os.path.join(weights_dir, f"results_{subject_id}.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(f"[INFO][P6] {subject_id}: baseline {result['baseline_test_accuracy']*100:.2f}% -> "
              f"enriched {result['enriched_test_accuracy']*100:.2f}% (delta {result['delta_pp']:+.2f}pp) | "
              f"+{result['n_overt_added']} overt samples | "
              f"sanity: shape={result['sanity_check']['shape_match']} value={result['sanity_check']['value_match']}")

    write_report(results, subject_ids, missing)
    return results


def write_report(results, all_subject_ids, missing_baseline):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    n_complete = len(results)
    n_total = len(all_subject_ids)

    lines = []
    lines.append("# P6 -- Transfer Overt->Imagined: Experiment Report")
    lines.append("")
    lines.append("Variable tested: training-data composition (imagined-only vs. imagined+overt "
                  "combined). Locked: standard windowing/filter, SVM, Barlow features (no spot-check). "
                  f"Baseline reused (not retrained) from `{BASELINE_PARADIGM}/{BASELINE_EXP}`. Test set: "
                  "always pure imagined, identical between baseline and enriched conditions.")
    lines.append("")
    if n_complete < n_total - len(missing_baseline):
        lines.append(f"> **Status: PARTIAL ({n_complete}/{n_total} subjects).** This report reflects a "
                      f"smoke-test or in-progress run. Re-run `run_p6_transfer_overt_imagined.py` on the "
                      f"lab machine to complete the grid; already-completed subjects are skipped "
                      f"automatically (auto-resume).")
        lines.append("")
    if missing_baseline:
        lines.append(f"**Subjects skipped -- missing baseline artifacts (not retrained):** {missing_baseline}")
        lines.append("")

    lines.append("## Per-subject Results")
    lines.append("")
    lines.append("| Subject | Baseline Test Acc (%) | Enriched Test Acc (%) | Delta (pp) | +Overt Samples | Sanity Check |")
    lines.append("|---|---|---|---|---|---|")
    for subject_id in all_subject_ids:
        if subject_id in missing_baseline:
            lines.append(f"| {subject_id} | -- | -- | -- | -- | missing baseline artifacts |")
            continue
        r = results.get(subject_id)
        if r is None:
            lines.append(f"| {subject_id} | -- | -- | -- | -- | not yet run |")
        else:
            sc = r["sanity_check"]
            sanity_str = "OK" if (sc["shape_match"] and sc["value_match"]) else "REVIEW (see JSON)"
            lines.append(f"| {subject_id} | {r['baseline_test_accuracy']*100:.4f} | "
                          f"{r['enriched_test_accuracy']*100:.4f} | {r['delta_pp']:+.4f} | "
                          f"{r['n_overt_added']} | {sanity_str} |")

    if results:
        deltas = [r["delta_pp"] for r in results.values()]
        lines.append("")
        lines.append(f"Mean delta so far: {np.mean(deltas):+.4f} pp (std {np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0:.4f} pp, n={len(deltas)})")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO][P6] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P6 Transfer Overt->Imagined: enriched-vs-baseline SVM comparison.")
    parser.add_argument("--subjects", nargs="+", default=None,
                         help="Restrict to specific subject IDs (e.g. --subjects S1 S2). "
                              "Default: all 12 subjects, auto-discovered.")
    args = parser.parse_args()
    run_fullscale(subject_ids=args.subjects)

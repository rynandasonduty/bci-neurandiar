"""
backend/src/experiments_p4_p7/run_p5_shifted_bandpass.py

P5 -- Shifted Bandpass Filter. Variable tested: bandpass range (15-65Hz vs.
the standard 0.5-50Hz broadband). Locked: standard 5x1s windowing, SVM, E0
Baseline (no augmentation), phase_filter='all'.

Structurally identical to run_p4_nowindowing.py (Stage A spot-check on S3
across 5 feature groups, Stage B full-scale on 12 subjects with the
auto-selected feature group and auto-resume) -- only the dataset builder
differs (ShiftedBandDatasetBuilder instead of NoWindowDatasetBuilder). No
prior-session artifacts exist for this paradigm; this is written fresh.

Note: ShiftedBandSignalProcessor's __init__ (see signal_processors_ext.py)
takes no `band` parameter -- the 15-65Hz range is fixed internally, since
that is exactly the single variable this experiment changes. P5_PROCESSOR_
PARAMS below intentionally omits the 'band' key that the standard E0 recipe
carries elsewhere in this codebase.

Usage:
    cd backend/src/experiments_p4_p7
    python run_p5_shifted_bandpass.py                # Stage A then Stage B
    python run_p5_shifted_bandpass.py --stage a       # spot-check only
    python run_p5_shifted_bandpass.py --stage b       # full-scale only
"""
import os
import sys
import glob
import json
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment
from features.extract_eeg_features import EEGFeatureExtractor, FEATURE_GROUPS
from models.classical_models import ClassicalClassifier
from utils.data_utils import three_way_split, fit_and_apply_scaler
from experiments_p4_p7.dataset_builders_ext import ShiftedBandDatasetBuilder, select_winning_feature_group

PILAR = "P5_ShiftedBandpass"
EXP_ID = "E0_Baseline"
SPOTCHECK_SUBJECT = "S3"
SPOTCHECK_STAGE_DIR = "Spotcheck_S3_E0"
FULLSCALE_STAGE_DIR = "Fullscale_12Subj_E0"

# P5's processor_params: no 'band' key (ShiftedBandSignalProcessor doesn't
# accept one -- the 15-65Hz range is hardcoded). apply_ica/target_fs match
# the standard E0 recipe used by P1-P3, since only the filter band varies.
P5_PROCESSOR_PARAMS = {"apply_ica": False, "target_fs": 256}
PHASE_FILTER = "all"

NUM_CLASSES = 19
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'P4_P7_Experiments'))
REPORT_PATH = os.path.join(REPORTS_DIR, "P5_ShiftedBandpass_report.md")


def discover_subject_ids(raw_data_dir):
    log_files = sorted(glob.glob(os.path.join(raw_data_dir, "logs", "*_experiment_log.txt")))
    return [os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files]


def load_shifted_band_epochs(subject_id, raw_data_dir):
    """Build one subject's standard-windowed, 15-65Hz-filtered dataset via
    ShiftedBandDatasetBuilder. Returns (X_3d, y) with X_3d shape
    (N, channels, time), or (None, None) if raw data is unavailable."""
    builder = ShiftedBandDatasetBuilder(
        exp_id="RawBuild", processor_params=P5_PROCESSOR_PARAMS,
        phase_filter=PHASE_FILTER, pilar=PILAR,
    )
    log_file = os.path.join(raw_data_dir, "logs", f"{subject_id}_experiment_log.txt")
    csv_files = glob.glob(os.path.join(raw_data_dir, f"{subject_id}*.csv"))
    if not os.path.exists(log_file) or not csv_files:
        return None, None

    X_list, y_list = builder.process_subject(subject_id, csv_files[0], log_file)
    if len(X_list) == 0:
        return None, None

    X_3d = np.transpose(np.array(X_list), (0, 2, 1))
    return X_3d, np.array(y_list)


def build_model_path(weights_dir, tag, feat_group, subject_id):
    return os.path.join(weights_dir, f"SVM_{tag}_{feat_group}_{subject_id}.pkl")


def train_and_evaluate_one(X_3d, y, feat_group, weights_dir, subject_id, tag):
    X_train_3d, X_val_3d, X_test_3d, y_train, y_val, y_test = three_way_split(X_3d, y)

    extractor = EEGFeatureExtractor(fs=P5_PROCESSOR_PARAMS["target_fs"])
    groups = None if feat_group == "all" else [feat_group]
    X_train = extractor.transform(X_train_3d, groups=groups)
    X_val = extractor.transform(X_val_3d, groups=groups)
    X_test = extractor.transform(X_test_3d, groups=groups)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    scaler_path = os.path.join(weights_dir, f"scaler_{tag}_{feat_group}_{subject_id}.pkl")
    X_train, X_val, X_test, scaler = fit_and_apply_scaler(X_train, X_val, X_test, save_path=scaler_path)

    np.save(os.path.join(weights_dir, f"Xtest_{tag}_{feat_group}_{subject_id}.npy"), X_test)
    np.save(os.path.join(weights_dir, f"ytest_{tag}_{feat_group}_{subject_id}.npy"), y_test)

    model = ClassicalClassifier(model_type='svm', C=10)
    model.train(X_train, y_train)

    val_acc = model.evaluate(X_val, y_val)
    test_acc = model.evaluate(X_test, y_test)

    y_test_pred = model.pipeline.predict(X_test)
    correct_mask = y_test_pred == y_test
    classes_covered = sorted(set(np.asarray(y_test)[correct_mask].tolist()))

    model_path = build_model_path(weights_dir, tag, feat_group, subject_id)
    model.save_model(model_path)

    result = {
        "paradigm": PILAR, "experiment": EXP_ID, "subject_id": subject_id,
        "feature_group": feat_group,
        "n_samples_total": int(X_3d.shape[0]), "n_train": int(len(y_train)),
        "n_val": int(len(y_val)), "n_test": int(len(y_test)),
        "val_accuracy": float(val_acc), "test_accuracy": float(test_acc),
        "n_classes_covered": len(classes_covered), "n_classes_total": NUM_CLASSES,
        "classes_covered": classes_covered,
        "model_path": model_path, "scaler_path": scaler_path,
    }
    return result


def run_stage_a_spotcheck():
    print(f"\n{'=' * 70}\n P5 Shifted Bandpass -- Stage A: Spot-check ({SPOTCHECK_SUBJECT}, {EXP_ID})\n{'=' * 70}")

    raw_dir = setup_experiment(SPOTCHECK_STAGE_DIR, pilar=PILAR)["raw_data"]
    stage_weights_dir = setup_experiment(SPOTCHECK_STAGE_DIR, pilar=PILAR)["weights"]

    X_3d, y = load_shifted_band_epochs(SPOTCHECK_SUBJECT, raw_dir)
    if X_3d is None:
        raise RuntimeError(f"[P5 Stage A] No epochs extracted for subject {SPOTCHECK_SUBJECT}.")
    print(f"[INFO][P5-A] Windowed samples extracted: {X_3d.shape[0]} (shape per sample: {X_3d.shape[1:]})")

    spotcheck_results = {}
    for feat_group in FEATURE_GROUPS:
        group_dir = os.path.join(stage_weights_dir, feat_group)
        os.makedirs(group_dir, exist_ok=True)

        result = train_and_evaluate_one(X_3d, y, feat_group, group_dir, SPOTCHECK_SUBJECT, tag="P5_Spotcheck")
        spotcheck_results[feat_group] = result

        with open(os.path.join(group_dir, f"results_{feat_group}.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(f"[INFO][P5-A] {feat_group:<10} test acc {result['test_accuracy']*100:6.2f}%  "
              f"coverage {result['n_classes_covered']}/19")

    with open(os.path.join(stage_weights_dir, "spotcheck_summary.json"), "w") as f:
        json.dump(spotcheck_results, f, indent=2)

    print(f"\n{'Feature Group':<15}{'Test Acc %':>12}{'Class Coverage':>18}")
    print("-" * 45)
    for g in FEATURE_GROUPS:
        r = spotcheck_results[g]
        print(f"{g:<15}{r['test_accuracy']*100:>12.4f}{r['n_classes_covered']:>15}/19")
    print("-" * 45)

    return spotcheck_results


def run_stage_b_fullscale(spotcheck_results=None, subject_ids=None):
    print(f"\n{'=' * 70}\n P5 Shifted Bandpass -- Stage B: Full-scale ({EXP_ID})\n{'=' * 70}")

    raw_dir = setup_experiment(FULLSCALE_STAGE_DIR, pilar=PILAR)["raw_data"]
    stage_weights_dir = setup_experiment(FULLSCALE_STAGE_DIR, pilar=PILAR)["weights"]

    if spotcheck_results is None:
        summary_path = os.path.join(setup_experiment(SPOTCHECK_STAGE_DIR, pilar=PILAR)["weights"], "spotcheck_summary.json")
        if not os.path.exists(summary_path):
            raise RuntimeError("[P5 Stage B] Stage A spot-check summary not found -- run Stage A first.")
        with open(summary_path) as f:
            spotcheck_results = json.load(f)

    selection = select_winning_feature_group(spotcheck_results, n_classes=NUM_CLASSES)
    winning_group = selection["winner"]
    print(f"[INFO][P5-B] Auto-selected feature group: {winning_group}  ({selection['reason']})")
    if selection["below_chance_warning"]:
        print(f"[PERINGATAN] Akurasi spot-check pemenang ({winning_group}) tidak melampaui chance level "
              f"({selection['chance_level_pct']:.2f}%). Hasil skala penuh tetap dijalankan otomatis, "
              f"namun perlu ditinjau kritis oleh peneliti.")

    with open(os.path.join(stage_weights_dir, "feature_selection_decision.json"), "w") as f:
        json.dump(selection, f, indent=2)

    if subject_ids is None:
        subject_ids = discover_subject_ids(raw_dir)
    fullscale_results = {}
    skipped_no_data = []

    for subject_id in subject_ids:
        model_path = build_model_path(stage_weights_dir, "P5_Fullscale", winning_group, subject_id)
        if os.path.exists(model_path):
            print(f"[SKIP][P5-B] Model for {subject_id} already exists.")
            existing_json = os.path.join(stage_weights_dir, f"results_{subject_id}.json")
            if os.path.exists(existing_json):
                with open(existing_json) as f:
                    fullscale_results[subject_id] = json.load(f)
            continue

        X_3d, y = load_shifted_band_epochs(subject_id, raw_dir)
        if X_3d is None:
            print(f"[WARNING][P5-B] No epochs extracted for subject {subject_id}; skipping.")
            skipped_no_data.append(subject_id)
            continue

        result = train_and_evaluate_one(X_3d, y, winning_group, stage_weights_dir, subject_id, tag="P5_Fullscale")
        fullscale_results[subject_id] = result

        with open(os.path.join(stage_weights_dir, f"results_{subject_id}.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(f"[INFO][P5-B] {subject_id}: test acc {result['test_accuracy']*100:6.2f}%  "
              f"val acc {result['val_accuracy']*100:6.2f}%  coverage {result['n_classes_covered']}/19")

    write_report(spotcheck_results, selection, fullscale_results, subject_ids, skipped_no_data)
    return fullscale_results, selection


def write_report(spotcheck_results, selection, fullscale_results, all_subject_ids, skipped_no_data):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    n_complete = len(fullscale_results)
    n_total = len(all_subject_ids)

    lines = []
    lines.append("# P5 -- Shifted Bandpass Filter: Experiment Report")
    lines.append("")
    lines.append("Variable tested: bandpass range (15-65Hz vs. the standard 0.5-50Hz broadband). "
                  "Locked: standard 5x1s windowing, SVM, E0 Baseline (no augmentation), phase_filter='all'.")
    lines.append("")
    if n_complete < n_total:
        lines.append(f"> **Status: PARTIAL ({n_complete}/{n_total} subjects).** This report reflects a "
                      f"smoke-test or in-progress run, not the full 12-subject grid. Re-run "
                      f"`run_p5_shifted_bandpass.py --stage b` on the lab machine to complete the grid; "
                      f"already-completed subjects are skipped automatically (auto-resume).")
        lines.append("")

    lines.append("## Stage A -- Feature Spot-check (S3, E0)")
    lines.append("")
    lines.append("| Feature Group | Test Accuracy (%) | Class Coverage |")
    lines.append("|---|---|---|")
    for g in FEATURE_GROUPS:
        r = spotcheck_results[g]
        lines.append(f"| {g} | {r['test_accuracy']*100:.4f} | {r['n_classes_covered']}/19 |")
    lines.append("")
    lines.append(f"**Automatic selection:** `{selection['winner']}` -- {selection['reason']}")
    lines.append("")
    if selection["below_chance_warning"]:
        lines.append(f"**[PERINGATAN]** Akurasi spot-check pemenang tidak melampaui chance level "
                      f"({selection['chance_level_pct']:.2f}% untuk 19 kelas). Hasil skala penuh tetap "
                      f"dijalankan otomatis; perlu ditinjau kritis sebelum dimasukkan ke Bab 6.")
        lines.append("")

    lines.append("## Stage B -- Full-scale Results")
    lines.append("")
    lines.append(f"Feature group used: `{selection['winner']}` | Subjects completed: {n_complete}/{n_total}")
    if skipped_no_data:
        lines.append(f"Subjects skipped (no raw data found): {skipped_no_data}")
    lines.append("")
    lines.append("| Subject | Test Accuracy (%) | Val Accuracy (%) | Class Coverage |")
    lines.append("|---|---|---|---|")
    for subject_id in all_subject_ids:
        r = fullscale_results.get(subject_id)
        if r is None:
            lines.append(f"| {subject_id} | -- | -- | not yet run |")
        else:
            lines.append(f"| {subject_id} | {r['test_accuracy']*100:.4f} | {r['val_accuracy']*100:.4f} | {r['n_classes_covered']}/19 |")
    if fullscale_results:
        accs = [r["test_accuracy"] * 100 for r in fullscale_results.values()]
        lines.append("")
        lines.append(f"Mean test accuracy so far: {np.mean(accs):.4f}% (std {np.std(accs, ddof=1) if len(accs) > 1 else 0.0:.4f} pp, n={len(accs)})")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO][P5] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P5 Shifted Bandpass: spot-check + full-scale SVM grid.")
    parser.add_argument("--stage", choices=["a", "b", "both"], default="both",
                         help="Run only Stage A (spot-check), only Stage B (full-scale), or both (default).")
    parser.add_argument("--subjects", nargs="+", default=None,
                         help="Restrict Stage B to specific subject IDs (e.g. --subjects S1 S2). "
                              "Default: all 12 subjects, auto-discovered.")
    args = parser.parse_args()

    spotcheck = None
    if args.stage in ("a", "both"):
        spotcheck = run_stage_a_spotcheck()
    if args.stage in ("b", "both"):
        run_stage_b_fullscale(spotcheck_results=spotcheck, subject_ids=args.subjects)

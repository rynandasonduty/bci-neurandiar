"""
backend/src/experiments_p4_p7/run_p7_coarse_to_fine.py

P7 -- Coarse-to-Fine Hierarchical Decoding. Variable tested: decision
structure (hierarchical vowel-group -> syllable vs. flat 19-way). Locked:
standard windowing/filter, SVM, E0 Baseline, phase_filter='all'.

Design principle: ONE three_way_split(random_state=42) per subject on the
standard E0/19-class dataset (built via the real, unmodified DatasetBuilder
-- identical methodology to P1-P3). All five sub-models are derived by
filtering that SAME split by label; none is independently re-split.

Sub-models (see dataset_builders_ext.SUBMODEL_LABEL_SETS for the exact label
sets, verified against real trial data by verify_p7_label_scheme.py):
    coarse    -- 9 first-syllable classes -> mapped to {A,I,E,O} (4 classes)
    fine_A    -- MA/MAN/SA (3 classes)
    fine_I    -- MI/PI/TI (3 classes)
    fine_E    -- BE/LE (2 classes)
    sa_branch -- KIT/YANG, second syllable, SA branch only (2 classes)
Group O (BO) has no fine-stage model: a coarse "O" prediction passes
straight through as BO.

Stage A (spot-check): coarse sub-model ONLY, S3 only, all 5 feature groups
    -> P7_CoarseToFine/Spotcheck_Coarse_S3/{feature_group}/
    fine_A/fine_I/fine_E/sa_branch always use Barlow directly (no spot-check
    -- their classification granularity is equivalent to P1-P3, per the
    agreed design).
Stage B (full-scale): all 12 subjects, all 5 sub-models plus two end-to-end
    metrics -> P7_CoarseToFine/Fullscale_12Subj/{coarse,fine_A,fine_I,
    fine_E,sa_branch}/

End-to-end metrics (beyond each sub-model's own accuracy):
  - First-syllable accuracy: computed directly from the shared held-out
    window-level test split (X_test/y_test) -- no trial pairing needed, no
    leakage. Compared against T18_p3_per_syllable_recall.csv (P3's own
    per-syllable recall), filtered to the 9 first-syllable rows.
  - Full-word accuracy: requires pairing a trial's slot-1 and slot-2 epochs,
    which the flat window-level split does not preserve. Reconstructed via
    pipeline/offline_trial_reader.OfflineTrialReader (already used by the
    production word-assembler training scripts), with an 80/20 trial-level
    holdout (test_size=0.2, random_state=42) mirroring
    models/train_word_assembler_s3.py's own methodology exactly. This
    trial-level split is independent of the window-level split used to
    train the sub-models, so it is not a strictly leakage-free estimate --
    the same caveat already applies to the existing word-assembler's
    reported accuracy (see docstring of compute_full_word_e2e_accuracy).

Usage:
    cd backend/src/experiments_p4_p7
    python run_p7_coarse_to_fine.py                # Stage A then Stage B
    python run_p7_coarse_to_fine.py --stage a       # spot-check only
    python run_p7_coarse_to_fine.py --stage b       # full-scale only
"""
import os
import sys
import glob
import json
import pickle
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment, RAW_DATA_DIR
from preprocessing.build_dataset import DatasetBuilder
from features.extract_eeg_features import EEGFeatureExtractor, FEATURE_GROUPS
from models.classical_models import ClassicalClassifier
from utils.data_utils import three_way_split, fit_and_apply_scaler
from pipeline.offline_trial_reader import OfflineTrialReader
from experiments_p4_p7.dataset_builders_ext import (
    select_winning_feature_group, filter_split_by_labels, map_labels_to_vowel_group_ids,
    SUBMODEL_LABEL_SETS, VOWEL_GROUP_TO_ID, ID_TO_VOWEL_GROUP, LABEL_TO_SYLLABLE,
    DETERMINISTIC_FIRST_SYLLABLE_TO_WORD, SA_BRANCH_SECOND_SYLLABLE_TO_WORD,
)

PILAR = "P7_CoarseToFine"
EXP_ID = "E0_Baseline"
SPOTCHECK_SUBJECT = "S3"
SPOTCHECK_STAGE_DIR = "Spotcheck_Coarse_S3"
FULLSCALE_STAGE_DIR = "Fullscale_12Subj"

E0_PROCESSOR_PARAMS = {"band": "broadband", "apply_ica": False, "target_fs": 256}
PHASE_FILTER = "all"
SPLIT_RANDOM_STATE = 42

NUM_CLASSES_FLAT = 19
NUM_CLASSES_COARSE = 4  # A/I/E/O

WORD_TO_SYLLABLES = {
    "MAKAN": ("MA", "KAN"), "MINUM": ("MI", "NUM"), "BERAK": ("BE", "RAK"),
    "PIPIS": ("PI", "PIS"), "MANDI": ("MAN", "DI"), "BOSAN": ("BO", "SAN"),
    "LELAH": ("LE", "LAH"), "SAKIT": ("SA", "KIT"), "TIDUR": ("TI", "DUR"),
    "SAYANG": ("SA", "YANG"),
}

REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'P4_P7_Experiments'))
REPORT_PATH = os.path.join(REPORTS_DIR, "P7_CoarseToFine_report.md")
T18_CSV_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'notebooks', 'reports', 'data_export_claude',
    'T18_p3_per_syllable_recall.csv'
))


def discover_subject_ids():
    log_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "logs", "*_experiment_log.txt")))
    return [os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files]


def build_standard_e0_split_raw(subject_id):
    """Build one subject's standard E0/19-class dataset (raw, unsplit) via
    the real, unmodified DatasetBuilder -- identical methodology to P1-P3."""
    builder = DatasetBuilder(
        exp_id=f"P7_Standard_{subject_id}", phase_filter=PHASE_FILTER,
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


def sanity_check_against_p3(X_test_3d, y_test, subject_id):
    """Re-extract Barlow features from the rebuilt raw test epochs and
    re-apply P3's own already-fitted E0_Baseline/barlow scaler; compare
    against P3's loaded, already-scaled Xtest/ytest. Proof that this
    rebuild reproduces P3's exact test set, not just a shape check."""
    p3_weights_dir = setup_experiment("E0_Baseline", pilar="P3_SVM")["weights"]
    p3_model_path = os.path.join(p3_weights_dir, f"SVM_barlow_E0_Baseline_{subject_id}.pkl")
    p3_scaler_path = os.path.join(p3_weights_dir, f"scaler_SVM_barlow_E0_Baseline_{subject_id}.pkl")
    p3_xtest_path = os.path.join(p3_weights_dir, f"Xtest_SVM_barlow_E0_Baseline_{subject_id}.npy")
    p3_ytest_path = os.path.join(p3_weights_dir, f"ytest_SVM_barlow_E0_Baseline_{subject_id}.npy")

    if not all(os.path.exists(p) for p in (p3_model_path, p3_scaler_path, p3_xtest_path, p3_ytest_path)):
        return {"available": False}

    p3_xtest = np.load(p3_xtest_path)
    p3_ytest = np.load(p3_ytest_path)
    with open(p3_scaler_path, "rb") as f:
        p3_scaler = pickle.load(f)

    extractor = EEGFeatureExtractor(fs=E0_PROCESSOR_PARAMS["target_fs"])
    X_test_feat = extractor.transform(X_test_3d, groups=["barlow"])
    X_test_feat = np.nan_to_num(X_test_feat, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_rescaled = p3_scaler.transform(X_test_feat)

    shape_match = X_test_rescaled.shape == p3_xtest.shape
    y_match = bool(shape_match and np.array_equal(y_test, p3_ytest))
    value_match, max_abs_diff = False, None
    if shape_match:
        max_abs_diff = float(np.max(np.abs(X_test_rescaled - p3_xtest)))
        value_match = bool(np.allclose(X_test_rescaled, p3_xtest, atol=1e-6))

    return {
        "available": True, "shape_match": bool(shape_match), "y_match": y_match,
        "value_match": value_match, "max_abs_diff": max_abs_diff,
        "rebuilt_shape": list(X_test_rescaled.shape), "p3_shape": list(p3_xtest.shape),
    }


def build_and_evaluate_submodel(X_train_3d, y_train_full, X_val_3d, y_val_full, X_test_3d, y_test_full,
                                 label_set, feat_group, output_dir, subject_id, tag, label_transform=None):
    """Filter the shared split by label, extract features, fit a dedicated
    scaler on this sub-model's own filtered training subset, train, and
    evaluate. Shared by Stage A (coarse only) and Stage B (all 5)."""
    os.makedirs(output_dir, exist_ok=True)

    Xtr_3d, ytr = filter_split_by_labels(X_train_3d, y_train_full, label_set)
    Xva_3d, yva = filter_split_by_labels(X_val_3d, y_val_full, label_set)
    Xte_3d, yte = filter_split_by_labels(X_test_3d, y_test_full, label_set)

    extractor = EEGFeatureExtractor(fs=E0_PROCESSOR_PARAMS["target_fs"])
    groups = None if feat_group == "all" else [feat_group]
    Xtr = np.nan_to_num(extractor.transform(Xtr_3d, groups=groups), nan=0.0, posinf=0.0, neginf=0.0)
    Xva = np.nan_to_num(extractor.transform(Xva_3d, groups=groups), nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(extractor.transform(Xte_3d, groups=groups), nan=0.0, posinf=0.0, neginf=0.0)

    if label_transform is not None:
        ytr, yva, yte = label_transform(ytr), label_transform(yva), label_transform(yte)

    scaler_path = os.path.join(output_dir, f"scaler_{tag}_{feat_group}_{subject_id}.pkl")
    Xtr, Xva, Xte, scaler = fit_and_apply_scaler(Xtr, Xva, Xte, save_path=scaler_path)

    np.save(os.path.join(output_dir, f"Xtest_{tag}_{feat_group}_{subject_id}.npy"), Xte)
    np.save(os.path.join(output_dir, f"ytest_{tag}_{feat_group}_{subject_id}.npy"), yte)

    model = ClassicalClassifier(model_type='svm', C=10)
    model.train(Xtr, ytr)
    val_acc = model.evaluate(Xva, yva)
    test_acc = model.evaluate(Xte, yte)

    y_pred = model.pipeline.predict(Xte)
    classes_covered = sorted(set(np.asarray(yte)[y_pred == yte].tolist()))

    model_path = os.path.join(output_dir, f"SVM_{tag}_{feat_group}_{subject_id}.pkl")
    model.save_model(model_path)

    result = {
        "feature_group": feat_group, "subject_id": subject_id,
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "val_accuracy": float(val_acc), "test_accuracy": float(test_acc),
        "n_classes_covered": len(classes_covered), "classes_covered": classes_covered,
        "model_path": model_path, "scaler_path": scaler_path,
    }
    return result, model_path, scaler_path


class SubModelBundle:
    """Loaded model+scaler for one P7 sub-model, ready for single-epoch
    inference. Deliberately NOT pipeline.svm_champion.SVMChampion (that
    class hardcodes a fixed 19-class output normalization for one specific
    champion model) -- this is a small, generic analogue sized to whatever
    class space a given sub-model was trained on."""

    def __init__(self, model_path, scaler_path, feat_group, fs=256):
        self.model = ClassicalClassifier(model_type='svm', C=10)
        self.model.load_model(model_path)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.feat_group = feat_group
        self.extractor = EEGFeatureExtractor(fs=fs)

    def predict_single(self, epoch_2d):
        """epoch_2d: (channels, time). Returns one hard-label prediction
        (int), in whatever label space this sub-model was trained on."""
        X_3d = epoch_2d[np.newaxis, :, :]
        groups = None if self.feat_group == "all" else [self.feat_group]
        features = self.extractor.transform(X_3d, groups=groups)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = self.scaler.transform(features)
        pred = self.model.pipeline.predict(features_scaled)[0]
        return int(pred)


def predict_first_syllable(epoch_2d, coarse_bundle, fine_bundles):
    """Coarse -> fine hierarchical prediction for one slot-1 epoch. Returns
    a syllable name string (e.g. 'MA', 'BO', 'SA')."""
    vowel_group_id = coarse_bundle.predict_single(epoch_2d)
    vowel_group = ID_TO_VOWEL_GROUP[vowel_group_id]
    if vowel_group == "O":
        return "BO"  # single-member group, no fine-stage model by design
    syl_label = fine_bundles[vowel_group].predict_single(epoch_2d)
    return LABEL_TO_SYLLABLE[syl_label]


def predict_word_for_trial(epoch_slot1_2d, epoch_slot2_2d, coarse_bundle, fine_bundles, sa_branch_bundle):
    """Full coarse -> fine -> deterministic dictionary (+ sa_branch for the
    SA case) hierarchical prediction for one trial. Returns a word string."""
    first_syl = predict_first_syllable(epoch_slot1_2d, coarse_bundle, fine_bundles)
    if first_syl in DETERMINISTIC_FIRST_SYLLABLE_TO_WORD:
        return DETERMINISTIC_FIRST_SYLLABLE_TO_WORD[first_syl]
    if first_syl == "SA":
        sa_label = sa_branch_bundle.predict_single(epoch_slot2_2d)
        second_syl = LABEL_TO_SYLLABLE[sa_label]
        return SA_BRANCH_SECOND_SYLLABLE_TO_WORD.get(second_syl, "UNKNOWN")
    return "UNKNOWN"  # not reachable given full 9-class coarse->fine coverage


def compute_first_syllable_e2e_accuracy(X_test_3d, y_test, coarse_bundle, fine_bundles):
    """Leakage-free end-to-end first-syllable accuracy: computed directly
    from the shared held-out window-level test split, filtered to the 9
    first-syllable classes."""
    mask = np.isin(y_test, list(SUBMODEL_LABEL_SETS["coarse"]))
    X_sub, y_true = X_test_3d[mask], y_test[mask]

    y_pred_syl, y_true_syl = [], []
    for i in range(len(X_sub)):
        y_pred_syl.append(predict_first_syllable(X_sub[i], coarse_bundle, fine_bundles))
        y_true_syl.append(LABEL_TO_SYLLABLE[int(y_true[i])])

    accuracy = accuracy_score(y_true_syl, y_pred_syl) if y_true_syl else 0.0

    per_syllable = {}
    for syl in sorted(set(y_true_syl)):
        idxs = [i for i, t in enumerate(y_true_syl) if t == syl]
        correct = sum(1 for i in idxs if y_pred_syl[i] == syl)
        per_syllable[syl] = {"n": len(idxs), "correct": correct, "recall": correct / len(idxs)}

    return {"n_test_samples": len(y_true_syl), "accuracy": float(accuracy), "per_syllable_recall": per_syllable}


def compute_full_word_e2e_accuracy(subject_id, coarse_bundle, fine_bundles, sa_branch_bundle):
    """Full-word end-to-end accuracy via trial-level reconstruction.

    Requires pairing a trial's slot-1 and slot-2 epochs, which the flat
    window-level three_way_split does not preserve, so OfflineTrialReader
    (already used by models/train_word_assembler.py -- read-only reuse
    here) is used to reconstruct real trials directly from raw CSV+log.

    An 80/20 trial-level holdout (test_size=0.2, random_state=42) mirrors
    train_word_assembler_s3.py's own methodology exactly, so the two
    numbers are comparable "word accuracy" figures. This trial-level split
    is independent of the window-level three_way_split used to train the
    coarse/fine/sa_branch sub-models -- some evaluated trials may have
    contributed windows to those sub-models' own training data. The same
    caveat already applies to train_word_assembler.py/train_word_assembler_
    s3.py's own reported accuracy (they pool ALL trials, including ones
    whose windows trained the champion SVM, and only hold out their own 20%
    for the assembler's fit) -- so this is at least as rigorous as existing
    precedent, not less. Treat compute_first_syllable_e2e_accuracy's result
    as the leakage-free estimate; this one as a secondary,
    precedent-consistent number.
    """
    reader = OfflineTrialReader(RAW_DATA_DIR, E0_PROCESSOR_PARAMS)
    try:
        trials_meta = reader.list_valid_trials(subject_id)
    except Exception as e:
        return {"available": False, "reason": f"could not list trials: {e}"}

    valid_trials = []
    n_skipped_word, n_skipped_artifact = 0, 0
    for idx, meta in enumerate(trials_meta):
        word = meta["word"].strip().upper()
        if word not in WORD_TO_SYLLABLES:
            n_skipped_word += 1
            continue
        try:
            trial = reader.read_trial(subject_id, trial_index=idx)
        except ValueError:
            n_skipped_artifact += 1
            continue
        valid_trials.append(trial)

    if len(valid_trials) < 10:
        return {"available": False, "reason": f"only {len(valid_trials)} valid trials reconstructed (need >= 10)"}

    words = [t["word"].strip().upper() for t in valid_trials]
    _, counts = np.unique(words, return_counts=True)
    can_stratify = len(set(words)) > 1 and min(counts) >= 2
    _, test_trials = train_test_split(
        valid_trials, test_size=0.2, random_state=SPLIT_RANDOM_STATE,
        stratify=words if can_stratify else None,
    )

    y_true_words, y_pred_words = [], []
    for trial in test_trials:
        y_true_words.append(trial["word"].strip().upper())
        y_pred_words.append(predict_word_for_trial(
            trial["epoch_slot1"], trial["epoch_slot2"], coarse_bundle, fine_bundles, sa_branch_bundle
        ))

    accuracy = accuracy_score(y_true_words, y_pred_words)

    return {
        "available": True,
        "n_valid_trials_total": len(valid_trials),
        "n_skipped_unknown_word": n_skipped_word,
        "n_skipped_artifact_rejected": n_skipped_artifact,
        "n_test_trials": len(test_trials),
        "accuracy": float(accuracy),
        "caveat": (
            "Trial-level 80/20 holdout (test_size=0.2, random_state=42), mirroring "
            "train_word_assembler_s3.py's own methodology. Independent of the window-level "
            "three_way_split used to train the sub-models, so this is not a strictly "
            "leakage-free estimate -- same caveat already applies to the existing word "
            "assembler's reported accuracy. Use first_syllable_e2e for the leakage-free figure."
        ),
    }


def run_stage_a_spotcheck():
    print(f"\n{'=' * 70}\n P7 Coarse-to-Fine -- Stage A: Spot-check (coarse only, {SPOTCHECK_SUBJECT}, {EXP_ID})\n{'=' * 70}")

    X_3d, y = build_standard_e0_split_raw(SPOTCHECK_SUBJECT)
    if X_3d is None:
        raise RuntimeError(f"[P7 Stage A] No epochs extracted for subject {SPOTCHECK_SUBJECT}.")
    print(f"[INFO][P7-A] Windowed samples extracted: {X_3d.shape[0]} (shape per sample: {X_3d.shape[1:]})")

    X_train_3d, X_val_3d, X_test_3d, y_train, y_val, y_test = three_way_split(X_3d, y, random_state=SPLIT_RANDOM_STATE)

    stage_weights_dir = setup_experiment(SPOTCHECK_STAGE_DIR, pilar=PILAR)["weights"]
    spotcheck_results = {}
    for feat_group in FEATURE_GROUPS:
        group_dir = os.path.join(stage_weights_dir, feat_group)
        result, _, _ = build_and_evaluate_submodel(
            X_train_3d, y_train, X_val_3d, y_val, X_test_3d, y_test,
            label_set=SUBMODEL_LABEL_SETS["coarse"], feat_group=feat_group,
            output_dir=group_dir, subject_id=SPOTCHECK_SUBJECT, tag="P7_Spotcheck_Coarse",
            label_transform=map_labels_to_vowel_group_ids,
        )
        spotcheck_results[feat_group] = result
        with open(os.path.join(group_dir, f"results_{feat_group}.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO][P7-A] {feat_group:<10} test acc {result['test_accuracy']*100:6.2f}%  "
              f"coverage {result['n_classes_covered']}/4")

    with open(os.path.join(stage_weights_dir, "spotcheck_summary.json"), "w") as f:
        json.dump(spotcheck_results, f, indent=2)

    print(f"\n{'Feature Group':<15}{'Test Acc %':>12}{'Class Coverage':>18}")
    print("-" * 45)
    for g in FEATURE_GROUPS:
        r = spotcheck_results[g]
        print(f"{g:<15}{r['test_accuracy']*100:>12.4f}{r['n_classes_covered']:>15}/4")
    print("-" * 45)

    return spotcheck_results


def run_stage_b_fullscale(spotcheck_results=None, subject_ids=None):
    print(f"\n{'=' * 70}\n P7 Coarse-to-Fine -- Stage B: Full-scale ({EXP_ID})\n{'=' * 70}")

    if spotcheck_results is None:
        summary_path = os.path.join(setup_experiment(SPOTCHECK_STAGE_DIR, pilar=PILAR)["weights"], "spotcheck_summary.json")
        if not os.path.exists(summary_path):
            raise RuntimeError("[P7 Stage B] Stage A spot-check summary not found -- run Stage A first.")
        with open(summary_path) as f:
            spotcheck_results = json.load(f)

    selection = select_winning_feature_group(spotcheck_results, n_classes=NUM_CLASSES_COARSE)
    winning_coarse_group = selection["winner"]
    print(f"[INFO][P7-B] Auto-selected coarse feature group: {winning_coarse_group}  ({selection['reason']})")
    if selection["below_chance_warning"]:
        print(f"[PERINGATAN] Akurasi spot-check pemenang (coarse, {winning_coarse_group}) tidak melampaui "
              f"chance level ({selection['chance_level_pct']:.2f}% untuk 4 kelas). Hasil skala penuh tetap "
              f"dijalankan otomatis, namun perlu ditinjau kritis oleh peneliti.")

    fullscale_root = setup_experiment(FULLSCALE_STAGE_DIR, pilar=PILAR)["weights"]
    with open(os.path.join(fullscale_root, "feature_selection_decision.json"), "w") as f:
        json.dump(selection, f, indent=2)

    sub_dirs = {name: os.path.join(fullscale_root, name) for name in SUBMODEL_LABEL_SETS}

    if subject_ids is None:
        subject_ids = discover_subject_ids()

    fullscale_results = {}
    skipped_no_data = []

    for subject_id in subject_ids:
        e2e_json_path = os.path.join(fullscale_root, f"results_{subject_id}.json")
        if os.path.exists(e2e_json_path):
            print(f"[SKIP][P7-B] Subject {subject_id} already complete.")
            with open(e2e_json_path) as f:
                fullscale_results[subject_id] = json.load(f)
            continue

        X_3d, y = build_standard_e0_split_raw(subject_id)
        if X_3d is None:
            print(f"[WARNING][P7-B] No epochs extracted for subject {subject_id}; skipping.")
            skipped_no_data.append(subject_id)
            continue

        X_train_3d, X_val_3d, X_test_3d, y_train, y_val, y_test = three_way_split(X_3d, y, random_state=SPLIT_RANDOM_STATE)
        sanity = sanity_check_against_p3(X_test_3d, y_test, subject_id)

        sub_results = {}
        coarse_result, coarse_model_path, coarse_scaler_path = build_and_evaluate_submodel(
            X_train_3d, y_train, X_val_3d, y_val, X_test_3d, y_test,
            SUBMODEL_LABEL_SETS["coarse"], winning_coarse_group, sub_dirs["coarse"], subject_id, "P7_coarse",
            label_transform=map_labels_to_vowel_group_ids,
        )
        sub_results["coarse"] = coarse_result

        fine_paths = {}
        for name in ("fine_A", "fine_I", "fine_E", "sa_branch"):
            result, model_path, scaler_path = build_and_evaluate_submodel(
                X_train_3d, y_train, X_val_3d, y_val, X_test_3d, y_test,
                SUBMODEL_LABEL_SETS[name], "barlow", sub_dirs[name], subject_id, f"P7_{name}",
            )
            sub_results[name] = result
            fine_paths[name] = (model_path, scaler_path)

        for name, res in sub_results.items():
            print(f"[INFO][P7-B] {subject_id}/{name}: test acc {res['test_accuracy']*100:6.2f}%  "
                  f"coverage {res['n_classes_covered']}")

        coarse_bundle = SubModelBundle(coarse_model_path, coarse_scaler_path, winning_coarse_group,
                                        fs=E0_PROCESSOR_PARAMS["target_fs"])
        fine_bundles = {
            "A": SubModelBundle(*fine_paths["fine_A"], "barlow", fs=E0_PROCESSOR_PARAMS["target_fs"]),
            "I": SubModelBundle(*fine_paths["fine_I"], "barlow", fs=E0_PROCESSOR_PARAMS["target_fs"]),
            "E": SubModelBundle(*fine_paths["fine_E"], "barlow", fs=E0_PROCESSOR_PARAMS["target_fs"]),
        }
        sa_bundle = SubModelBundle(*fine_paths["sa_branch"], "barlow", fs=E0_PROCESSOR_PARAMS["target_fs"])

        first_syl_e2e = compute_first_syllable_e2e_accuracy(X_test_3d, y_test, coarse_bundle, fine_bundles)
        full_word_e2e = compute_full_word_e2e_accuracy(subject_id, coarse_bundle, fine_bundles, sa_bundle)

        print(f"[INFO][P7-B] {subject_id}: first-syllable e2e acc {first_syl_e2e['accuracy']*100:6.2f}%  "
              f"(n={first_syl_e2e['n_test_samples']}) | full-word e2e acc "
              f"{(full_word_e2e['accuracy']*100 if full_word_e2e['available'] else float('nan')):6.2f}%  "
              f"(n={full_word_e2e.get('n_test_trials', 0)})")

        result = {
            "subject_id": subject_id, "winning_coarse_feature_group": winning_coarse_group,
            "sub_models": sub_results, "sanity_check_vs_p3": sanity,
            "first_syllable_e2e": first_syl_e2e, "full_word_e2e": full_word_e2e,
        }
        fullscale_results[subject_id] = result
        with open(e2e_json_path, "w") as f:
            json.dump(result, f, indent=2)

    write_report(spotcheck_results, selection, fullscale_results, subject_ids, skipped_no_data)
    return fullscale_results, selection


def load_t18_first_syllable_baseline():
    """Read-only load of the existing P3 per-syllable recall table, filtered
    to the 9 first-syllable rows, for reference in the report. Returns None
    if the file isn't found (e.g. this repo checkout predates that export)."""
    if not os.path.exists(T18_CSV_PATH):
        return None
    try:
        df = pd.read_csv(T18_CSV_PATH)
        first_syllables = list(SUBMODEL_LABEL_SETS["coarse"])
        names = [LABEL_TO_SYLLABLE[v] for v in first_syllables]
        return df[df["Syllable"].isin(names)].copy()
    except Exception:
        return None


def write_report(spotcheck_results, selection, fullscale_results, all_subject_ids, skipped_no_data):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    n_complete = len(fullscale_results)
    n_total = len(all_subject_ids)

    lines = []
    lines.append("# P7 -- Coarse-to-Fine Hierarchical Decoding: Experiment Report")
    lines.append("")
    lines.append("Variable tested: decision structure (hierarchical vowel-group -> syllable vs. flat "
                  "19-way). Locked: standard windowing/filter, SVM, E0 Baseline, phase_filter='all'. "
                  "One three_way_split(seed=42) per subject; all 5 sub-models derived by filtering "
                  "that same split by label.")
    lines.append("")
    if n_complete < n_total:
        lines.append(f"> **Status: PARTIAL ({n_complete}/{n_total} subjects).** This report reflects a "
                      f"smoke-test or in-progress run. Re-run `run_p7_coarse_to_fine.py --stage b` on the "
                      f"lab machine to complete the grid; already-completed subjects are skipped "
                      f"automatically (auto-resume).")
        lines.append("")

    lines.append("## Stage A -- Coarse Sub-model Feature Spot-check (S3, E0)")
    lines.append("")
    lines.append("| Feature Group | Test Accuracy (%) | Class Coverage (/4) |")
    lines.append("|---|---|---|")
    for g in FEATURE_GROUPS:
        r = spotcheck_results[g]
        lines.append(f"| {g} | {r['test_accuracy']*100:.4f} | {r['n_classes_covered']} |")
    lines.append("")
    lines.append(f"**Automatic selection (coarse sub-model):** `{selection['winner']}` -- {selection['reason']}")
    lines.append("")
    lines.append("`fine_A`/`fine_I`/`fine_E`/`sa_branch` always use Barlow directly (no spot-check), per "
                  "the agreed design -- their classification granularity is equivalent to P1-P3.")
    lines.append("")
    if selection["below_chance_warning"]:
        lines.append(f"**[PERINGATAN]** Akurasi spot-check pemenang (coarse) tidak melampaui chance level "
                      f"({selection['chance_level_pct']:.2f}% untuk 4 kelas). Hasil skala penuh tetap "
                      f"dijalankan otomatis; perlu ditinjau kritis sebelum dimasukkan ke Bab 6.")
        lines.append("")

    lines.append("## Stage B -- Full-scale Sub-model Results")
    lines.append("")
    lines.append(f"Coarse feature group used: `{selection['winner']}` | Subjects completed: {n_complete}/{n_total}")
    if skipped_no_data:
        lines.append(f"Subjects skipped (no raw data found): {skipped_no_data}")
    lines.append("")
    lines.append("| Subject | coarse (%) | fine_A (%) | fine_I (%) | fine_E (%) | sa_branch (%) |")
    lines.append("|---|---|---|---|---|---|")
    for subject_id in all_subject_ids:
        r = fullscale_results.get(subject_id)
        if r is None:
            lines.append(f"| {subject_id} | -- | -- | -- | -- | -- |")
        else:
            sm = r["sub_models"]
            lines.append(f"| {subject_id} | {sm['coarse']['test_accuracy']*100:.2f} | "
                          f"{sm['fine_A']['test_accuracy']*100:.2f} | {sm['fine_I']['test_accuracy']*100:.2f} | "
                          f"{sm['fine_E']['test_accuracy']*100:.2f} | {sm['sa_branch']['test_accuracy']*100:.2f} |")

    lines.append("")
    lines.append("## End-to-end Metrics")
    lines.append("")
    lines.append("| Subject | First-syllable e2e acc (%) | Full-word e2e acc (%) | Full-word n test trials |")
    lines.append("|---|---|---|---|")
    for subject_id in all_subject_ids:
        r = fullscale_results.get(subject_id)
        if r is None:
            lines.append(f"| {subject_id} | -- | -- | -- |")
        else:
            fw = r["full_word_e2e"]
            fw_acc = f"{fw['accuracy']*100:.2f}" if fw.get("available") else "n/a"
            fw_n = fw.get("n_test_trials", "n/a") if fw.get("available") else "n/a"
            lines.append(f"| {subject_id} | {r['first_syllable_e2e']['accuracy']*100:.2f} | {fw_acc} | {fw_n} |")

    if fullscale_results:
        first_accs = [r["first_syllable_e2e"]["accuracy"] * 100 for r in fullscale_results.values()]
        word_accs = [r["full_word_e2e"]["accuracy"] * 100 for r in fullscale_results.values() if r["full_word_e2e"].get("available")]
        lines.append("")
        lines.append(f"Mean first-syllable e2e accuracy so far: {np.mean(first_accs):.4f}% (n={len(first_accs)})")
        if word_accs:
            lines.append(f"Mean full-word e2e accuracy so far: {np.mean(word_accs):.4f}% (n={len(word_accs)})")

    lines.append("")
    lines.append("### Full-word e2e methodology caveat")
    lines.append("")
    lines.append(
        "Trial-level 80/20 holdout (test_size=0.2, random_state=42), mirroring "
        "`train_word_assembler_s3.py`'s own methodology exactly, so the two are comparable 'word "
        "accuracy' figures. This trial-level split is independent of the window-level three_way_split "
        "used to train the coarse/fine/sa_branch sub-models -- some evaluated trials may have "
        "contributed windows to those sub-models' own training data, so this is not a strictly "
        "leakage-free estimate. The same caveat already applies to the existing word assembler's "
        "reported accuracy (it pools ALL trials, including ones whose windows trained the champion SVM, "
        "and only holds out its own 20% for the assembler's fit) -- so this is at least as rigorous as "
        "existing precedent. Treat 'first-syllable e2e accuracy' above (computed from the proper "
        "held-out window-level test split) as the leakage-free estimate for this paradigm."
    )

    t18 = load_t18_first_syllable_baseline()
    lines.append("")
    lines.append("### Baseline reference: P3 per-syllable recall, 9 first-syllable classes (T18)")
    lines.append("")
    if t18 is not None and len(t18) > 0:
        lines.append("| Syllable | Mean Recall (P3, %) | Std Recall (pp) | N Subjects |")
        lines.append("|---|---|---|---|")
        for _, row in t18.iterrows():
            lines.append(f"| {row['Syllable']} | {row['Mean Recall']*100:.2f} | {row['Std Recall']*100:.2f} | {int(row['N Subjects'])} |")
    else:
        lines.append(f"_T18 not found at `{T18_CSV_PATH}` -- baseline comparison to be filled in manually or via `P4_P7_Analysis.ipynb`._")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO][P7] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P7 Coarse-to-Fine: spot-check + full-scale hierarchical SVM grid.")
    parser.add_argument("--stage", choices=["a", "b", "both"], default="both",
                         help="Run only Stage A (coarse spot-check), only Stage B (full-scale), or both (default).")
    parser.add_argument("--subjects", nargs="+", default=None,
                         help="Restrict Stage B to specific subject IDs (e.g. --subjects S1 S2). "
                              "Default: all 12 subjects, auto-discovered.")
    args = parser.parse_args()

    spotcheck = None
    if args.stage in ("a", "both"):
        spotcheck = run_stage_a_spotcheck()
    if args.stage in ("b", "both"):
        run_stage_b_fullscale(spotcheck_results=spotcheck, subject_ids=args.subjects)

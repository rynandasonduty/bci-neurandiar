"""
backend/src/experiments_p4_p7/run_p7_coarse_ablation.py

P7 coarse sub-model ablation, Fase 1: five factors tested INDIVIDUALLY
against the existing coarse baseline (E0, Barlow, unweighted, C=10 --
P7_CoarseToFine/Fullscale_12Subj/coarse/, never retrained/modified here).
fine_A/fine_I/fine_E/sa_branch are untouched throughout -- only the coarse
sub-model is varied.

All five variants reuse p7_coarse_cache.py's cached raw split + Barlow
features (built once per subject, on first use, then persisted) instead of
re-reading raw CSVs -- see that module's docstring for the caching
rationale. This is the main lever for hitting the "2-5 hour, not 24 hour"
training budget: raw-CSV-read cost is paid ONCE per subject across all five
variants, not five times.

Varian A -- Class-weight balanced: WeightedClassicalClassifier
    (classical_models_ext.py, class_weight='balanced' SVC). Cached Barlow
    features used as-is.
Varian B -- E5 augmentation: EXPERIMENT_RECIPES["E5_Data_Augmentation"]'s
    augmentation_params (models/run_subject_dependent.py, read-only
    reference, copied as a constant below) applied via
    SignalProcessor.apply_augmentation() (called, never modified) to the
    cached RAW training epochs only; val/test reuse cached Barlow features
    unchanged. Only the newly-augmented half needs fresh feature
    extraction -- the original half's Barlow features are already cached,
    so this variant's feature-extraction cost is ~50% of a naive
    "re-extract everything" implementation.
Varian C -- C tuning per subject: C in {0.1, 1, 10, 100, 1000}, cached
    Barlow features scaled ONCE (scaling doesn't depend on C) and reused
    across the whole grid; the C with the highest VALIDATION accuracy is
    selected per subject (not a single C assumed optimal for everyone),
    and only that winning model is evaluated against the test set.
Varian D -- Feature group 'all': the only alternative feature group tested
    (not a full 5-group replay), because Stage A's spot-check already
    showed 'all' tied with 'barlow' (41.90% both) while the other three
    groups (time 35.24%, hjorth 39.05%, band_ratio 36.19%) were clearly
    behind -- no justification for re-testing groups already known to
    lose. Needs fresh 'all'-feature extraction from the cached RAW epochs
    (cache only stores Barlow features), but skips the raw-CSV-read step.
Varian E -- Probability calibration: the baseline SVM config (E0, Barlow,
    unweighted, C=10) wrapped in sklearn.calibration.CalibratedClassifierCV
    (cv=5, method='sigmoid'). Reports both raw accuracy and calibration
    quality (multi-class Brier score + a simple top-label ECE), since this
    variant's purpose is enabling Fase 3's confidence-gated coarse->fine
    post-processing, not raw accuracy by itself.

Fase 1 records each variant's mean delta vs. baseline and a paired
Wilcoxon test, and previews the Fase 2 (run_p7_coarse_combined.py)
automatic inclusion rule (delta > 1.0 pp) -- the actual inclusion decision
and the combined model are built by that script, not this one; Varian E's
special "always structurally included" exception is also decided there.

Usage:
    cd backend/src/experiments_p4_p7
    python run_p7_coarse_ablation.py                 # all 12 subjects
    python run_p7_coarse_ablation.py --subjects S1 S2
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np

from scipy.stats import wilcoxon
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.extract_eeg_features import EEGFeatureExtractor
from models.classical_models import ClassicalClassifier
from utils.data_utils import fit_and_apply_scaler
from preprocessing.signal_processor import SignalProcessor

from experiments_p4_p7 import run_p7_coarse_to_fine as p7base
from experiments_p4_p7.classical_models_ext import WeightedClassicalClassifier
from experiments_p4_p7.p7_coarse_cache import get_or_build_cached_coarse

PILAR = p7base.PILAR
FULLSCALE_STAGE_DIR = p7base.FULLSCALE_STAGE_DIR
REPORTS_DIR = p7base.REPORTS_DIR
PHASE1_REPORT_PATH = os.path.join(REPORTS_DIR, "P7_CoarseAblation_Phase1_report.md")

VARIANT_DIRS = {
    "A_balanced": "coarse_variant_a_balanced",
    "B_augmented": "coarse_variant_b_augmented",
    "C_tuned": "coarse_variant_c_tuned",
    "D_feat_all": "coarse_variant_d_feat_all",
    "E_calibrated": "coarse_variant_e_calibrated",
}
VARIANT_ORDER = ["A_balanced", "B_augmented", "C_tuned", "D_feat_all", "E_calibrated"]

# EXPERIMENT_RECIPES["E5_Data_Augmentation"]["augmentation_params"],
# models/run_subject_dependent.py -- read-only reference, copied verbatim.
E5_AUGMENTATION_PARAMS = {"add_noise": True, "noise_factor": 0.05, "apply_jitter": True, "jitter_ms": 10}

C_GRID = [0.1, 1, 10, 100, 1000]
DEFAULT_C = 10
MIN_PAIRS_FOR_WILCOXON = 5
INCLUSION_THRESHOLD_PP = 1.0


def phase1_summary_path(fullscale_root):
    return os.path.join(fullscale_root, "phase1_summary.json")


def augment_training_epochs(X_3d, proc, aug_params):
    """Apply SignalProcessor.apply_augmentation() to each training epoch.

    X_3d is (samples, channels, time) -- the convention used throughout
    P4-P7 (see EEGFeatureExtractor.transform). apply_augmentation expects
    (time, channels), so each sample is transposed in and back out."""
    aug_list = [proc.apply_augmentation(sample.T, **aug_params).T for sample in X_3d]
    return np.array(aug_list, dtype=X_3d.dtype)


def classes_covered_from_predictions(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return sorted(set(y_true[y_pred == y_true].tolist()))


def multiclass_brier_score(y_true, proba, classes):
    """Mean squared error between one-hot true labels and predicted
    probability vectors, generalised to multi-class (lower is better)."""
    y_true = np.asarray(y_true)
    class_to_col = {int(c): i for i, c in enumerate(classes)}
    one_hot = np.zeros((len(y_true), len(classes)))
    for row, label in enumerate(y_true):
        one_hot[row, class_to_col[int(label)]] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true, proba, classes, n_bins=10):
    """Simple top-label ECE: bin samples by predicted top-1 confidence,
    compare each bin's mean confidence to its mean accuracy, weight by bin
    size (lower is better; 0 = perfectly calibrated)."""
    y_true = np.asarray(y_true)
    confidences = proba.max(axis=1)
    pred_labels = np.asarray(classes)[proba.argmax(axis=1)]
    correct = (pred_labels == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if not np.any(in_bin):
            continue
        ece += (in_bin.sum() / n) * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)


def baseline_svc_pipeline(C=DEFAULT_C, class_weight=None):
    """Builds an SVC/Pipeline matching ClassicalClassifier's own config
    (kernel='rbf', probability=True, random_state=42) directly, rather than
    going through ClassicalClassifier itself. CalibratedClassifierCV clones
    its estimator via sklearn.base.clone() internally (once per CV fold),
    which requires a real scikit-learn-compatible object (get_params/
    set_params) -- ClassicalClassifier is a plain wrapper class, not
    clonable, so it cannot be passed to CalibratedClassifierCV directly.
    classical_models.py itself is never modified; this only mirrors its
    SVC configuration in a clone-compatible shape."""
    kwargs = dict(kernel='rbf', C=C, probability=True, random_state=42)
    if class_weight is not None:
        kwargs['class_weight'] = class_weight
    return Pipeline([('classifier', SVC(**kwargs))])


def train_variant_a(cached, output_dir, subject_id):
    """Varian A: WeightedClassicalClassifier (class_weight='balanced'),
    cached Barlow features used as-is."""
    os.makedirs(output_dir, exist_ok=True)
    Xtr, ytr = cached["X_train_feat_barlow"], cached["y_train_coarse"]
    Xva, yva = cached["X_val_feat_barlow"], cached["y_val_coarse"]
    Xte, yte = cached["X_test_feat_barlow"], cached["y_test_coarse"]

    scaler_path = os.path.join(output_dir, f"scaler_P7_coarse_variant_a_barlow_{subject_id}.pkl")
    Xtr_s, Xva_s, Xte_s, _ = fit_and_apply_scaler(Xtr, Xva, Xte, save_path=scaler_path)

    np.save(os.path.join(output_dir, f"Xtest_P7_coarse_variant_a_barlow_{subject_id}.npy"), Xte_s)
    np.save(os.path.join(output_dir, f"ytest_P7_coarse_variant_a_barlow_{subject_id}.npy"), yte)

    model = WeightedClassicalClassifier(model_type='svm', C=DEFAULT_C)
    model.train(Xtr_s, ytr)
    val_acc = float(model.evaluate(Xva_s, yva))
    y_test_pred = model.pipeline.predict(Xte_s)
    test_acc = float(accuracy_score(yte, y_test_pred))

    model_path = os.path.join(output_dir, f"SVM_P7_coarse_variant_a_barlow_{subject_id}.pkl")
    model.save_model(model_path)

    return {
        "variant": "A_balanced", "subject_id": subject_id, "feature_group": "barlow",
        "val_accuracy": val_acc, "test_accuracy": test_acc,
        "classes_covered": classes_covered_from_predictions(yte, y_test_pred),
        "model_path": model_path, "scaler_path": scaler_path,
    }


def train_variant_b(cached, extractor, proc, output_dir, subject_id):
    """Varian B: plain ClassicalClassifier, coarse training set enriched
    with E5-recipe augmented copies. Only the augmented half is freshly
    feature-extracted -- the original half reuses cached Barlow features."""
    os.makedirs(output_dir, exist_ok=True)
    X_train_3d, ytr = cached["X_train_3d_coarse"], cached["y_train_coarse"]
    Xva, yva = cached["X_val_feat_barlow"], cached["y_val_coarse"]
    Xte, yte = cached["X_test_feat_barlow"], cached["y_test_coarse"]

    X_aug_3d = augment_training_epochs(X_train_3d, proc, E5_AUGMENTATION_PARAMS)
    Xtr_aug_feat = extractor.transform(X_aug_3d, groups=["barlow"])
    Xtr_aug_feat = np.nan_to_num(Xtr_aug_feat, nan=0.0, posinf=0.0, neginf=0.0)

    Xtr_enriched = np.concatenate([cached["X_train_feat_barlow"], Xtr_aug_feat], axis=0)
    y_train_enriched = np.concatenate([ytr, ytr], axis=0)

    # Shuffle after concatenation, mirroring EXPERIMENT_RECIPES["E5_Data_Augmentation"]'s
    # own pattern in run_subject_dependent.py. SVC's solution does not
    # depend on training-sample order.
    shuffle_idx = np.random.permutation(len(y_train_enriched))
    Xtr_enriched = Xtr_enriched[shuffle_idx]
    y_train_enriched = y_train_enriched[shuffle_idx]

    scaler_path = os.path.join(output_dir, f"scaler_P7_coarse_variant_b_barlow_{subject_id}.pkl")
    Xtr_s, Xva_s, Xte_s, _ = fit_and_apply_scaler(Xtr_enriched, Xva, Xte, save_path=scaler_path)

    np.save(os.path.join(output_dir, f"Xtest_P7_coarse_variant_b_barlow_{subject_id}.npy"), Xte_s)
    np.save(os.path.join(output_dir, f"ytest_P7_coarse_variant_b_barlow_{subject_id}.npy"), yte)

    model = ClassicalClassifier(model_type='svm', C=DEFAULT_C)
    model.train(Xtr_s, y_train_enriched)
    val_acc = float(model.evaluate(Xva_s, yva))
    y_test_pred = model.pipeline.predict(Xte_s)
    test_acc = float(accuracy_score(yte, y_test_pred))

    model_path = os.path.join(output_dir, f"SVM_P7_coarse_variant_b_barlow_{subject_id}.pkl")
    model.save_model(model_path)

    return {
        "variant": "B_augmented", "subject_id": subject_id, "feature_group": "barlow",
        "n_train_original": int(len(ytr)), "n_train_augmented_total": int(len(y_train_enriched)),
        "val_accuracy": val_acc, "test_accuracy": test_acc,
        "classes_covered": classes_covered_from_predictions(yte, y_test_pred),
        "model_path": model_path, "scaler_path": scaler_path,
    }


def train_variant_c(cached, output_dir, subject_id):
    """Varian C: C in {0.1,1,10,100,1000}, cached Barlow features scaled
    ONCE (scaling is independent of C) and reused across the whole grid;
    the C with the highest VAL accuracy is selected per subject, only that
    winner is evaluated on the test set."""
    os.makedirs(output_dir, exist_ok=True)
    Xtr, ytr = cached["X_train_feat_barlow"], cached["y_train_coarse"]
    Xva, yva = cached["X_val_feat_barlow"], cached["y_val_coarse"]
    Xte, yte = cached["X_test_feat_barlow"], cached["y_test_coarse"]

    scaler_path = os.path.join(output_dir, f"scaler_P7_coarse_variant_c_barlow_{subject_id}.pkl")
    Xtr_s, Xva_s, Xte_s, _ = fit_and_apply_scaler(Xtr, Xva, Xte, save_path=scaler_path)

    val_acc_per_c = {}
    models_per_c = {}
    for C in C_GRID:
        m = ClassicalClassifier(model_type='svm', C=C)
        m.train(Xtr_s, ytr)
        val_acc_per_c[str(C)] = float(m.evaluate(Xva_s, yva))
        models_per_c[C] = m

    best_C = max(C_GRID, key=lambda c: val_acc_per_c[str(c)])
    best_model = models_per_c[best_C]
    y_test_pred = best_model.pipeline.predict(Xte_s)
    test_acc = float(accuracy_score(yte, y_test_pred))

    np.save(os.path.join(output_dir, f"Xtest_P7_coarse_variant_c_barlow_{subject_id}.npy"), Xte_s)
    np.save(os.path.join(output_dir, f"ytest_P7_coarse_variant_c_barlow_{subject_id}.npy"), yte)

    model_path = os.path.join(output_dir, f"SVM_P7_coarse_variant_c_barlow_{subject_id}.pkl")
    best_model.save_model(model_path)

    return {
        "variant": "C_tuned", "subject_id": subject_id, "feature_group": "barlow",
        "chosen_C": best_C, "val_accuracy_per_C": val_acc_per_c,
        "val_accuracy": val_acc_per_c[str(best_C)], "test_accuracy": test_acc,
        "classes_covered": classes_covered_from_predictions(yte, y_test_pred),
        "model_path": model_path, "scaler_path": scaler_path,
    }


def train_variant_d(cached, extractor, output_dir, subject_id):
    """Varian D: only alternative feature group tested ('all', tied with
    'barlow' at Stage A spot-check). Needs fresh 'all'-feature extraction
    from cached RAW epochs (cache only stores Barlow), but skips the raw
    CSV read."""
    os.makedirs(output_dir, exist_ok=True)
    X_train_3d, ytr = cached["X_train_3d_coarse"], cached["y_train_coarse"]
    X_val_3d, yva = cached["X_val_3d_coarse"], cached["y_val_coarse"]
    X_test_3d, yte = cached["X_test_3d_coarse"], cached["y_test_coarse"]

    Xtr = np.nan_to_num(extractor.transform(X_train_3d, groups=["all"]), nan=0.0, posinf=0.0, neginf=0.0)
    Xva = np.nan_to_num(extractor.transform(X_val_3d, groups=["all"]), nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(extractor.transform(X_test_3d, groups=["all"]), nan=0.0, posinf=0.0, neginf=0.0)

    scaler_path = os.path.join(output_dir, f"scaler_P7_coarse_variant_d_all_{subject_id}.pkl")
    Xtr_s, Xva_s, Xte_s, _ = fit_and_apply_scaler(Xtr, Xva, Xte, save_path=scaler_path)

    np.save(os.path.join(output_dir, f"Xtest_P7_coarse_variant_d_all_{subject_id}.npy"), Xte_s)
    np.save(os.path.join(output_dir, f"ytest_P7_coarse_variant_d_all_{subject_id}.npy"), yte)

    model = ClassicalClassifier(model_type='svm', C=DEFAULT_C)
    model.train(Xtr_s, ytr)
    val_acc = float(model.evaluate(Xva_s, yva))
    y_test_pred = model.pipeline.predict(Xte_s)
    test_acc = float(accuracy_score(yte, y_test_pred))

    model_path = os.path.join(output_dir, f"SVM_P7_coarse_variant_d_all_{subject_id}.pkl")
    model.save_model(model_path)

    return {
        "variant": "D_feat_all", "subject_id": subject_id, "feature_group": "all",
        "val_accuracy": val_acc, "test_accuracy": test_acc,
        "classes_covered": classes_covered_from_predictions(yte, y_test_pred),
        "model_path": model_path, "scaler_path": scaler_path,
    }


def train_variant_e(cached, output_dir, subject_id):
    """Varian E: baseline SVM config (E0, Barlow, unweighted, C=10) wrapped
    in CalibratedClassifierCV(cv=5, method='sigmoid'). Reports both raw
    accuracy and calibration quality (Brier score + simple top-label ECE).

    Assumes every coarse class has enough training samples for 5-fold
    stratified CV (CalibratedClassifierCV raises ValueError otherwise) --
    intentionally not guarded/fallback-wrapped, so such a data-adequacy
    problem surfaces as a clear crash rather than being silently patched
    over, consistent with this pipeline's "never swallow errors" rule
    (see run_p6_transfer_overt_imagined.py's exit-code fix)."""
    os.makedirs(output_dir, exist_ok=True)
    Xtr, ytr = cached["X_train_feat_barlow"], cached["y_train_coarse"]
    Xva, yva = cached["X_val_feat_barlow"], cached["y_val_coarse"]
    Xte, yte = cached["X_test_feat_barlow"], cached["y_test_coarse"]

    scaler_path = os.path.join(output_dir, f"scaler_P7_coarse_variant_e_barlow_{subject_id}.pkl")
    Xtr_s, Xva_s, Xte_s, _ = fit_and_apply_scaler(Xtr, Xva, Xte, save_path=scaler_path)

    base_pipeline = baseline_svc_pipeline(C=DEFAULT_C)
    calibrated = CalibratedClassifierCV(base_pipeline, cv=5, method='sigmoid')
    calibrated.fit(Xtr_s, ytr)

    val_pred = calibrated.predict(Xva_s)
    val_acc = float(accuracy_score(yva, val_pred))

    test_pred = calibrated.predict(Xte_s)
    test_proba = calibrated.predict_proba(Xte_s)
    test_acc = float(accuracy_score(yte, test_pred))
    test_brier = multiclass_brier_score(yte, test_proba, calibrated.classes_)
    test_ece = expected_calibration_error(yte, test_proba, calibrated.classes_)

    np.save(os.path.join(output_dir, f"Xtest_P7_coarse_variant_e_barlow_{subject_id}.npy"), Xte_s)
    np.save(os.path.join(output_dir, f"ytest_P7_coarse_variant_e_barlow_{subject_id}.npy"), yte)

    model_path = os.path.join(output_dir, f"CalibratedSVM_P7_coarse_variant_e_barlow_{subject_id}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(calibrated, f)

    return {
        "variant": "E_calibrated", "subject_id": subject_id, "feature_group": "barlow",
        "val_accuracy": val_acc, "test_accuracy": test_acc,
        "test_brier_score": test_brier, "test_ece": test_ece,
        "classes_covered": classes_covered_from_predictions(yte, test_pred),
        "model_path": model_path, "scaler_path": scaler_path,
    }


def summarize_phase1(per_subject):
    """Mean/std/delta-vs-baseline/Wilcoxon p-value per variant (n=subjects
    with a Phase 1 result), plus a preview of Fase 2's automatic inclusion
    rule (delta > 1.0 pp). The actual inclusion decision (including Varian
    E's always-included-structurally exception) is made by
    run_p7_coarse_combined.py, not here -- this is a preview for the Fase
    1 report only."""
    subjects = sorted(per_subject.keys())
    baseline_accs = [per_subject[s]["baseline"]["test_accuracy"] * 100.0 for s in subjects]

    means = {"baseline": float(np.mean(baseline_accs))}
    stds = {"baseline": float(np.std(baseline_accs, ddof=1)) if len(baseline_accs) > 1 else 0.0}
    deltas_pp = {"baseline": 0.0}
    wilcoxon_p = {"baseline": None}
    per_candidate_pct = {"baseline": baseline_accs}

    for label in VARIANT_ORDER:
        accs = [per_subject[s][label]["test_accuracy"] * 100.0 for s in subjects]
        means[label] = float(np.mean(accs))
        stds[label] = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        deltas_pp[label] = means[label] - means["baseline"]
        per_candidate_pct[label] = accs

        if len(subjects) >= MIN_PAIRS_FOR_WILCOXON:
            diffs = np.array(accs) - np.array(baseline_accs)
            if np.all(diffs == 0):
                wilcoxon_p[label] = None
            else:
                _, p = wilcoxon(accs, baseline_accs)
                wilcoxon_p[label] = float(p)
        else:
            wilcoxon_p[label] = None

    included_preview = {label: bool(deltas_pp[label] > INCLUSION_THRESHOLD_PP) for label in VARIANT_ORDER}
    chosen_c_per_subject = {s: per_subject[s]["C_tuned"]["chosen_C"] for s in subjects}
    calibration_quality = {
        s: {"test_brier_score": per_subject[s]["E_calibrated"]["test_brier_score"],
            "test_ece": per_subject[s]["E_calibrated"]["test_ece"]}
        for s in subjects
    }

    return {
        "n_subjects": len(subjects), "subjects": subjects,
        "means_pct": means, "stds_pct": stds, "deltas_pp_vs_baseline": deltas_pp,
        "wilcoxon_p_vs_baseline": wilcoxon_p, "per_candidate_pct": per_candidate_pct,
        "included_preview_gt_1pp": included_preview,
        "inclusion_threshold_pp": INCLUSION_THRESHOLD_PP,
        "chosen_c_per_subject": chosen_c_per_subject,
        "calibration_quality": calibration_quality,
    }


def write_phase1_report(per_subject, summary):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    lines = []
    lines.append("# P7 Coarse Sub-model Ablation Study")
    lines.append("")
    lines.append(
        "Scope: ONLY the `coarse` sub-model is varied below. `fine_A`/`fine_I`/`fine_E`/`sa_branch` "
        "are untouched throughout -- reused as-is from `P7_CoarseToFine/Fullscale_12Subj/`. Every "
        "variant shares the exact same coarse-filtered `three_way_split(seed=42)` as the existing "
        "coarse baseline (via `p7_coarse_cache.py`'s cached raw split + Barlow features)."
    )
    lines.append("")
    lines.append("## Fase 1 -- Individual Factor Ablation")
    lines.append("")
    lines.append(
        "- **Varian A (balanced):** `WeightedClassicalClassifier` -- `class_weight='balanced'` SVC, "
        "cached Barlow features.\n"
        "- **Varian B (augmented):** plain SVM, training set enriched with E5-recipe augmented copies "
        "(`add_noise=True, noise_factor=0.05, apply_jitter=True, jitter_ms=10`) applied to training "
        "epochs only.\n"
        "- **Varian C (tuned):** per-subject C in {0.1, 1, 10, 100, 1000}, selected by validation "
        "accuracy.\n"
        "- **Varian D (feat=all):** only alternative feature group tested ('all' tied with 'barlow' "
        "at Stage A spot-check; the other three groups were clearly behind).\n"
        "- **Varian E (calibrated):** baseline SVM wrapped in `CalibratedClassifierCV(cv=5, "
        "method='sigmoid')`; reports raw accuracy AND calibration quality (Brier score, ECE) since "
        "its purpose is enabling Fase 3's confidence gating, not raw accuracy alone."
    )
    lines.append("")

    lines.append("### Per-subject coarse test accuracy (%)")
    lines.append("")
    lines.append("| Subject | Baseline | A: balanced | B: augmented | C: tuned | D: feat=all | E: calibrated |")
    lines.append("|---|---|---|---|---|---|---|")
    for subj in summary["subjects"]:
        r = per_subject[subj]
        lines.append(
            f"| {subj} | {r['baseline']['test_accuracy']*100:.2f} | "
            f"{r['A_balanced']['test_accuracy']*100:.2f} | {r['B_augmented']['test_accuracy']*100:.2f} | "
            f"{r['C_tuned']['test_accuracy']*100:.2f} | {r['D_feat_all']['test_accuracy']*100:.2f} | "
            f"{r['E_calibrated']['test_accuracy']*100:.2f} |"
        )
    lines.append("")

    lines.append(f"### Mean test accuracy, delta vs. baseline, Wilcoxon p-value (n={summary['n_subjects']})")
    lines.append("")
    lines.append("| Candidate | Mean (%) | Std (pp) | Delta vs baseline (pp) | Wilcoxon p |")
    lines.append("|---|---|---|---|---|")
    for label in ["baseline"] + VARIANT_ORDER:
        p = summary["wilcoxon_p_vs_baseline"][label]
        p_str = f"{p:.4f}" if p is not None else "n/a"
        lines.append(
            f"| {label} | {summary['means_pct'][label]:.2f} | {summary['stds_pct'][label]:.2f} | "
            f"{summary['deltas_pp_vs_baseline'][label]:+.2f} | {p_str} |"
        )
    lines.append("")

    lines.append("### Chosen C per subject (Varian C)")
    lines.append("")
    lines.append("| Subject | Chosen C | " + " | ".join(f"val@C={c}" for c in C_GRID) + " |")
    lines.append("|---|---|" + "---|" * len(C_GRID))
    for subj in summary["subjects"]:
        vc = per_subject[subj]["C_tuned"]
        val_cells = " | ".join(f"{vc['val_accuracy_per_C'][str(c)]*100:.2f}" for c in C_GRID)
        lines.append(f"| {subj} | {vc['chosen_C']} | {val_cells} |")
    lines.append("")

    lines.append("### Varian E calibration quality")
    lines.append("")
    lines.append("| Subject | Test Brier score (lower better) | Test ECE (lower better) |")
    lines.append("|---|---|---|")
    for subj in summary["subjects"]:
        cq = summary["calibration_quality"][subj]
        lines.append(f"| {subj} | {cq['test_brier_score']:.4f} | {cq['test_ece']:.4f} |")
    lines.append("")

    lines.append(f"### Fase 2 inclusion preview (automatic rule: delta > {INCLUSION_THRESHOLD_PP:.1f} pp)")
    lines.append("")
    lines.append(
        "This is a PREVIEW only -- the actual inclusion decision (and Varian E's exception, see "
        "below) is made by `run_p7_coarse_combined.py` when it runs, using these same numbers."
    )
    lines.append("")
    lines.append("| Variant | Delta vs baseline (pp) | Included by threshold? |")
    lines.append("|---|---|---|")
    for label in VARIANT_ORDER:
        included = summary["included_preview_gt_1pp"][label]
        lines.append(f"| {label} | {summary['deltas_pp_vs_baseline'][label]:+.2f} | {'YES' if included else 'no'} |")
    lines.append("")
    lines.append(
        "**Note on Varian E:** regardless of whether it clears the 1pp threshold above, Varian E "
        "(calibration) is always included STRUCTURALLY in the Fase 2 combined model (wrapped at the "
        "outer layer) -- its purpose is supporting Fase 3's confidence gating, not raising raw "
        "accuracy, so the inclusion threshold does not apply to it the same way. This is intentional, "
        "not a rule violation -- see run_p7_coarse_combined.py's docstring."
    )

    with open(PHASE1_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO][P7-Ablation] Report written to {PHASE1_REPORT_PATH}")


def run_ablation_phase1(subject_ids=None):
    print(f"\n{'=' * 70}\n P7 Coarse Sub-model Ablation -- Fase 1 (Varian A-E)\n{'=' * 70}")

    fullscale_root = p7base.setup_experiment(FULLSCALE_STAGE_DIR, pilar=PILAR)["weights"]
    variant_dirs = {label: os.path.join(fullscale_root, d) for label, d in VARIANT_DIRS.items()}

    if subject_ids is None:
        subject_ids = p7base.discover_subject_ids()

    extractor = EEGFeatureExtractor(fs=p7base.E0_PROCESSOR_PARAMS["target_fs"])
    proc = SignalProcessor(target_fs=p7base.E0_PROCESSOR_PARAMS["target_fs"])

    per_subject = {}
    for subject_id in subject_ids:
        baseline_json = os.path.join(fullscale_root, f"results_{subject_id}.json")
        if not os.path.exists(baseline_json):
            print(f"[WARNING][P7-Ablation] No existing P7 baseline for {subject_id} -- skipping "
                  f"(run `run_p7_coarse_to_fine.py --stage b` first).")
            continue
        with open(baseline_json) as f:
            baseline_result = json.load(f)
        baseline_coarse = baseline_result["sub_models"]["coarse"]

        comparison_json = os.path.join(fullscale_root, f"phase1_comparison_{subject_id}.json")
        if os.path.exists(comparison_json):
            print(f"[SKIP][P7-Ablation] {subject_id} Fase 1 comparison already exists.")
            with open(comparison_json) as f:
                per_subject[subject_id] = json.load(f)
            continue

        print(f"\n[INFO][P7-Ablation] {subject_id}: loading/building coarse cache...")
        cached = get_or_build_cached_coarse(subject_id)
        if cached is None:
            print(f"[WARNING][P7-Ablation] No raw epochs available for {subject_id}; skipping.")
            continue

        result_a = train_variant_a(cached, variant_dirs["A_balanced"], subject_id)
        result_b = train_variant_b(cached, extractor, proc, variant_dirs["B_augmented"], subject_id)
        result_c = train_variant_c(cached, variant_dirs["C_tuned"], subject_id)
        result_d = train_variant_d(cached, extractor, variant_dirs["D_feat_all"], subject_id)
        result_e = train_variant_e(cached, variant_dirs["E_calibrated"], subject_id)

        combined = {
            "subject_id": subject_id,
            "baseline": {
                "val_accuracy": baseline_coarse["val_accuracy"],
                "test_accuracy": baseline_coarse["test_accuracy"],
            },
            "A_balanced": result_a, "B_augmented": result_b, "C_tuned": result_c,
            "D_feat_all": result_d, "E_calibrated": result_e,
        }
        per_subject[subject_id] = combined
        with open(comparison_json, "w") as f:
            json.dump(combined, f, indent=2)

        print(f"[INFO][P7-Ablation] {subject_id}: baseline {baseline_coarse['test_accuracy']*100:.2f}% | "
              f"A {result_a['test_accuracy']*100:.2f}% | B {result_b['test_accuracy']*100:.2f}% | "
              f"C(C={result_c['chosen_C']}) {result_c['test_accuracy']*100:.2f}% | "
              f"D {result_d['test_accuracy']*100:.2f}% | E {result_e['test_accuracy']*100:.2f}%")

    if not per_subject:
        raise RuntimeError("[P7-Ablation] No subjects processed -- nothing to summarize/report.")

    summary = summarize_phase1(per_subject)
    with open(phase1_summary_path(fullscale_root), "w") as f:
        json.dump(summary, f, indent=2)

    write_phase1_report(per_subject, summary)
    return per_subject, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P7 coarse sub-model ablation Fase 1: Varian A (balanced), B (augmented), "
                     "C (C-tuned), D (feat=all), E (calibrated), each vs. the existing coarse baseline."
    )
    parser.add_argument("--subjects", nargs="+", default=None,
                         help="Restrict to specific subject IDs (e.g. --subjects S1 S2). "
                              "Default: all 12 subjects, auto-discovered.")
    args = parser.parse_args()
    run_ablation_phase1(subject_ids=args.subjects)

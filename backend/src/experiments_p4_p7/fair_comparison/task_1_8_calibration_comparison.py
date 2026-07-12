"""
Task 1.8 -- Calibration Comparison (P3 champion vs P6 coarse baseline, S3).

Both ClassicalClassifier SVMs are trained with probability=True (Platt
scaling) unconditionally (see models/classical_models.py), so predict_proba
is available on BOTH the P3 champion and the P6 coarse BASELINE (no
CalibratedClassifierCV needed to get probabilities at all -- that wrapper,
used only in P6's Varian E ablation, adds an EXTRA calibration layer on top
of Platt scaling, it is not a prerequisite for having probabilities).

Brier score and ECE are computed with formulas copied verbatim from
run_p7_coarse_ablation.py's multiclass_brier_score/expected_calibration_error
(n_bins=10, top-label ECE) so the baseline numbers computed here are
directly comparable to the already-reported Varian E numbers (which used
the same formulas) -- Varian E itself is NOT recomputed here, only cited
for context, per ATURAN #0 (no re-running the ablation).
"""
import numpy as np

from _common import (
    P3_CHAMPION_SUBJECT, P3_CHAMPION_EXP, P3_CHAMPION_FEATURE_GROUP, P6_CHAMPION_SUBJECT,
    p3_champion_paths, p6_submodel_paths, check_exists, load_pickle, save_json,
)

# Cited verbatim from backend/reports/P4_P7_Experiments/P7_CoarseAblation_Phase1_report.md
# ("Varian E calibration quality" table) -- NOT recomputed here, S3 row only, for context.
VARIANT_E_S3_CITED = {"test_brier_score": 0.6720, "test_ece": 0.0844,
                       "source": "P7_CoarseAblation_Phase1_report.md, Varian E calibration quality table, S3 row"}


def multiclass_brier_score(y_true, proba, classes):
    """Verbatim copy of run_p7_coarse_ablation.multiclass_brier_score."""
    y_true = np.asarray(y_true)
    class_to_col = {int(c): i for i, c in enumerate(classes)}
    one_hot = np.zeros_like(proba)
    for row, label in enumerate(y_true):
        one_hot[row, class_to_col[int(label)]] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true, proba, classes, n_bins=10):
    """Verbatim copy of run_p7_coarse_ablation.expected_calibration_error."""
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


def evaluate_calibration(model_path, xtest_path, ytest_path, label):
    missing = check_exists(model_path, xtest_path, ytest_path)
    if missing:
        return {"status": "DATA_NOT_AVAILABLE", "missing_paths": missing}

    model = load_pickle(model_path)
    if not hasattr(model, "predict_proba"):
        return {"status": "NO_PREDICT_PROBA", "note": f"{label} model has no predict_proba"}

    X = np.load(xtest_path)
    y = np.load(ytest_path)
    proba = model.predict_proba(X)
    classes = model.classes_

    brier = multiclass_brier_score(y, proba, classes)
    ece = expected_calibration_error(y, proba, classes, n_bins=10)
    test_acc = float(np.mean(np.asarray(classes)[proba.argmax(axis=1)] == y))

    return {
        "status": "OK", "n_test_samples": int(len(y)), "n_classes": int(len(classes)),
        "test_accuracy_from_proba_argmax": test_acc,
        "test_brier_score": brier, "test_ece": ece,
    }


def main():
    p3_model, p3_scaler, p3_xtest, p3_ytest = p3_champion_paths(
        P3_CHAMPION_SUBJECT, P3_CHAMPION_EXP, P3_CHAMPION_FEATURE_GROUP
    )
    p3_result = evaluate_calibration(p3_model, p3_xtest, p3_ytest, "P3 champion")

    p6_model, p6_scaler, p6_xtest, p6_ytest = p6_submodel_paths(P6_CHAMPION_SUBJECT, "coarse")
    p6_result = evaluate_calibration(p6_model, p6_xtest, p6_ytest, "P6 coarse baseline")

    result = {
        "description": (
            "Brier score (multi-class, lower better) and top-label ECE (n_bins=10, lower "
            "better) for the P3 champion (19-class) and the P6 coarse BASELINE (4-class, S3) "
            "-- both use plain Platt-scaled SVC(probability=True), no CalibratedClassifierCV. "
            "Varian E (P6 ablation, WITH CalibratedClassifierCV) is cited for context only, not "
            "recomputed."
        ),
        "p3_champion": {"subject": P3_CHAMPION_SUBJECT, "exp_id": P3_CHAMPION_EXP,
                         "feature_group": P3_CHAMPION_FEATURE_GROUP, **p3_result},
        "p6_coarse_baseline": {"subject": P6_CHAMPION_SUBJECT, "feature_group": "barlow", **p6_result},
        "p6_variant_e_cited_for_context": VARIANT_E_S3_CITED,
    }
    save_json(result, "calibration_comparison_p3_vs_p6.json")

    if p3_result["status"] == "OK":
        print(f"[TASK 1.8] P3 champion: Brier={p3_result['test_brier_score']:.4f} ECE={p3_result['test_ece']:.4f}")
    if p6_result["status"] == "OK":
        print(f"[TASK 1.8] P6 coarse baseline: Brier={p6_result['test_brier_score']:.4f} ECE={p6_result['test_ece']:.4f}")
    print(f"[TASK 1.8] P6 Varian E (cited, not recomputed): Brier={VARIANT_E_S3_CITED['test_brier_score']:.4f} "
          f"ECE={VARIANT_E_S3_CITED['test_ece']:.4f}")
    return result


if __name__ == "__main__":
    main()

"""
Task 1.1 -- P3 First-Syllable-Only Accuracy (9-class subset of the flat
19-class champion classifier), for an apples-to-apples Tier 2 comparison
against P6's first_syllable_e2e (also 9-way).

Read-only: loads the champion P3 model (S3, E5_Data_Augmentation, barlow)
and its already-scaled Xtest/ytest, predicts once on the full 19-class
output space, then scores that prediction restricted to ground-truth rows
whose label is one of the 9 first-syllable classes. No retraining, no
re-splitting -- reuses the exact saved test split.
"""
import numpy as np
from sklearn.metrics import accuracy_score

from _common import (
    P3_CHAMPION_SUBJECT, P3_CHAMPION_EXP, P3_CHAMPION_FEATURE_GROUP,
    SUBMODEL_LABEL_SETS, LABEL_TO_SYLLABLE,
    p3_champion_paths, check_exists, load_pickle, save_json,
)

FIRST_SYLLABLE_CLASSES = sorted(SUBMODEL_LABEL_SETS["coarse"])
SECOND_SYLLABLE_CLASSES = sorted(set(range(19)) - set(FIRST_SYLLABLE_CLASSES))


def main():
    model_p, scaler_p, xtest_p, ytest_p = p3_champion_paths(
        P3_CHAMPION_SUBJECT, P3_CHAMPION_EXP, P3_CHAMPION_FEATURE_GROUP
    )
    missing = check_exists(model_p, xtest_p, ytest_p)
    if missing:
        result = {
            "status": "DATA_NOT_AVAILABLE",
            "missing_paths": missing,
            "note": "P3 champion artefact(s) not found -- per ATURAN #0, no retrain was attempted.",
        }
        save_json(result, "p3_first_syllable_only_accuracy.json")
        print(f"[TASK 1.1] MISSING artefacts: {missing}")
        return result

    print(f"[TASK 1.1] Loading P3 champion: {P3_CHAMPION_SUBJECT}/{P3_CHAMPION_EXP}/{P3_CHAMPION_FEATURE_GROUP}")
    model = load_pickle(model_p)
    X_test = np.load(xtest_p)   # already scaled -- do NOT re-apply scaler_p
    y_test = np.load(ytest_p)

    y_pred_full = model.predict(X_test)  # full 19-way prediction, unrestricted

    mask = np.isin(y_test, FIRST_SYLLABLE_CLASSES)
    y_true_9 = y_test[mask]
    y_pred_9 = y_pred_full[mask]
    n_test_samples = int(len(y_true_9))

    accuracy = float(accuracy_score(y_true_9, y_pred_9)) if n_test_samples else 0.0

    per_class_recall = {}
    n_classes_covered = 0
    for c in FIRST_SYLLABLE_CLASSES:
        idx = y_true_9 == c
        n_c = int(idx.sum())
        correct_c = int((y_pred_9[idx] == c).sum()) if n_c else 0
        recall_c = (correct_c / n_c) if n_c else 0.0
        if recall_c > 0:
            n_classes_covered += 1
        per_class_recall[LABEL_TO_SYLLABLE[c]] = {
            "label_int": int(c), "n": n_c, "correct": correct_c, "recall": recall_c,
        }

    wrong_mask = y_true_9 != y_pred_9
    n_wrong = int(wrong_mask.sum())
    wrong_preds = y_pred_9[wrong_mask]
    n_wrong_guessed_second_syllable = int(np.isin(wrong_preds, SECOND_SYLLABLE_CLASSES).sum())
    n_wrong_guessed_other_first_syllable = n_wrong - n_wrong_guessed_second_syllable

    result = {
        "status": "OK",
        "description": (
            "P3 champion (S3/E5_Data_Augmentation/barlow) evaluated on ONLY the 9 "
            "first-syllable ground-truth classes, for apples-to-apples Tier 2 comparison "
            "with P6's first_syllable_e2e. Model itself is unchanged (still predicts over "
            "all 19 classes); only the evaluation subset (ground truth) is restricted."
        ),
        "subject_id": P3_CHAMPION_SUBJECT,
        "exp_id": P3_CHAMPION_EXP,
        "feature_group": P3_CHAMPION_FEATURE_GROUP,
        "n_test_samples": n_test_samples,
        "n_test_samples_full_19class": int(len(y_test)),
        "accuracy_9way": accuracy,
        "n_classes_covered": n_classes_covered,
        "n_classes_total": 9,
        "per_class_recall": per_class_recall,
        "error_breakdown": {
            "n_wrong_total": n_wrong,
            "n_wrong_guessed_second_syllable_class": n_wrong_guessed_second_syllable,
            "n_wrong_guessed_other_first_syllable_class": n_wrong_guessed_other_first_syllable,
            "pct_wrong_guessed_second_syllable_class": (
                round(100.0 * n_wrong_guessed_second_syllable / n_wrong, 2) if n_wrong else 0.0
            ),
            "note": (
                "The flat 19-class model has no structural constraint limiting it to the 9 "
                "first-syllable classes (unlike P6's coarse stage, which structurally has only "
                "4 possible outputs). A meaningful fraction of P3's errors on this subset can "
                "therefore be 'impossible' guesses toward a second-syllable class -- this is "
                "expected behaviour, not a bug, and is itself an architectural point of "
                "comparison against P6."
            ),
        },
    }
    save_json(result, "p3_first_syllable_only_accuracy.json")
    print(f"[TASK 1.1] accuracy_9way={accuracy*100:.2f}% coverage={n_classes_covered}/9 "
          f"n={n_test_samples} wrong_to_second_syllable={n_wrong_guessed_second_syllable}/{n_wrong}")
    return result


if __name__ == "__main__":
    main()

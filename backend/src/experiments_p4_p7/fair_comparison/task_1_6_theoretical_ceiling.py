"""
Task 1.6 -- Theoretical Ceiling Calculation (P6, all 12 subjects).

Ceiling assumes independence between the coarse and fine stages:
    ceiling = P(coarse correct) * sum_g[ proportion(true_group=g) * P(fine_g correct) ]
with P(fine_O correct) = 1.0 (group O has no fine stage -- a correct coarse
"O" prediction passes straight through to "BO" deterministically).

Group proportions come directly from the saved, already vowel-group-mapped
ytest_P7_coarse_barlow_S{n}.npy (map_labels_to_vowel_group_ids was applied
at save time -- see dataset_builders_ext.py) -- no raw data or model
reload needed, just the saved test-split labels.

actual < ceiling  => stage errors are positively correlated (cases hard for
                     coarse tend to also be hard for fine -- worse than
                     independence predicts).
actual > ceiling  => stage errors are negatively correlated / independence
                     assumption is pessimistic for this subject.
"""
import numpy as np

from _common import (
    ALL_SUBJECTS, P6_CHAMPION_SUBJECT, ID_TO_VOWEL_GROUP,
    p6_submodel_paths, p6_results_json_path, check_exists, load_json, save_json,
)

GROUPS = ["A", "I", "E", "O"]


def compute_for_subject(subject_id):
    results_path = p6_results_json_path(subject_id)
    _, _, coarse_xtest_p, coarse_ytest_p = p6_submodel_paths(subject_id, "coarse")
    missing = check_exists(results_path, coarse_ytest_p)
    if missing:
        return {"status": "DATA_NOT_AVAILABLE", "missing_paths": missing}

    data = load_json(results_path)
    sub_models = data["sub_models"]
    coarse_acc = sub_models["coarse"]["test_accuracy"]
    fine_acc = {
        "A": sub_models["fine_A"]["test_accuracy"],
        "I": sub_models["fine_I"]["test_accuracy"],
        "E": sub_models["fine_E"]["test_accuracy"],
        "O": 1.0,  # pass-through, no fine stage
    }

    y_group = np.load(coarse_ytest_p)  # vowel-group ids, 0=A,1=I,2=E,3=O
    n_total = len(y_group)
    group_ids, counts = np.unique(y_group, return_counts=True)
    proportions = {ID_TO_VOWEL_GROUP[int(gid)]: int(c) / n_total for gid, c in zip(group_ids, counts)}
    for g in GROUPS:
        proportions.setdefault(g, 0.0)

    expected_fine_given_coarse_correct = sum(proportions[g] * fine_acc[g] for g in GROUPS)
    ceiling = coarse_acc * expected_fine_given_coarse_correct

    actual = data["first_syllable_e2e"]["accuracy"]
    delta = actual - ceiling

    if abs(delta) < 1e-9:
        interpretation = "actual == ceiling (stage errors behave as independent)"
    elif delta < 0:
        interpretation = "actual < ceiling: stage errors positively correlated (compounding -- cases hard for coarse tend to also be hard for fine)"
    else:
        interpretation = "actual > ceiling: stage errors negatively correlated (independence assumption pessimistic here)"

    return {
        "status": "OK",
        "n_test_samples": n_total,
        "group_proportions": proportions,
        "p_coarse_correct": coarse_acc,
        "p_fine_correct_by_group": fine_acc,
        "expected_fine_accuracy_weighted": expected_fine_given_coarse_correct,
        "theoretical_ceiling": ceiling,
        "actual_first_syllable_e2e": actual,
        "delta_actual_minus_ceiling": delta,
        "interpretation": interpretation,
    }


def main():
    per_subject = {subj: compute_for_subject(subj) for subj in ALL_SUBJECTS}

    ok = {s: v for s, v in per_subject.items() if v["status"] == "OK"}
    summary_table = [
        {
            "subject": s,
            "ceiling_pct": round(v["theoretical_ceiling"] * 100, 2),
            "actual_pct": round(v["actual_first_syllable_e2e"] * 100, 2),
            "delta_pp": round(v["delta_actual_minus_ceiling"] * 100, 2),
        }
        for s, v in ok.items()
    ]

    result = {
        "description": (
            "Theoretical first-syllable e2e ceiling under a coarse/fine independence "
            "assumption, vs actual measured accuracy, per subject."
        ),
        "champion_subject": P6_CHAMPION_SUBJECT,
        "per_subject": per_subject,
        "summary_table": summary_table,
    }
    if ok:
        mean_delta_pp = float(np.mean([v["delta_actual_minus_ceiling"] for v in ok.values()])) * 100
        result["mean_delta_pp_across_subjects"] = mean_delta_pp
        n_negative = sum(1 for v in ok.values() if v["delta_actual_minus_ceiling"] < 0)
        result["n_subjects_actual_below_ceiling"] = n_negative
        result["n_subjects_total"] = len(ok)

    save_json(result, "p6_theoretical_ceiling_analysis.json")

    champ = per_subject.get(P6_CHAMPION_SUBJECT, {})
    if champ.get("status") == "OK":
        print(f"[TASK 1.6] {P6_CHAMPION_SUBJECT}: ceiling={champ['theoretical_ceiling']*100:.2f}% "
              f"actual={champ['actual_first_syllable_e2e']*100:.2f}% "
              f"delta={champ['delta_actual_minus_ceiling']*100:+.2f}pp")
        print(f"[TASK 1.6]   {champ['interpretation']}")
    if ok:
        print(f"[TASK 1.6] Mean delta across {len(ok)} subjects: {result['mean_delta_pp_across_subjects']:+.2f}pp "
              f"({result['n_subjects_actual_below_ceiling']}/{result['n_subjects_total']} subjects below ceiling)")
    return result


if __name__ == "__main__":
    main()

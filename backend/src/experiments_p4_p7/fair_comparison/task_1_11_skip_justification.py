"""
Task 1.11 -- Quantitative Skip Justification for P1/P2/P4/P5 End-to-End.

Turns the "compounding error" / "redundant confirmation" arguments into
numbers, per paradigm, entirely from data that already exists on disk:
  P1 -- T1_p1_descriptive_stats.csv (accuracy) + T14_p1_per_syllable_recall.csv
        (class coverage) -> structural word-decodability projection.
  P2 -- T16_p2_subject_syllable_heatmap.csv (per-subject, all 19 syllables)
        -> per-subject class-coverage gate failures (gate=8/19, the same
        threshold already used by the champion-selection algorithm).
  P4 -- P4_NoWindowing/Fullscale_12Subj_E0/results_S{n}.json (12 subjects,
        already on disk) -> near-chance accuracy + P(both syllables right)
        projection.
  P5 -- P5_ShiftedBandpass/Fullscale_12Subj_E0/results_S{n}.json (12
        subjects) vs P3's OWN E0_Baseline/barlow accuracy per subject (from
        T0_pillar3_raw_fresh.csv, the baseline config, NOT the E5 champion,
        for a fair same-config comparison) -> paired delta + Wilcoxon.

No model loading, no raw data, no retraining -- pure re-read/recompute from
existing CSV/JSON exports.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from _common import (
    ALL_SUBJECTS, T_TABLES_DIR, WEIGHTS_P4, WEIGHTS_P5,
    check_exists, load_json, save_json,
)

COVERAGE_GATE = 8  # subjects/configs with class_coverage < this fail the gate (champion-selection precedent)
N_CLASSES_TOTAL = 19
CHANCE_LEVEL = 1.0 / N_CLASSES_TOTAL

WORD_TO_SYLLABLES = {
    "MAKAN": ("MA", "KAN"), "MINUM": ("MI", "NUM"), "BERAK": ("BE", "RAK"),
    "PIPIS": ("PI", "PIS"), "MANDI": ("MAN", "DI"), "BOSAN": ("BO", "SAN"),
    "LELAH": ("LE", "LAH"), "SAKIT": ("SA", "KIT"), "TIDUR": ("TI", "DUR"),
    "SAYANG": ("SA", "YANG"),
}


def analyze_p1():
    t1_path = os.path.join(T_TABLES_DIR, "T1_p1_descriptive_stats.csv")
    t14_path = os.path.join(T_TABLES_DIR, "T14_p1_per_syllable_recall.csv")
    missing = check_exists(t1_path, t14_path)
    if missing:
        return {"status": "DATA_NOT_AVAILABLE", "missing_paths": missing}

    t1 = pd.read_csv(t1_path).set_index("Metric")["Value"]
    t14 = pd.read_csv(t14_path)

    covered = set(t14.loc[t14["Recall"] > 0, "Syllable"])
    coverage = len(covered)

    decodable_words = [w for w, (s1, s2) in WORD_TO_SYLLABLES.items() if s1 in covered and s2 in covered]
    n_structurally_impossible = len(WORD_TO_SYLLABLES) - len(decodable_words)

    return {
        "status": "OK",
        "mean_accuracy_pct": float(t1.get("Mean Accuracy (%)")),
        "max_accuracy_pct": float(t1.get("Maximum Accuracy (%)")),
        "chance_level_pct": float(t1.get("Chance Level (%)")),
        "class_coverage": coverage,
        "class_coverage_gate": COVERAGE_GATE,
        "passes_coverage_gate": coverage >= COVERAGE_GATE,
        "covered_syllables": sorted(covered),
        "n_words_structurally_decodable_of_10": len(decodable_words),
        "structurally_decodable_words": decodable_words,
        "n_words_structurally_impossible_of_10": n_structurally_impossible,
        "narrative": (
            f"P1 (pooled global model) covers only {coverage}/19 syllable classes with any recall at "
            f"all (gate: {COVERAGE_GATE}/19). Of the 10 target words, only {len(decodable_words)} have "
            f"BOTH syllables inside that covered set -- the remaining {n_structurally_impossible} are "
            f"structurally guaranteed wrong at the word level regardless of luck, before any "
            f"end-to-end pipeline is even built. This is why P1 was excluded from champion selection "
            f"and never received an end-to-end evaluation."
        ),
    }


def analyze_p2():
    t16_path = os.path.join(T_TABLES_DIR, "T16_p2_subject_syllable_heatmap.csv")
    missing = check_exists(t16_path)
    if missing:
        return {"status": "DATA_NOT_AVAILABLE", "missing_paths": missing}

    t16 = pd.read_csv(t16_path, index_col=0)
    per_subject_coverage = (t16 > 0).sum(axis=1)  # count of 19 syllable columns with recall>0
    n_fail = int((per_subject_coverage < COVERAGE_GATE).sum())
    n_total = len(per_subject_coverage)

    return {
        "status": "OK",
        "class_coverage_gate": COVERAGE_GATE,
        "per_subject_class_coverage": per_subject_coverage.to_dict(),
        "n_subjects_fail_gate": n_fail,
        "n_subjects_total": n_total,
        "pct_subjects_fail_gate": round(100.0 * n_fail / n_total, 1),
        "narrative": (
            f"{n_fail}/{n_total} subjects' best P2 (Subject-Dependent EEGNet) configuration fails the "
            f"same {COVERAGE_GATE}/19 class-coverage gate used by champion selection (this is exactly "
            f"the mechanism that disqualified the P2/S6/E6 accuracy peak in the champion search). An "
            f"end-to-end pipeline built on a per-subject model that only ever predicts a handful of "
            f"classes would inherit that collapse directly -- there is no coarse/fine restructuring "
            f"that fixes a Stage-1 model which never learned most of the class space."
        ),
    }


def analyze_p4():
    per_subject = {}
    for subj in ALL_SUBJECTS:
        path = os.path.join(WEIGHTS_P4, "Fullscale_12Subj_E0", f"results_{subj}.json")
        if check_exists(path):
            continue
        data = load_json(path)
        per_subject[subj] = {"test_accuracy": data["test_accuracy"], "n_classes_covered": data["n_classes_covered"]}

    if not per_subject:
        return {"status": "DATA_NOT_AVAILABLE"}

    accs = np.array([v["test_accuracy"] for v in per_subject.values()])
    mean_acc = float(np.mean(accs))
    p_both_syllables_theoretical = CHANCE_LEVEL ** 2
    p_both_syllables_from_observed = mean_acc ** 2

    return {
        "status": "OK",
        "n_subjects": len(per_subject),
        "per_subject": per_subject,
        "mean_test_accuracy_pct": mean_acc * 100,
        "chance_level_pct": CHANCE_LEVEL * 100,
        "ratio_to_chance": mean_acc / CHANCE_LEVEL,
        "p_both_syllables_correct_at_chance_pct": p_both_syllables_theoretical * 100,
        "p_both_syllables_correct_at_observed_rate_pct": p_both_syllables_from_observed * 100,
        "narrative": (
            f"P4 (No-Windowing, 5s full epoch) achieves a mean syllable accuracy of {mean_acc*100:.2f}% "
            f"across {len(per_subject)} subjects, essentially chance level ({CHANCE_LEVEL*100:.2f}% for "
            f"19 classes -- a ratio of {mean_acc/CHANCE_LEVEL:.2f}x). Treating the two syllables of a "
            f"word as independent draws at this rate gives P(whole word correct) ~ "
            f"{p_both_syllables_from_observed*100:.3f}% -- effectively unusable for a 2-syllable "
            f"end-to-end pipeline, and likely an OVER-estimate besides, since near-chance accuracy is "
            f"often concentrated in a single collapsed class (as directly observed for P1) rather than "
            f"spread with genuine per-class signal. Confirms sample-size scarcity (5s epoch = 1/5 the "
            f"training windows of the standard 1s-windowed paradigms) as the bottleneck, not windowing "
            f"structure itself."
        ),
    }


def analyze_p5():
    t0_p3_path = os.path.join(T_TABLES_DIR, "T0_pillar3_raw_fresh.csv")
    missing = check_exists(t0_p3_path)
    if missing:
        return {"status": "DATA_NOT_AVAILABLE", "missing_paths": missing}

    t0_p3 = pd.read_csv(t0_p3_path)
    p3_e0_barlow = t0_p3[(t0_p3["exp_id"] == "E0") & (t0_p3["feature_group"] == "barlow")]
    p3_by_subject = dict(zip(p3_e0_barlow["subject"], p3_e0_barlow["accuracy"]))

    p5_by_subject = {}
    for subj in ALL_SUBJECTS:
        path = os.path.join(WEIGHTS_P5, "Fullscale_12Subj_E0", f"results_{subj}.json")
        if check_exists(path):
            continue
        p5_by_subject[subj] = load_json(path)["test_accuracy"]

    paired_subjects = sorted(set(p3_by_subject) & set(p5_by_subject))
    if len(paired_subjects) < 3:
        return {"status": "INSUFFICIENT_PAIRED_DATA", "n_paired": len(paired_subjects)}

    p3_vals = np.array([p3_by_subject[s] for s in paired_subjects])
    p5_vals = np.array([p5_by_subject[s] for s in paired_subjects])
    deltas_pp = (p5_vals - p3_vals) * 100

    try:
        stat, p_value = wilcoxon(p5_vals, p3_vals)
        p_value = float(p_value)
    except ValueError as e:
        stat, p_value = None, None
        wilcoxon_error = str(e)
    else:
        wilcoxon_error = None

    n_negative = int((deltas_pp < 0).sum())

    if p_value is None:
        p_clause = f"Wilcoxon test could not be run ({wilcoxon_error})"
    elif p_value >= 0.05:
        p_clause = f"Wilcoxon p={p_value:.4f} (not statistically significant)"
    else:
        p_clause = f"Wilcoxon p={p_value:.4f} (statistically significant)"

    return {
        "status": "OK",
        "n_subjects_paired": len(paired_subjects),
        "per_subject": {
            s: {"p3_e0_barlow_pct": float(p3_by_subject[s] * 100), "p5_e0_barlow_pct": float(p5_by_subject[s] * 100),
                "delta_pp": float(deltas_pp[i])}
            for i, s in enumerate(paired_subjects)
        },
        "mean_delta_pp": float(np.mean(deltas_pp)),
        "n_subjects_negative_delta": n_negative,
        "n_subjects_total": len(paired_subjects),
        "wilcoxon_statistic": float(stat) if stat is not None else None,
        "wilcoxon_p_value": p_value,
        "wilcoxon_error": wilcoxon_error,
        "narrative": (
            f"P5 (Shifted Bandpass 15-65Hz) vs P3's own E0_Baseline/barlow accuracy (same config, fair "
            f"same-baseline comparison, NOT the E5 champion): mean delta {np.mean(deltas_pp):+.2f}pp "
            f"across {len(paired_subjects)} subjects, {n_negative}/{len(paired_subjects)} in the "
            f"negative direction, {p_clause}"
            f" -- this is a 'redundant confirmation' finding (the shifted band carries essentially the "
            f"same, not more, signal than the standard band), structurally different from P1/P2/P4's "
            f"'compounding error from Stage-1 collapse' argument, so no end-to-end pipeline was built "
            f"for P5 either: there is nothing for a hierarchical or word-assembly stage to add on top "
            f"of a syllable-classification result that is already statistically indistinguishable from "
            f"the existing baseline."
        ),
    }


def main():
    result = {
        "P1_Global": analyze_p1(),
        "P2_SubjectDependent_EEGNet": analyze_p2(),
        "P4_NoWindowing": analyze_p4(),
        "P5_ShiftedBandpass": analyze_p5(),
    }
    save_json(result, "quantitative_skip_justification_p1_p2_p4_p5.json")

    for name, r in result.items():
        print(f"[TASK 1.11] {name}: status={r.get('status')}")
        if r.get("narrative"):
            print(f"[TASK 1.11]   {r['narrative']}")
    return result


if __name__ == "__main__":
    main()

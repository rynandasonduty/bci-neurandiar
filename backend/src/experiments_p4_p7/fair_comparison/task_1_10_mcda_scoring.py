"""
Task 1.10 -- MCDA (Multi-Criteria Decision Analysis) Table, P3 vs P6.

Pulls together the outputs of Tasks 1.1, 1.2, 1.4, and 1.8 (must already
have been run -- this script only reads their saved JSON, it does not
recompute anything). It does NOT declare a winner; the champion decision
itself remains the researcher's call.

Weights CONFIRMED by the researcher at the Fase 1 checkpoint (2026-07-12):
accuracy criteria (Tier 2 + Tier 3) raised to 60 total (30 each). The
remaining 4 criteria were scaled down proportionally from the original
proposal (20/10/15/5, summing to 50) to sum to the remaining 40, preserving
their relative emphasis: class_coverage 20*0.8=16, system_complexity
10*0.8=8, latency 15*0.8=12, calibration_quality 5*0.8=4.

Normalization rule (applied uniformly to all 6 criteria, standard
"relative-to-best" linear MCDA normalization): for each criterion, the
better-performing paradigm scores 100; the other scores
100 * (their_value / best_value) for higher-is-better criteria, or
100 * (best_value / their_value) for lower-is-better criteria (latency,
system complexity). This avoids criteria that happen to already live on a
0-100 scale (e.g. coverage %) silently dominating raw-percentage criteria
like Tier 2/3 accuracy (~10-30% range) in the weighted sum.
"""
import os

import pandas as pd

from _common import (
    P3_CHAMPION_SUBJECT, P6_CHAMPION_SUBJECT, RESULTS_DIR, T_TABLES_DIR,
    p6_results_json_path, check_exists, load_json, save_json,
)

# CONFIRMED weights (sum=100) -- researcher decision, 2026-07-12. See docstring.
PROPOSED_WEIGHTS = {
    "tier2_first_syllable_accuracy": 30,
    "tier3_full_word_accuracy": 30,
    "class_coverage": 16,
    "system_complexity": 8,
    "latency": 12,
    "calibration_quality": 4,
}


def relative_score(p3_value, p6_value, higher_is_better):
    """Better performer -> 100; the other -> 100 * ratio. Returns (p3_score, p6_score)."""
    if p3_value is None or p6_value is None:
        return None, None
    if higher_is_better:
        best = max(p3_value, p6_value)
        if best == 0:
            return 0.0, 0.0
        return 100.0 * p3_value / best, 100.0 * p6_value / best
    else:
        best = min(p3_value, p6_value)
        if p3_value == 0 or p6_value == 0:
            return (100.0 if p3_value == best else 0.0), (100.0 if p6_value == best else 0.0)
        return 100.0 * best / p3_value, 100.0 * best / p6_value


def load_dependency(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    return load_json(path)


def main():
    task_1_1 = load_dependency("p3_first_syllable_only_accuracy.json")
    task_1_4 = load_dependency("p6_latency_measurement.json")
    task_1_8 = load_dependency("calibration_comparison_p3_vs_p6.json")

    p6_results_path = p6_results_json_path(P6_CHAMPION_SUBJECT)
    p6_results = load_json(p6_results_path) if not check_exists(p6_results_path) else None

    missing_deps = [
        name for name, d in [("Task 1.1", task_1_1), ("Task 1.4", task_1_4), ("Task 1.8", task_1_8),
                              ("P6 results_S3.json", p6_results)]
        if d is None
    ]

    t5_path = os.path.join(T_TABLES_DIR, "T5_champion_candidate_coverage.csv")
    p3_coverage_pct = None
    if os.path.exists(t5_path):
        t5 = pd.read_csv(t5_path)
        row = t5[(t5["pilar"] == "P3") & (t5["subject"] == P3_CHAMPION_SUBJECT) & (t5["feature_group"] == "barlow")]
        row = row[row["exp_id"].astype(str).str.contains("E5")]
        if not row.empty:
            p3_coverage_pct = float(row.iloc[0]["class_coverage"]) / 19.0 * 100.0

    # ---- raw values per criterion ----
    raw = {}
    raw["tier2_first_syllable_accuracy"] = {
        "P3": task_1_1["accuracy_9way"] * 100 if task_1_1 and task_1_1.get("status") == "OK" else None,
        "P6": p6_results["first_syllable_e2e"]["accuracy"] * 100 if p6_results else None,
        "unit": "%", "higher_is_better": True,
    }
    raw["tier3_full_word_accuracy"] = {
        "P3": 11.11,  # ATURAN #0 point 3, S3, single data point (train_word_assembler_s3.py) -- not recomputed here
        "P3_note": "Cited from existing train_word_assembler_s3.py result, S3's only P3 full-word data point -- not recomputed in Fase 1.",
        "P6": p6_results["full_word_e2e"]["accuracy"] * 100 if (p6_results and p6_results["full_word_e2e"].get("available")) else None,
        "unit": "%", "higher_is_better": True,
    }
    raw["class_coverage"] = {
        "P3": p3_coverage_pct, "P3_raw": "18/19" if p3_coverage_pct else None,
        "P6": (sum(1 for v in p6_results["first_syllable_e2e"]["per_syllable_recall"].values() if v["correct"] > 0)
               / 9.0 * 100.0) if p6_results else None,
        "P6_raw": "9/9" if p6_results else None,
        "unit": "% of own class space", "higher_is_better": True,
    }
    raw["system_complexity"] = {
        "P3": 1, "P6": 5, "unit": "number of models in the inference path", "higher_is_better": False,
    }
    p6_latency = None
    if task_1_4:
        we = task_1_4.get("weighted_expected_latency")
        if we:
            p6_latency = we["expected_mean_ms"]
        else:
            sc = task_1_4.get("p6_cascade_latency_measured", {}).get("scenarios", {})
            if sc.get("non_sa_case", {}).get("available"):
                p6_latency = sc["non_sa_case"]["mean_ms"]
    # IMPORTANT: use the FAIR, same-basis P3 latency (raw epoch -> feature extraction ->
    # scale -> predict), NOT p3_champion_latency_cited (T9), which times ONLY
    # scaler.transform+predict on an already feature-extracted vector and so excludes
    # Barlow feature extraction entirely -- using it here would score P3 as if it pays
    # zero feature-extraction cost while P6 pays it 2-3x, a ~1300x-skewed, misleading
    # criterion score (see Task 1.4's fair_comparison_same_basis for the diagnosis).
    p3_latency_fair = None
    if task_1_4 and task_1_4.get("p3_fair_with_extraction", {}).get("available"):
        p3_latency_fair = task_1_4["p3_fair_with_extraction"]["mean_ms"]
    raw["latency"] = {
        "P3": p3_latency_fair, "P6": p6_latency, "unit": "ms (mean, same-basis w/ feature extraction)",
        "higher_is_better": False,
        "note": "P3 value is the fair same-basis figure (Task 1.4 p3_fair_with_extraction), not the T9-cited 0.447ms (which excludes feature extraction).",
    }
    raw["calibration_quality"] = {
        "P3": task_1_8["p3_champion"]["test_ece"] if task_1_8 and task_1_8["p3_champion"].get("status") == "OK" else None,
        "P6": task_1_8["p6_coarse_baseline"]["test_ece"] if task_1_8 and task_1_8["p6_coarse_baseline"].get("status") == "OK" else None,
        "unit": "ECE (top-label, n_bins=10)", "higher_is_better": False,
    }

    # ---- scores ----
    criteria_table = []
    total_p3, total_p6, total_weight_used = 0.0, 0.0, 0.0
    for crit, weight in PROPOSED_WEIGHTS.items():
        r = raw[crit]
        p3_score, p6_score = relative_score(r["P3"], r["P6"], r["higher_is_better"])
        row = {
            "criterion": crit, "weight": weight,
            "p3_raw_value": r["P3"], "p6_raw_value": r["P6"], "unit": r["unit"],
            "higher_is_better": r["higher_is_better"],
            "p3_score_0_100": round(p3_score, 2) if p3_score is not None else None,
            "p6_score_0_100": round(p6_score, 2) if p6_score is not None else None,
        }
        criteria_table.append(row)
        if p3_score is not None and p6_score is not None:
            total_p3 += weight * p3_score
            total_p6 += weight * p6_score
            total_weight_used += weight

    final_p3 = total_p3 / total_weight_used if total_weight_used else None
    final_p6 = total_p6 / total_weight_used if total_weight_used else None

    result = {
        "status": "INCOMPLETE_DEPENDENCIES" if missing_deps else "OK",
        "missing_dependencies": missing_deps,
        "description": (
            "MCDA scoring table, P3 champion vs P6 champion (S3 for both). Weights CONFIRMED by "
            "the researcher 2026-07-12 -- see PROPOSED_WEIGHTS and docstring. No champion decision "
            "is made here; that remains the researcher's call."
        ),
        "confirmed_weights": PROPOSED_WEIGHTS,
        "total_weight_used_pct_of_100": total_weight_used,
        "criteria_table": criteria_table,
        "final_weighted_score_P3": round(final_p3, 2) if final_p3 is not None else None,
        "final_weighted_score_P6": round(final_p6, 2) if final_p6 is not None else None,
        "note": (
            "final_weighted_score is provisional and only sums the criteria with complete data "
            "(total_weight_used_pct_of_100 shows how much of the 100pp weight budget was actually "
            "scored) -- if total_weight_used < 100, some criteria are missing upstream data and the "
            "two scores are not yet fully comparable."
        ),
    }
    save_json(result, "mcda_scoring_table_p3_vs_p6.json")

    md_lines = ["| Criterion | Weight | P3 raw | P6 raw | P3 score | P6 score |",
                "|---|---|---|---|---|---|"]
    for row in criteria_table:
        md_lines.append(
            f"| {row['criterion']} | {row['weight']} | {row['p3_raw_value']} | {row['p6_raw_value']} | "
            f"{row['p3_score_0_100']} | {row['p6_score_0_100']} |"
        )
    md_lines.append(f"| **Weighted total** | 100 | | | **{result['final_weighted_score_P3']}** | **{result['final_weighted_score_P6']}** |")
    md_table = "\n".join(md_lines)
    with open(os.path.join(RESULTS_DIR, "mcda_scoring_table_p3_vs_p6.md"), "w") as f:
        f.write(md_table + "\n")
    print(f"[SAVED] {os.path.join(RESULTS_DIR, 'mcda_scoring_table_p3_vs_p6.md')}")

    if missing_deps:
        print(f"[TASK 1.10] INCOMPLETE -- missing dependency outputs: {missing_deps}")
    print(md_table)
    print(f"[TASK 1.10] Weight budget actually scored: {total_weight_used}/100")
    return result


if __name__ == "__main__":
    main()

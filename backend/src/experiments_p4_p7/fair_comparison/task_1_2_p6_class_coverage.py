"""
Task 1.2 -- P6 Class Coverage Check (S3 + all 12 subjects).

Per ATURAN #0 / the task's own "LANGKAH PERTAMA": results_S{n}.json under
P7_CoarseToFine/Fullscale_12Subj/ already contains everything needed
(sub_models.*.n_classes_covered/.classes_covered, first_syllable_e2e.
per_syllable_recall). This script ONLY reads and reformats those existing
JSON files -- it does not load any model or re-run any prediction.
"""
from _common import (
    ALL_SUBJECTS, SUBMODEL_NAMES, P6_CHAMPION_SUBJECT,
    p6_results_json_path, check_exists, load_json, save_json,
)

N_CLASSES_TOTAL = {"coarse": 4, "fine_A": 3, "fine_I": 3, "fine_E": 2, "sa_branch": 2}
COVERAGE_GATE = {"coarse": 2, "fine_A": 2, "fine_I": 2, "fine_E": 1, "sa_branch": 1}  # informational only


def main():
    per_subject = {}
    missing_subjects = []

    for subj in ALL_SUBJECTS:
        path = p6_results_json_path(subj)
        missing = check_exists(path)
        if missing:
            missing_subjects.append(subj)
            per_subject[subj] = {"status": "DATA_NOT_AVAILABLE", "missing_paths": missing}
            continue

        data = load_json(path)
        sub_models = data.get("sub_models", {})
        sm_summary = {}
        for name in SUBMODEL_NAMES:
            sm = sub_models.get(name)
            if sm is None:
                sm_summary[name] = {"status": "MISSING_IN_JSON"}
                continue
            sm_summary[name] = {
                "n_classes_covered": sm["n_classes_covered"],
                "n_classes_total": N_CLASSES_TOTAL[name],
                "classes_covered": sm["classes_covered"],
                "test_accuracy": sm["test_accuracy"],
                "n_test": sm["n_test"],
            }

        fse = data.get("first_syllable_e2e", {})
        per_syl = fse.get("per_syllable_recall", {})
        syllables_with_correct = sorted([s for s, v in per_syl.items() if v.get("correct", 0) > 0])
        pipeline_coverage = {
            "n_syllables_covered": len(syllables_with_correct),
            "n_syllables_total": 9,
            "syllables_covered": syllables_with_correct,
            "syllables_never_correct": sorted(set(per_syl.keys()) - set(syllables_with_correct)),
            "first_syllable_e2e_accuracy": fse.get("accuracy"),
            "n_test_samples": fse.get("n_test_samples"),
        }

        per_subject[subj] = {
            "status": "OK",
            "winning_coarse_feature_group": data.get("winning_coarse_feature_group"),
            "sub_models": sm_summary,
            "first_syllable_pipeline_coverage": pipeline_coverage,
        }

    coverage_json = {
        "description": (
            "Class coverage per P6 sub-model (coarse/fine_A/fine_I/fine_E/sa_branch) and "
            "first-syllable pipeline coverage, read directly from the existing Stage B "
            "results_S{n}.json files -- no model reload, no re-prediction."
        ),
        "champion_subject": P6_CHAMPION_SUBJECT,
        "missing_subjects": missing_subjects,
        "per_subject": per_subject,
    }
    save_json(coverage_json, "p6_class_coverage_all_submodels.json")

    pipeline_only = {
        subj: v.get("first_syllable_pipeline_coverage")
        for subj, v in per_subject.items() if v.get("status") == "OK"
    }
    save_json(pipeline_only, "p6_first_syllable_pipeline_coverage.json")

    champ = per_subject.get(P6_CHAMPION_SUBJECT, {})
    if champ.get("status") == "OK":
        cov = champ["first_syllable_pipeline_coverage"]
        print(f"[TASK 1.2] {P6_CHAMPION_SUBJECT} pipeline coverage: "
              f"{cov['n_syllables_covered']}/9 syllables ever correct "
              f"(never correct: {cov['syllables_never_correct']})")
        for name in SUBMODEL_NAMES:
            sm = champ["sub_models"][name]
            print(f"[TASK 1.2]   {name}: {sm['n_classes_covered']}/{sm['n_classes_total']} classes covered")
    if missing_subjects:
        print(f"[TASK 1.2] WARNING missing subjects: {missing_subjects}")
    else:
        print("[TASK 1.2] All 12 subjects present and OK.")
    return coverage_json


if __name__ == "__main__":
    main()

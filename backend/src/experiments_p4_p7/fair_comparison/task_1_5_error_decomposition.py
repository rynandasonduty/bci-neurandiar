"""
Task 1.5 -- Error Decomposition / Propagation Analysis (P6, S3).

Reconstructs the IDENTICAL trial-level 80/20 test split that
compute_full_word_e2e_accuracy uses (test_size=0.2, random_state=42,
stratified by word when possible -- OfflineTrialReader-based, memory-safe),
but records EVERY stage's prediction (coarse, fine, sa_branch) against
ground truth instead of only the final word, then buckets every wrong
trial into exactly one failure category:
  - coarse_wrong               : predicted vowel group != true vowel group
  - fine_wrong_coarse_correct  : coarse right, fine stage (A/I/E) got it wrong
  - sa_branch_wrong            : first syllable correctly identified as SA,
                                  but KIT/YANG disambiguation failed
  - correct_all                : whole word correct (sanity-check bucket)
Isolated in its own subprocess (raw CSV via OfflineTrialReader), consistent
with Tasks 1.3/1.4.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from sklearn.model_selection import train_test_split

from _common import (
    P6_CHAMPION_SUBJECT, RESULTS_DIR, RAW_DATA_DIR,
    p6_load_bundles, save_json,
)

THIS_FILE = os.path.abspath(__file__)
WORD_TO_SYLLABLES = {
    "MAKAN": ("MA", "KAN"), "MINUM": ("MI", "NUM"), "BERAK": ("BE", "RAK"),
    "PIPIS": ("PI", "PIS"), "MANDI": ("MAN", "DI"), "BOSAN": ("BO", "SAN"),
    "LELAH": ("LE", "LAH"), "SAKIT": ("SA", "KIT"), "TIDUR": ("TI", "DUR"),
    "SAYANG": ("SA", "YANG"),
}
SPLIT_RANDOM_STATE = 42


def worker_decompose(subject_id, out_path):
    from experiments_p4_p7.run_p7_coarse_to_fine import E0_PROCESSOR_PARAMS
    from experiments_p4_p7.dataset_builders_ext import (
        ID_TO_VOWEL_GROUP, VOWEL_GROUP_OF_LABEL, LABEL_TO_SYLLABLE,
        DETERMINISTIC_FIRST_SYLLABLE_TO_WORD, SA_BRANCH_SECOND_SYLLABLE_TO_WORD,
    )
    from pipeline.offline_trial_reader import OfflineTrialReader

    syllable_to_label = {v: k for k, v in LABEL_TO_SYLLABLE.items()}
    coarse_bundle, fine_bundles, sa_bundle = p6_load_bundles(subject_id)
    reader = OfflineTrialReader(RAW_DATA_DIR, E0_PROCESSOR_PARAMS)
    trials_meta = reader.list_valid_trials(subject_id)

    valid_trials = []
    for idx, meta in enumerate(trials_meta):
        word = meta["word"].strip().upper()
        if word not in WORD_TO_SYLLABLES:
            continue
        try:
            valid_trials.append(reader.read_trial(subject_id, trial_index=idx))
        except ValueError:
            continue

    words = [t["word"].strip().upper() for t in valid_trials]
    _, counts = np.unique(words, return_counts=True)
    can_stratify = len(set(words)) > 1 and min(counts) >= 2
    _, test_trials = train_test_split(
        valid_trials, test_size=0.2, random_state=SPLIT_RANDOM_STATE,
        stratify=words if can_stratify else None,
    )

    records = []
    for trial in test_trials:
        word_true = trial["word"].strip().upper()
        syl1_true, syl2_true = WORD_TO_SYLLABLES[word_true]
        group_true = VOWEL_GROUP_OF_LABEL[syllable_to_label[syl1_true]]

        coarse_pred_id = coarse_bundle.predict_single(trial["epoch_slot1"])
        coarse_pred_group = ID_TO_VOWEL_GROUP[coarse_pred_id]
        coarse_correct = (coarse_pred_group == group_true)

        if coarse_pred_group == "O":
            first_syl_pred = "BO"
            fine_called = False
        else:
            fine_pred_label = fine_bundles[coarse_pred_group].predict_single(trial["epoch_slot1"])
            first_syl_pred = LABEL_TO_SYLLABLE[fine_pred_label]
            fine_called = True
        first_syl_correct = (first_syl_pred == syl1_true)

        sa_called = False
        second_syl_pred = None
        if first_syl_pred in DETERMINISTIC_FIRST_SYLLABLE_TO_WORD:
            word_pred = DETERMINISTIC_FIRST_SYLLABLE_TO_WORD[first_syl_pred]
        elif first_syl_pred == "SA":
            sa_called = True
            sa_label = sa_bundle.predict_single(trial["epoch_slot2"])
            second_syl_pred = LABEL_TO_SYLLABLE[sa_label]
            word_pred = SA_BRANCH_SECOND_SYLLABLE_TO_WORD.get(second_syl_pred, "UNKNOWN")
        else:
            word_pred = "UNKNOWN"
        word_correct = (word_pred == word_true)

        if not first_syl_correct:
            category = "coarse_wrong" if not coarse_correct else "fine_wrong_coarse_correct"
        elif syl1_true == "SA":
            category = "correct_all" if word_correct else "sa_branch_wrong"
        else:
            category = "correct_all" if word_correct else "unexpected_deterministic_mismatch"

        records.append({
            "word_true": word_true, "group_true": group_true,
            "coarse_pred_group": coarse_pred_group, "coarse_correct": bool(coarse_correct),
            "fine_called": bool(fine_called), "first_syl_pred": first_syl_pred,
            "first_syl_true": syl1_true, "first_syl_correct": bool(first_syl_correct),
            "sa_called": bool(sa_called), "second_syl_pred": second_syl_pred,
            "second_syl_true": syl2_true if syl1_true == "SA" else None,
            "word_pred": word_pred, "word_correct": bool(word_correct),
            "category": category,
        })

    with open(out_path, "w") as f:
        json.dump({"subject_id": subject_id, "n_test_trials": len(records), "records": records}, f)


def run_worker_subprocess(subject_id, timeout_sec=900):
    out_path = os.path.join(RESULTS_DIR, f"_worker_tmp_errdecomp_{subject_id}.json")
    if os.path.exists(out_path):
        os.remove(out_path)
    cmd = [sys.executable, THIS_FILE, "--worker-subject", subject_id, "--worker-out", out_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": f"worker subprocess timed out after {timeout_sec}s"}
    if not os.path.exists(out_path):
        return {"available": False, "reason": "worker subprocess produced no output (crashed)",
                "returncode": proc.returncode, "stderr_tail": (proc.stderr or "")[-3000:]}
    with open(out_path) as f:
        result = json.load(f)
    os.remove(out_path)
    return result


def summarize(raw):
    if not raw.get("records"):
        return {"available": False, "reason": raw.get("reason", "no records")}

    records = raw["records"]
    n_total = len(records)
    cat_counts = {}
    for r in records:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1

    n_wrong = n_total - cat_counts.get("correct_all", 0)
    error_categories = {k: v for k, v in cat_counts.items() if k != "correct_all"}
    error_pct_of_total_errors = {
        k: (round(100.0 * v / n_wrong, 2) if n_wrong else 0.0) for k, v in error_categories.items()
    }

    return {
        "available": True,
        "n_test_trials": n_total,
        "n_correct_all": cat_counts.get("correct_all", 0),
        "n_wrong_total": n_wrong,
        "category_counts": cat_counts,
        "error_category_pct_of_total_errors": error_pct_of_total_errors,
        "records": records,
    }


def main():
    print(f"[TASK 1.5] Decomposing full-word errors for {P6_CHAMPION_SUBJECT} (isolated subprocess)...")
    raw = run_worker_subprocess(P6_CHAMPION_SUBJECT)
    if "records" in raw:
        summary = summarize(raw)
    else:
        summary = {"available": False, "reason": raw.get("reason", "unknown failure"), "raw": raw}

    result = {
        "description": (
            "Full-word error decomposition for P6 S3, categorised by which pipeline stage first "
            "diverged from ground truth: coarse_wrong, fine_wrong_coarse_correct, sa_branch_wrong, "
            "or correct_all (sanity bucket). Same trial-level 80/20 split as compute_full_word_e2e_accuracy."
        ),
        "subject_id": P6_CHAMPION_SUBJECT,
        **summary,
    }
    save_json(result, "p6_error_decomposition_s3.json")

    if result.get("available"):
        print(f"[TASK 1.5] n_test_trials={result['n_test_trials']} n_correct={result['n_correct_all']} "
              f"n_wrong={result['n_wrong_total']}")
        for cat, pct in result["error_category_pct_of_total_errors"].items():
            print(f"[TASK 1.5]   {cat}: {result['category_counts'][cat]} ({pct}% of errors)")
    else:
        print(f"[TASK 1.5] UNAVAILABLE: {result.get('reason')}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-subject", default=None)
    parser.add_argument("--worker-out", default=None)
    args = parser.parse_args()
    if args.worker_subject:
        worker_decompose(args.worker_subject, args.worker_out)
    else:
        main()

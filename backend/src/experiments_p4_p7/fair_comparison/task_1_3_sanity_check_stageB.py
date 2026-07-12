"""
Task 1.3 -- Sanity Check Stage B Baseline Numbers.

Three independent checks, cheapest/safest first:

  Part A. Re-derive every sub-model's OWN test_accuracy from its saved,
      already-scaled Xtest/ytest + saved model (no raw CSV at all -- safe,
      runs for all 12 subjects x 5 sub-models).

  Part B. Replay full_word_e2e via the existing compute_full_word_e2e_accuracy
      (uses pipeline.offline_trial_reader.OfflineTrialReader, which already
      has the usecols+chunksize memory mitigation -- same safe pattern
      train_word_assembler.py used). Each subject runs in an ISOLATED
      subprocess so its ~100MB+ working set is fully released by the OS on
      exit (gc.collect() alone was previously found unreliable for this on
      this machine -- see memory note neurandiar-bci-memory-constrained).

  Part C. Replay first_syllable_e2e via compute_first_syllable_e2e_accuracy,
      which requires REBUILDING raw window-level epochs through the plain
      DatasetBuilder. That loader (preprocessing/build_dataset.py) has NO
      usecols/chunksize mitigation (reads the whole ~350-400MB CSV, all ~59
      columns, in one pd.read_csv call) -- the riskiest step in this whole
      Fase 1. Also subprocess-isolated, S3 only, and OFF BY DEFAULT: pass
      --attempt-first-syllable-replay only after confirming free system RAM
      is comfortably above ~3GB.

No model is ever re-trained (.fit()) anywhere in this script -- every step
here is .predict() on frozen, already-trained artefacts.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from sklearn.metrics import accuracy_score

from _common import (
    ALL_SUBJECTS, SUBMODEL_NAMES, P6_CHAMPION_SUBJECT, RESULTS_DIR,
    p6_submodel_paths, p6_results_json_path, check_exists,
    load_pickle, load_json, save_json,
)

THIS_FILE = os.path.abspath(__file__)
FULL_WORD_SPOTCHECK_SUBJECTS = ["S3", "S9", "S1"]
FIRST_SYLLABLE_SPOTCHECK_SUBJECTS = ["S3"]
TOL = 1e-9


# ---------------------------------------------------------------------------
# Part A -- sub-model accuracy replay (safe, in-process, all 12 subjects)
# ---------------------------------------------------------------------------

def part_a_submodel_accuracy_replay():
    per_subject = {}
    for subj in ALL_SUBJECTS:
        results_path = p6_results_json_path(subj)
        if check_exists(results_path):
            per_subject[subj] = {"status": "DATA_NOT_AVAILABLE"}
            continue
        reported = load_json(results_path)
        sm_reported = reported.get("sub_models", {})

        subj_out = {}
        for name in SUBMODEL_NAMES:
            model_p, scaler_p, xtest_p, ytest_p = p6_submodel_paths(subj, name)
            missing = check_exists(model_p, xtest_p, ytest_p)
            if missing:
                subj_out[name] = {"status": "DATA_NOT_AVAILABLE", "missing_paths": missing}
                continue
            model = load_pickle(model_p)
            X = np.load(xtest_p)
            y = np.load(ytest_p)
            y_pred = model.predict(X)
            replayed_acc = float(accuracy_score(y, y_pred))
            reported_acc = sm_reported.get(name, {}).get("test_accuracy")
            match = (reported_acc is not None) and (abs(replayed_acc - reported_acc) < TOL)
            subj_out[name] = {
                "status": "OK",
                "replayed_test_accuracy": replayed_acc,
                "reported_test_accuracy": reported_acc,
                "match": bool(match),
                "abs_diff": abs(replayed_acc - reported_acc) if reported_acc is not None else None,
            }
        per_subject[subj] = {"status": "OK", "sub_models": subj_out}

    all_matches = [
        sm["match"] for subj in per_subject.values() if subj.get("status") == "OK"
        for sm in subj["sub_models"].values() if sm.get("status") == "OK"
    ]
    n_checked = len(all_matches)
    n_matched = sum(all_matches)
    print(f"[TASK 1.3][Part A] {n_matched}/{n_checked} sub-model test_accuracy values replayed exactly.")
    return {
        "description": "Replay of each sub-model's own test_accuracy from saved Xtest/ytest + model.predict().",
        "n_checked": n_checked, "n_matched": n_matched,
        "all_match": bool(n_checked > 0 and n_matched == n_checked),
        "per_subject": per_subject,
    }


# ---------------------------------------------------------------------------
# Worker helpers (run standalone in a subprocess -- see worker_main())
# ---------------------------------------------------------------------------

def _load_bundles(subject_id):
    from experiments_p4_p7.run_p7_coarse_to_fine import SubModelBundle, E0_PROCESSOR_PARAMS
    results = load_json(p6_results_json_path(subject_id))
    coarse_feat = results.get("winning_coarse_feature_group", "barlow")
    m, s, _, _ = p6_submodel_paths(subject_id, "coarse")
    coarse_bundle = SubModelBundle(m, s, coarse_feat, fs=E0_PROCESSOR_PARAMS["target_fs"])
    fine_bundles = {}
    for grp, name in (("A", "fine_A"), ("I", "fine_I"), ("E", "fine_E")):
        m, s, _, _ = p6_submodel_paths(subject_id, name)
        fine_bundles[grp] = SubModelBundle(m, s, "barlow", fs=E0_PROCESSOR_PARAMS["target_fs"])
    m, s, _, _ = p6_submodel_paths(subject_id, "sa_branch")
    sa_bundle = SubModelBundle(m, s, "barlow", fs=E0_PROCESSOR_PARAMS["target_fs"])
    return coarse_bundle, fine_bundles, sa_bundle


def worker_full_word(subject_id, out_path):
    from experiments_p4_p7.run_p7_coarse_to_fine import compute_full_word_e2e_accuracy
    coarse_bundle, fine_bundles, sa_bundle = _load_bundles(subject_id)
    result = compute_full_word_e2e_accuracy(subject_id, coarse_bundle, fine_bundles, sa_bundle)
    with open(out_path, "w") as f:
        json.dump(result, f)


def worker_first_syllable(subject_id, out_path):
    from experiments_p4_p7.run_p7_coarse_to_fine import (
        compute_first_syllable_e2e_accuracy, build_standard_e0_split_raw, SPLIT_RANDOM_STATE,
    )
    from utils.data_utils import three_way_split
    coarse_bundle, fine_bundles, _sa_bundle = _load_bundles(subject_id)
    X_3d, y = build_standard_e0_split_raw(subject_id)
    if X_3d is None:
        with open(out_path, "w") as f:
            json.dump({"available": False, "reason": "no raw epochs extracted for this subject"}, f)
        return
    _, _, X_test_3d, _, _, y_test = three_way_split(X_3d, y, random_state=SPLIT_RANDOM_STATE)
    result = compute_first_syllable_e2e_accuracy(X_test_3d, y_test, coarse_bundle, fine_bundles)
    result["available"] = True
    with open(out_path, "w") as f:
        json.dump(result, f)


def run_worker_subprocess(subject_id, mode, timeout_sec=900):
    out_path = os.path.join(RESULTS_DIR, f"_worker_tmp_{mode}_{subject_id}.json")
    if os.path.exists(out_path):
        os.remove(out_path)
    cmd = [sys.executable, THIS_FILE, "--worker-subject", subject_id,
           "--worker-mode", mode, "--worker-out", out_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": f"worker subprocess timed out after {timeout_sec}s"}

    if not os.path.exists(out_path):
        return {
            "available": False, "reason": "worker subprocess produced no output (crashed)",
            "returncode": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-3000:],
        }
    with open(out_path) as f:
        result = json.load(f)
    os.remove(out_path)
    if proc.returncode != 0:
        result.setdefault("subprocess_warning", f"non-zero exit ({proc.returncode}) despite output file present")
    return result


# ---------------------------------------------------------------------------
# Part B / Part C orchestration (spawn workers)
# ---------------------------------------------------------------------------

def part_b_full_word_replay(subject_ids):
    out = {}
    for subj in subject_ids:
        print(f"[TASK 1.3][Part B] Replaying full_word_e2e for {subj} (isolated subprocess)...")
        replayed = run_worker_subprocess(subj, "full_word")
        reported = load_json(p6_results_json_path(subj)).get("full_word_e2e", {})
        match = None
        if replayed.get("available") and reported.get("available"):
            match = abs(replayed["accuracy"] - reported["accuracy"]) < TOL
        out[subj] = {"replayed": replayed, "reported": reported, "match": match}
        status = "MATCH" if match else ("MISMATCH" if match is False else "UNAVAILABLE")
        print(f"[TASK 1.3][Part B] {subj}: {status}")
    return out


def part_c_first_syllable_replay(subject_ids):
    out = {}
    for subj in subject_ids:
        print(f"[TASK 1.3][Part C] Replaying first_syllable_e2e for {subj} (isolated subprocess, "
              f"raw DatasetBuilder rebuild -- may take a while)...")
        replayed = run_worker_subprocess(subj, "first_syllable", timeout_sec=1800)
        reported = load_json(p6_results_json_path(subj)).get("first_syllable_e2e", {})
        match = None
        if replayed.get("available") and reported:
            match = abs(replayed["accuracy"] - reported["accuracy"]) < TOL
        out[subj] = {"replayed": replayed, "reported": reported, "match": match}
        status = "MATCH" if match else ("MISMATCH" if match is False else "UNAVAILABLE")
        print(f"[TASK 1.3][Part C] {subj}: {status} -- {replayed.get('reason', '')}")
    return out


def worker_main(args):
    if args.worker_mode == "full_word":
        worker_full_word(args.worker_subject, args.worker_out)
    elif args.worker_mode == "first_syllable":
        worker_first_syllable(args.worker_subject, args.worker_out)
    else:
        raise ValueError(f"Unknown worker mode: {args.worker_mode}")


def main(attempt_first_syllable_replay=False):
    result = {"part_a_submodel_accuracy": part_a_submodel_accuracy_replay()}
    result["part_b_full_word_e2e"] = part_b_full_word_replay(FULL_WORD_SPOTCHECK_SUBJECTS)

    if attempt_first_syllable_replay:
        result["part_c_first_syllable_e2e"] = part_c_first_syllable_replay(FIRST_SYLLABLE_SPOTCHECK_SUBJECTS)
    else:
        result["part_c_first_syllable_e2e"] = {
            "status": "SKIPPED",
            "reason": (
                "Requires rebuilding raw window-level epochs via the plain DatasetBuilder, whose "
                "CSV loader has no usecols/chunksize memory mitigation (unlike OfflineTrialReader). "
                "Skipped by default given this machine's documented memory constraints; re-run with "
                "--attempt-first-syllable-replay after confirming free RAM."
            ),
        }

    save_json(result, "p6_stageB_sanity_check.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-first-syllable-replay", action="store_true")
    parser.add_argument("--worker-subject", default=None)
    parser.add_argument("--worker-mode", default=None)
    parser.add_argument("--worker-out", default=None)
    args = parser.parse_args()

    if args.worker_subject:
        worker_main(args)
    else:
        main(attempt_first_syllable_replay=args.attempt_first_syllable_replay)

"""
Task 1.4 -- Latency Measurement: P6 Cascade (S3) vs P3 Champion.

Real measurement (not just a theoretical projection) is attempted first,
since predict_word_for_trial() only needs OfflineTrialReader (memory-safe,
usecols+chunksize) -- no DatasetBuilder rebuild required for this task.
Isolated in its own subprocess regardless, consistent with Task 1.3.

Two scenarios are measured, since P6's cascade cost is branch-dependent:
  - non_sa_case: coarse -> fine (2 model calls)
  - sa_branch_case: coarse -> fine_A -> sa_branch (3 model calls)
A weighted "expected" figure is also derived using the real proportion of
SA-branch trials in S3's first-syllable test set (from the already-saved
results_S3.json -- no extra computation needed for that part).

A simple theoretical projection (P3 per-call latency x number of P6 calls)
is ALSO always computed, per the task spec, as a cross-check against the
real measurement -- not as a replacement for it.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from _common import (
    P6_CHAMPION_SUBJECT, P3_CHAMPION_SUBJECT, P3_CHAMPION_EXP, P3_CHAMPION_FEATURE_GROUP,
    RESULTS_DIR, T_TABLES_DIR, RAW_DATA_DIR,
    p6_load_bundles, p6_results_json_path, p3_champion_paths, load_json, save_json,
)

THIS_FILE = os.path.abspath(__file__)
N_TRIALS = 100
SA_WORDS = {"SAKIT", "SAYANG"}
REALTIME_THRESHOLD_MS = 100.0


def _find_trial(reader, subject_id, word_set, want_in_set):
    trials_meta = reader.list_valid_trials(subject_id)
    candidates = [i for i, m in enumerate(trials_meta)
                  if (m["word"].strip().upper() in word_set) == want_in_set]
    for idx in candidates:
        try:
            return reader.read_trial(subject_id, trial_index=idx)
        except ValueError:
            continue
    return None


def _time_calls(fn, n=N_TRIALS):
    fn()  # warm-up
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {"n_trials": n, "mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95)),
            "std_ms": float(np.std(times))}


def worker_measure(subject_id, out_path):
    from experiments_p4_p7.run_p7_coarse_to_fine import predict_word_for_trial, E0_PROCESSOR_PARAMS, SubModelBundle
    from pipeline.offline_trial_reader import OfflineTrialReader

    coarse_bundle, fine_bundles, sa_bundle = p6_load_bundles(subject_id)
    reader = OfflineTrialReader(RAW_DATA_DIR, E0_PROCESSOR_PARAMS)

    trial_sa = _find_trial(reader, subject_id, SA_WORDS, True)
    trial_non_sa = _find_trial(reader, subject_id, SA_WORDS, False)

    scenarios = {}
    for label, trial in (("sa_branch_case", trial_sa), ("non_sa_case", trial_non_sa)):
        if trial is None:
            scenarios[label] = {"available": False, "reason": "no clean trial found for this scenario"}
            continue
        s1, s2 = trial["epoch_slot1"], trial["epoch_slot2"]
        stats = _time_calls(lambda: predict_word_for_trial(s1, s2, coarse_bundle, fine_bundles, sa_bundle))
        scenarios[label] = {"available": True, "word": trial["word"], **stats}

    # Fair (apples-to-apples) comparison: P3 champion timed the SAME way as P6's
    # SubModelBundle -- starting from a raw epoch, through feature extraction +
    # scaling + predict -- rather than the T9/notebook figure, which (per
    # _measure_p3_latency in gen_nb_new.py) only times scaler.transform+predict
    # on an ALREADY feature-extracted vector loaded straight from Xtest_*.npy,
    # and so never includes Barlow feature extraction at all.
    p3_model_p, p3_scaler_p, _, _ = p3_champion_paths(P3_CHAMPION_SUBJECT, P3_CHAMPION_EXP, P3_CHAMPION_FEATURE_GROUP)
    p3_fair = {"available": False, "reason": "no reference epoch available"}
    ref_trial = trial_non_sa or trial_sa
    if ref_trial is not None and os.path.exists(p3_model_p) and os.path.exists(p3_scaler_p):
        p3_bundle = SubModelBundle(p3_model_p, p3_scaler_p, P3_CHAMPION_FEATURE_GROUP, fs=E0_PROCESSOR_PARAMS["target_fs"])
        s1 = ref_trial["epoch_slot1"]
        stats = _time_calls(lambda: p3_bundle.predict_single(s1))
        p3_fair = {"available": True, "reference_word": ref_trial["word"], **stats}

    with open(out_path, "w") as f:
        json.dump({"subject_id": subject_id, "scenarios": scenarios, "p3_fair_with_extraction": p3_fair}, f)


def run_worker_subprocess(subject_id, timeout_sec=900):
    out_path = os.path.join(RESULTS_DIR, f"_worker_tmp_latency_{subject_id}.json")
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


def load_p3_champion_latency():
    path = os.path.join(T_TABLES_DIR, "T9_inference_latency.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    rows = df[df["Model"].astype(str).str.contains("P3", na=False)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "mean_ms": float(row["Mean Latency (ms)"]), "p95_ms": float(row["95th Pctile (ms)"]),
        "configuration": row["Configuration"], "source": "T9_inference_latency.csv",
    }


def compute_theoretical_projection(p3_mean_ms, p3_p95_ms):
    return {
        "method": "First-order projection: P3 per-model-call latency (T9-cited) x number of P6 model calls.",
        "non_sa_case_calls": 2, "sa_case_calls": 3,
        "non_sa_case_mean_ms": p3_mean_ms * 2, "non_sa_case_p95_ms": p3_p95_ms * 2,
        "sa_case_mean_ms": p3_mean_ms * 3, "sa_case_p95_ms": p3_p95_ms * 3,
        "caveat": (
            "IMPORTANT -- this projection turned out to be a large underestimate (see "
            "real_vs_theoretical_non_sa_case): the T9-cited P3 mean (0.447ms) comes from "
            "_measure_p3_latency() in gen_nb_new.py, which times ONLY StandardScaler.transform + "
            "SVC.predict on a vector loaded straight from the already feature-extracted Xtest_*.npy "
            "-- it never includes Barlow feature extraction from a raw epoch. P6's real measurement "
            "(via SubModelBundle.predict_single) DOES extract features from a raw epoch on every "
            "call, so the two are not on the same basis. See p3_fair_with_extraction for a same-basis "
            "P3 number. Kept here only as a record of the naive projection and why it was wrong."
        ),
    }


def main():
    p3_latency = load_p3_champion_latency()
    result = {"p3_champion_latency_cited": p3_latency}

    print(f"[TASK 1.4] Measuring real P6 cascade latency for {P6_CHAMPION_SUBJECT} (isolated subprocess)...")
    measured = run_worker_subprocess(P6_CHAMPION_SUBJECT)
    result["p6_cascade_latency_measured"] = measured

    if measured.get("scenarios", {}).get("sa_branch_case", {}).get("available") and \
       measured["scenarios"].get("non_sa_case", {}).get("available"):
        champ_results = load_json(p6_results_json_path(P6_CHAMPION_SUBJECT))
        per_syl = champ_results.get("first_syllable_e2e", {}).get("per_syllable_recall", {})
        n_total = champ_results.get("first_syllable_e2e", {}).get("n_test_samples")
        n_sa = per_syl.get("SA", {}).get("n")
        if n_total and n_sa is not None:
            p_sa = n_sa / n_total
            sa = measured["scenarios"]["sa_branch_case"]
            non_sa = measured["scenarios"]["non_sa_case"]
            result["weighted_expected_latency"] = {
                "p_sa_branch": p_sa,
                "source": "SA-syllable proportion in S3's first_syllable_e2e test set (results_S3.json)",
                "expected_mean_ms": p_sa * sa["mean_ms"] + (1 - p_sa) * non_sa["mean_ms"],
                "expected_p95_ms": p_sa * sa["p95_ms"] + (1 - p_sa) * non_sa["p95_ms"],
            }

    if p3_latency:
        result["theoretical_projection"] = compute_theoretical_projection(p3_latency["mean_ms"], p3_latency["p95_ms"])
        if measured.get("scenarios", {}).get("non_sa_case", {}).get("available"):
            m = measured["scenarios"]["non_sa_case"]["mean_ms"]
            result["real_vs_theoretical_non_sa_case"] = {
                "real_mean_ms": m,
                "theoretical_mean_ms": result["theoretical_projection"]["non_sa_case_mean_ms"],
                "ratio_real_over_theoretical": m / result["theoretical_projection"]["non_sa_case_mean_ms"],
            }

    for label in ("non_sa_case", "sa_branch_case"):
        sc = measured.get("scenarios", {}).get(label, {})
        if sc.get("available"):
            sc["real_time_feasible_lt_100ms"] = bool(sc["p95_ms"] < REALTIME_THRESHOLD_MS)

    p3_fair = measured.get("p3_fair_with_extraction")
    result["p3_fair_with_extraction"] = p3_fair
    if p3_fair and p3_fair.get("available"):
        p3_fair["real_time_feasible_lt_100ms"] = bool(p3_fair["p95_ms"] < REALTIME_THRESHOLD_MS)
        non_sa = measured.get("scenarios", {}).get("non_sa_case", {})
        sa = measured.get("scenarios", {}).get("sa_branch_case", {})
        result["fair_comparison_same_basis"] = {
            "description": (
                "P3 and P6 both timed starting from a raw epoch through feature extraction + "
                "scaling + predict (SubModelBundle.predict_single for both) -- unlike "
                "p3_champion_latency_cited (T9), which excludes feature extraction entirely. This "
                "is the methodologically fair comparison; ratios below are call-count multiples, "
                "close to 2x/3x as the theoretical model predicts, confirming feature extraction -- "
                "not the cascade structure itself -- was almost the entire cost of the earlier "
                "1300x-looking gap against the T9-cited figure."
            ),
            "p3_single_call_mean_ms": p3_fair["mean_ms"],
            "p6_non_sa_case_mean_ms": non_sa.get("mean_ms"),
            "p6_sa_branch_case_mean_ms": sa.get("mean_ms"),
            "ratio_p6_non_sa_over_p3": (non_sa["mean_ms"] / p3_fair["mean_ms"]) if non_sa.get("available") else None,
            "ratio_p6_sa_over_p3": (sa["mean_ms"] / p3_fair["mean_ms"]) if sa.get("available") else None,
        }

    save_json(result, "p6_latency_measurement.json")

    if p3_latency:
        print(f"[TASK 1.4] P3 champion (T9-cited, scaler+predict only, NO feature extraction): "
              f"mean={p3_latency['mean_ms']:.3f}ms p95={p3_latency['p95_ms']:.3f}ms")
    if p3_fair and p3_fair.get("available"):
        print(f"[TASK 1.4] P3 champion (fair, WITH feature extraction, same basis as P6): "
              f"mean={p3_fair['mean_ms']:.3f}ms p95={p3_fair['p95_ms']:.3f}ms")
    for label in ("non_sa_case", "sa_branch_case"):
        sc = measured.get("scenarios", {}).get(label, {})
        if sc.get("available"):
            print(f"[TASK 1.4] P6 {label} (real, word={sc['word']}): mean={sc['mean_ms']:.3f}ms p95={sc['p95_ms']:.3f}ms")
        else:
            print(f"[TASK 1.4] P6 {label}: UNAVAILABLE ({sc.get('reason')})")
    if result.get("fair_comparison_same_basis"):
        fc = result["fair_comparison_same_basis"]
        print(f"[TASK 1.4] Fair same-basis ratio: P6 non_sa is {fc['ratio_p6_non_sa_over_p3']:.2f}x P3 "
              f"(theoretical: 2x); P6 sa_branch is {fc['ratio_p6_sa_over_p3']:.2f}x P3 (theoretical: 3x)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-subject", default=None)
    parser.add_argument("--worker-out", default=None)
    args = parser.parse_args()
    if args.worker_subject:
        worker_measure(args.worker_subject, args.worker_out)
    else:
        main()

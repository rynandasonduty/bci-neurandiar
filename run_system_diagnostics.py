"""
NEURANDIAR-BCI — System Diagnostics Script
===========================================
Performs a comprehensive integrity check of the trained model artefacts,
test sets, raw dataset, MLflow database, and backend source code.

Usage (from repo root):
    python run_system_diagnostics.py

Exit codes:
    0  All checks passed (or passed with warnings).
    1  One or more FAIL conditions detected.
"""

import os
import sys
import ast
import glob
import time

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
REPO_ROOT    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.join(REPO_ROOT, "backend")
WEIGHTS_DIR  = os.path.join(BACKEND_DIR, "models", "weights")
LOGS_DIR     = os.path.join(BACKEND_DIR, "logs")
MLFLOW_DB    = os.path.join(LOGS_DIR, "mlflow", "mlruns.db")
RAW_DATA_DIR = os.path.join(BACKEND_DIR, "dataset", "raw")
SRC_DIR      = os.path.join(BACKEND_DIR, "src")

PARADIGMS = ["P1_Global", "P2_EEGNet", "P3_SVM"]

EXPERIMENTS = [
    "E0_Baseline",
    "E1_ICA_Filtering",
    "E2_Resampling_512Hz",
    "E3_ERP_N400",
    "E4_Channel_Language",
    "E5_Data_Augmentation",
    "E6_CrossModality_ImaginedOnly",
    "E7_Band_Alpha",
]

SUBJECTS = [f"S{i}" for i in range(1, 13)]  # S1 – S12
SVM_FEAT_GROUPS = ["all", "time", "hjorth", "barlow", "band_ratio"]

# ---------------------------------------------------------------------------
# RESULT TRACKING
# ---------------------------------------------------------------------------
PASS   = "PASS"
WARN   = "WARN"
FAIL   = "FAIL"

results = []

def record(status, category, description, detail=""):
    results.append({"status": status, "category": category, "description": description, "detail": detail})

def log_header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

def log_result(status, label, detail=""):
    icons = {PASS: "[PASS]", WARN: "[WARN]", FAIL: "[FAIL]"}
    icon = icons.get(status, "[????]")
    line = f"  {icon}  {label}"
    if detail:
        line += f"\n         Detail: {detail}"
    print(line)

# ---------------------------------------------------------------------------
# CHECK 1: P1_GLOBAL MODEL ARTEFACTS
# ---------------------------------------------------------------------------
def check_p1_models():
    log_header("CHECK 1 — P1_Global Model Artefacts (8 Experiments)")
    p1_root = os.path.join(WEIGHTS_DIR, "P1_Global")
    total_pass = 0
    total_fail = 0

    for exp in EXPERIMENTS:
        exp_dir = os.path.join(p1_root, exp)
        model_file  = os.path.join(exp_dir, f"eegnet_trained_{exp}.h5")
        scaler_file = os.path.join(exp_dir, f"scaler_{exp}.pkl")
        x_test      = os.path.join(exp_dir, "X_test.npy")
        y_test      = os.path.join(exp_dir, "y_test.npy")

        all_present = all(os.path.exists(f) for f in [model_file, scaler_file, x_test, y_test])
        missing = [os.path.basename(f) for f in [model_file, scaler_file, x_test, y_test]
                   if not os.path.exists(f)]

        if all_present:
            log_result(PASS, f"P1_Global / {exp}")
            record(PASS, "P1_Global", exp)
            total_pass += 1
        else:
            log_result(FAIL, f"P1_Global / {exp}", f"Missing: {', '.join(missing)}")
            record(FAIL, "P1_Global", exp, f"Missing: {missing}")
            total_fail += 1

    # LogReg assembler (stored in P1_Global / E0_Baseline as reference)
    logreg_path = os.path.join(p1_root, "E0_Baseline", "logreg_assembler_E0_Baseline.pkl")
    if os.path.exists(logreg_path):
        log_result(PASS, "P1_Global / LogReg Assembler (E0_Baseline)")
        record(PASS, "P1_Global", "LogReg Assembler E0_Baseline")
    else:
        log_result(WARN, "P1_Global / LogReg Assembler (E0_Baseline) — not found", logreg_path)
        record(WARN, "P1_Global", "LogReg Assembler E0_Baseline", logreg_path)

    print(f"\n  Summary — P1_Global: {total_pass} passed, {total_fail} failed (of {len(EXPERIMENTS)} experiments).")

# ---------------------------------------------------------------------------
# CHECK 2: P2_EEGNET MODEL ARTEFACTS
# ---------------------------------------------------------------------------
def check_p2_models():
    log_header("CHECK 2 — P2_EEGNet Subject-Dependent Artefacts (8 Experiments x 12 Subjects)")
    p2_root = os.path.join(WEIGHTS_DIR, "P2_EEGNet")
    total_pass = 0
    total_fail = 0

    for exp in EXPERIMENTS:
        exp_dir = os.path.join(p2_root, exp)
        for subj in SUBJECTS:
            model_file  = os.path.join(exp_dir, f"{exp}_{subj}.h5")
            scaler_file = os.path.join(exp_dir, f"scaler_{exp}_{subj}.pkl")
            xtest_file  = os.path.join(exp_dir, f"Xtest_{exp}_{subj}.npy")
            ytest_file  = os.path.join(exp_dir, f"ytest_{exp}_{subj}.npy")

            all_present = all(os.path.exists(f) for f in [model_file, scaler_file, xtest_file, ytest_file])
            missing = [os.path.basename(f) for f in [model_file, scaler_file, xtest_file, ytest_file]
                       if not os.path.exists(f)]

            if all_present:
                total_pass += 1
            else:
                log_result(FAIL, f"P2_EEGNet / {exp} / {subj}", f"Missing: {', '.join(missing)}")
                record(FAIL, "P2_EEGNet", f"{exp}/{subj}", f"Missing: {missing}")
                total_fail += 1

    if total_fail == 0:
        log_result(PASS, f"All P2_EEGNet artefacts present ({total_pass} model+scaler+test-set quadruplets)")
        record(PASS, "P2_EEGNet", "All artefacts")
    print(f"\n  Summary — P2_EEGNet: {total_pass} quadruplets passed, {total_fail} missing (model, scaler, Xtest, ytest).")

# ---------------------------------------------------------------------------
# CHECK 3: P3_SVM MODEL ARTEFACTS
# ---------------------------------------------------------------------------
def check_p3_models():
    log_header("CHECK 3 — P3_SVM Feature Ablation Artefacts (8 Experiments x 12 Subjects x 5 Groups)")
    p3_root = os.path.join(WEIGHTS_DIR, "P3_SVM")
    total_pass = 0
    total_fail = 0

    for exp in EXPERIMENTS:
        exp_dir = os.path.join(p3_root, exp)
        for subj in SUBJECTS:
            for feat_grp in SVM_FEAT_GROUPS:
                model_file  = os.path.join(exp_dir, f"SVM_{feat_grp}_{exp}_{subj}.pkl")
                scaler_file = os.path.join(exp_dir, f"scaler_SVM_{feat_grp}_{exp}_{subj}.pkl")

                # Accept the 'all' group as mandatory; other groups are optional per ablation design
                all_present = all(os.path.exists(f) for f in [model_file, scaler_file])
                missing = [os.path.basename(f) for f in [model_file, scaler_file]
                           if not os.path.exists(f)]

                if all_present:
                    total_pass += 1
                else:
                    record(WARN, "P3_SVM", f"{exp}/{subj}/{feat_grp}", f"Missing: {missing}")
                    total_fail += 1

    if total_fail == 0:
        log_result(PASS, f"All P3_SVM artefacts present ({total_pass} model+scaler pairs)")
        record(PASS, "P3_SVM", "All artefacts")
    else:
        log_result(WARN, f"P3_SVM: {total_fail} model/scaler pairs absent",
                   f"{total_pass} present, {total_fail} missing (check feature group coverage)")
    print(f"\n  Summary — P3_SVM: {total_pass} passed, {total_fail} missing across all feat groups.")

# ---------------------------------------------------------------------------
# CHECK 4: TEST SET ARTEFACTS (P1 FOCUS — ground truth integrity)
# ---------------------------------------------------------------------------
def check_test_sets():
    log_header("CHECK 4 — Test Set Artefacts (P1_Global .npy integrity)")
    p1_root = os.path.join(WEIGHTS_DIR, "P1_Global")
    all_pass = True

    for exp in EXPERIMENTS:
        exp_dir = os.path.join(p1_root, exp)
        x_test  = os.path.join(exp_dir, "X_test.npy")
        y_test  = os.path.join(exp_dir, "y_test.npy")

        if not os.path.exists(x_test) or not os.path.exists(y_test):
            log_result(FAIL, f"Test set missing — {exp}")
            record(FAIL, "TestSets", exp, "X_test.npy or y_test.npy not found")
            all_pass = False
            continue

        try:
            import numpy as np
            X = np.load(x_test, allow_pickle=False)
            y = np.load(y_test, allow_pickle=False)
            n_x, n_y = len(X), len(y)
            if n_x != n_y:
                log_result(FAIL, f"Test set shape mismatch — {exp}", f"X: {n_x}, y: {n_y}")
                record(FAIL, "TestSets", exp, f"Sample count mismatch: X={n_x}, y={n_y}")
                all_pass = False
            else:
                log_result(PASS, f"Test set — {exp}", f"X: {X.shape}, y: {y.shape}")
                record(PASS, "TestSets", exp)
        except Exception as e:
            log_result(FAIL, f"Test set load error — {exp}", str(e))
            record(FAIL, "TestSets", exp, str(e))
            all_pass = False

    return all_pass

# ---------------------------------------------------------------------------
# CHECK 5: RAW DATASET (12 SUBJECTS)
# ---------------------------------------------------------------------------
def check_raw_dataset():
    log_header("CHECK 5 — Raw Dataset (12 Subjects, CSV + Experiment Log)")

    if not os.path.isdir(RAW_DATA_DIR):
        log_result(FAIL, "Raw data directory not found", RAW_DATA_DIR)
        record(FAIL, "RawData", "Directory", RAW_DATA_DIR)
        return

    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    log_files = glob.glob(os.path.join(RAW_DATA_DIR, "logs", "*_experiment_log.txt"))

    found_subj = sorted(set(
        os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files
    ))

    for subj in SUBJECTS:
        subj_csv  = glob.glob(os.path.join(RAW_DATA_DIR, f"{subj}*.csv"))
        subj_log  = os.path.join(RAW_DATA_DIR, "logs", f"{subj}_experiment_log.txt")

        has_csv = len(subj_csv) > 0
        has_log = os.path.exists(subj_log)

        if has_csv and has_log:
            csv_size = os.path.getsize(subj_csv[0]) // 1024
            log_result(PASS, f"Subject {subj}", f"CSV: {os.path.basename(subj_csv[0])} ({csv_size} KB), log found")
            record(PASS, "RawData", subj)
        elif not has_csv and not has_log:
            log_result(FAIL, f"Subject {subj} — CSV and log both absent")
            record(FAIL, "RawData", subj, "CSV and log missing")
        elif not has_csv:
            log_result(WARN, f"Subject {subj} — CSV absent, log present")
            record(WARN, "RawData", subj, "CSV missing")
        else:
            log_result(WARN, f"Subject {subj} — CSV present, log absent")
            record(WARN, "RawData", subj, "Experiment log missing")

    print(f"\n  Found {len(csv_files)} CSV file(s) and {len(log_files)} experiment log(s) in raw data directory.")

# ---------------------------------------------------------------------------
# CHECK 6: MLFLOW DATABASE
# ---------------------------------------------------------------------------
def check_mlflow_db():
    log_header("CHECK 6 — MLflow Tracking Database")

    if not os.path.exists(MLFLOW_DB):
        log_result(WARN, "MLflow database not found", MLFLOW_DB)
        record(WARN, "MLflow", "Database", f"Expected at: {MLFLOW_DB}")
        return

    db_size_kb = os.path.getsize(MLFLOW_DB) // 1024
    log_result(PASS, f"MLflow database present ({db_size_kb} KB)", MLFLOW_DB)
    record(PASS, "MLflow", "Database")

    try:
        import sqlite3
        conn = sqlite3.connect(MLFLOW_DB)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM runs")
        n_runs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT experiment_id) FROM runs")
        n_exps = cur.fetchone()[0]
        conn.close()
        log_result(PASS, f"MLflow database readable — {n_runs} runs across {n_exps} experiment(s)")
        record(PASS, "MLflow", f"Readable: {n_runs} runs, {n_exps} experiments")
    except Exception as e:
        log_result(WARN, "MLflow database exists but could not be queried", str(e))
        record(WARN, "MLflow", "Query error", str(e))

# ---------------------------------------------------------------------------
# CHECK 7: BACKEND SOURCE SYNTAX (AST DRY-RUN)
# ---------------------------------------------------------------------------
def check_source_syntax():
    log_header("CHECK 7 — Backend Source Code Syntax (AST Dry-Run)")

    py_files = glob.glob(os.path.join(SRC_DIR, "**", "*.py"), recursive=True)
    py_files += glob.glob(os.path.join(REPO_ROOT, "run_system_diagnostics.py"))

    total_pass = 0
    total_fail = 0

    for filepath in sorted(py_files):
        rel_path = os.path.relpath(filepath, REPO_ROOT)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=filepath)
            total_pass += 1
        except SyntaxError as e:
            log_result(FAIL, f"Syntax error: {rel_path}", f"Line {e.lineno}: {e.msg}")
            record(FAIL, "Syntax", rel_path, f"Line {e.lineno}: {e.msg}")
            total_fail += 1
        except Exception as e:
            log_result(WARN, f"Could not parse: {rel_path}", str(e))
            record(WARN, "Syntax", rel_path, str(e))

    if total_fail == 0:
        log_result(PASS, f"All {total_pass} Python source files parsed without syntax errors.")
        record(PASS, "Syntax", "All files")
    print(f"\n  Summary — Syntax: {total_pass} clean, {total_fail} error(s).")

# ---------------------------------------------------------------------------
# CHECK 8: INFERENCE HISTORY CSV
# ---------------------------------------------------------------------------
def check_inference_log():
    log_header("CHECK 8 — Inference History Log")
    history_file = os.path.join(LOGS_DIR, "inference_history.csv")

    if os.path.exists(history_file):
        size_kb = os.path.getsize(history_file) // 1024
        log_result(PASS, f"Inference history log present ({size_kb} KB)", history_file)
        record(PASS, "InferenceLog", "File")
    else:
        log_result(WARN, "Inference history log absent (will be created on first API call)", history_file)
        record(WARN, "InferenceLog", "File", "Will be created automatically on first POST /api/logs")

# ---------------------------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------------------------
def print_final_report(elapsed_sec):
    counts = {PASS: 0, WARN: 0, FAIL: 0}
    for r in results:
        counts[r["status"]] += 1

    total = sum(counts.values())

    print(f"\n{'=' * 70}")
    print(f"  FINAL DIAGNOSTIC REPORT — NEURANDIAR-BCI")
    print(f"{'=' * 70}")
    print(f"  Total checks : {total}")
    print(f"  PASS         : {counts[PASS]}")
    print(f"  WARN         : {counts[WARN]}")
    print(f"  FAIL         : {counts[FAIL]}")
    print(f"  Elapsed time : {elapsed_sec:.2f}s")
    print(f"{'=' * 70}")

    if counts[FAIL] > 0:
        print(f"\n  FAILED ITEMS:")
        for r in results:
            if r["status"] == FAIL:
                print(f"    [FAIL] [{r['category']}] {r['description']}")
                if r["detail"]:
                    print(f"           {r['detail']}")

    if counts[WARN] > 0:
        print(f"\n  WARNINGS:")
        for r in results:
            if r["status"] == WARN:
                print(f"    [WARN] [{r['category']}] {r['description']}")
                if r["detail"]:
                    print(f"           {r['detail']}")

    overall = "SYSTEM READY" if counts[FAIL] == 0 else "SYSTEM DEGRADED"
    print(f"\n  Overall Status: {overall}")
    print(f"{'=' * 70}\n")

    return counts[FAIL]

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  NEURANDIAR-BCI — SYSTEM DIAGNOSTICS")
    print(f"  Repository root : {REPO_ROOT}")
    print(f"  Weights root    : {WEIGHTS_DIR}")
    print("=" * 70)

    start = time.time()

    check_p1_models()
    check_p2_models()
    check_p3_models()
    check_test_sets()
    check_raw_dataset()
    check_mlflow_db()
    check_source_syntax()
    check_inference_log()

    elapsed = time.time() - start
    fail_count = print_final_report(elapsed)

    sys.exit(1 if fail_count > 0 else 0)

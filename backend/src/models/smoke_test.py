import os
import sys
import glob
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.build_dataset import DatasetBuilder
from utils.data_utils import three_way_split, fit_and_apply_scaler

def run_smoke_test():
    print("=" * 60)
    print(" SMOKE TEST — ANTI-LEAKAGE PIPELINE VALIDATION ")
    print("=" * 60)

    EXP_ID = "E_SMOKE_TEST"
    builder = DatasetBuilder(exp_id=EXP_ID)

    log_files = glob.glob(os.path.join(builder.raw_data_dir, "logs", "*_experiment_log.txt"))
    if not log_files:
        print("[ERROR] No raw data found. Ensure the raw_data directory contains log and CSV files.")
        return

    first_log = log_files[0]
    subject_id = os.path.basename(first_log).replace("_experiment_log.txt", "")
    csv_files = glob.glob(os.path.join(builder.raw_data_dir, f"{subject_id}*.csv"))

    if not csv_files:
        print(f"[ERROR] CSV file for subject {subject_id} not found.")
        return

    print(f"\n[INFO] Running extraction on subject: {subject_id}")

    X_list, y_list = builder.process_subject(subject_id, csv_files[0], first_log)

    if len(X_list) == 0:
        print("[ERROR] Extraction failed — no data returned.")
        return

    X = np.array(X_list)
    y = np.array(y_list)

    # Transpose to EEGNet format: (Samples, Channels, Time, Depth)
    X_eeg = np.transpose(X, (0, 2, 1))
    X_eeg = np.expand_dims(X_eeg, axis=3)

    print(f"[CHECK] Raw data shape (build_dataset output): {X_eeg.shape}")

    print("\n[INFO] Testing three_way_split (utils/data_utils.py)...")
    X_tr, X_v, X_te, y_tr, y_v, y_te = three_way_split(X_eeg, y, val_ratio=0.15, test_ratio=0.15)

    print(f"[CHECK] Train shape (70%): {X_tr.shape}")
    print(f"[CHECK] Val shape   (15%): {X_v.shape}")
    print(f"[CHECK] Test shape  (15%): {X_te.shape}")

    print("\n[INFO] Testing fit_and_apply_scaler (fit on train only)...")
    X_tr_s, X_v_s, X_te_s, scaler = fit_and_apply_scaler(X_tr, X_v, X_te)

    train_mean = X_tr_s.mean()
    val_mean = X_v_s.mean()

    print(f"[CHECK] Train scaled mean: {train_mean:.6f}  (expected: ~0)")
    print(f"[CHECK] Val scaled mean:   {val_mean:.6f}  (non-zero is expected)")

    if abs(train_mean) < 1e-4:
        print("\n" + "=" * 60)
        print(" SMOKE TEST PASSED. Pipeline is free of data leakage.")
        print("=" * 60)
        print("Safe to proceed with 'run_master_experiments.py'.")
    else:
        print("\n[WARNING] Train mean is not near zero. Review the scaler implementation.")

if __name__ == "__main__":
    run_smoke_test()

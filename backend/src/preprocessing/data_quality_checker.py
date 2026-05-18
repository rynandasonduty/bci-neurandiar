import os
import sys
import glob
import numpy as np
import pandas as pd
from signal_processor import SignalProcessor
from collections import Counter

class DataQualityChecker:
    def __init__(self, csv_filepath):
        self.csv_filepath = csv_filepath
        self.processor = SignalProcessor()

        if not os.path.exists(csv_filepath):
            raise FileNotFoundError(f"File not found: {csv_filepath}")

    def run_qc(self):
        print("\n" + "=" * 60)
        print(f" DATA QUALITY INSPECTION: {os.path.basename(self.csv_filepath)} ")
        print("=" * 60)

        try:
            # Locate the CSV header row containing EEG channel names
            header_idx = 0
            with open(self.csv_filepath, 'r') as f:
                for i, line in enumerate(f):
                    if 'EEG.AF3' in line or 'AF3' in line:
                        header_idx = i
                        break

            df = pd.read_csv(self.csv_filepath, header=header_idx, low_memory=False)

            # Drop the EmotivPRO units row ('uV', 'Hz') if present
            try:
                float(df.iloc[0][self.processor.eeg_channels[0]])
            except (ValueError, TypeError):
                df = df.iloc[1:].reset_index(drop=True)

            for col in self.processor.eeg_channels:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            marker_col = 'MarkerValueInt' if 'MarkerValueInt' in df.columns else 'Marker'
            if marker_col in df.columns:
                df[marker_col] = pd.to_numeric(df[marker_col], errors='coerce').fillna(0)

        except Exception as e:
            print(f"[ERROR] Failed to read CSV — file may be corrupt or invalid. ({e})")
            return

        score = 100
        eeg_cols = self.processor.eeg_channels

        # ==========================================
        # 1. MISSING VALUES
        # ==========================================
        print("\n[1] Missing Values (NaN detection)...")
        missing_count = df[eeg_cols].isnull().sum().sum()
        if missing_count > 0:
            print(f"    [FAIL] {missing_count} missing data points detected.")
            score -= 20
        else:
            print("    [PASS] No missing values detected.")

        # ==========================================
        # 2. MARKER INTEGRITY AND CLASS COVERAGE
        # ==========================================
        print("\n[2] Marker Integrity and Class Coverage...")
        marker_col = 'MarkerValueInt' if 'MarkerValueInt' in df.columns else 'Marker'
        if marker_col not in df.columns:
            print("    [FAIL] LSL Marker column not found.")
            return

        markers = df[df[marker_col] > 0][marker_col].astype(int).tolist()
        total_markers = len(markers)
        print(f"    -> Total markers recorded: {total_markers}  (target: 400)")

        if total_markers < 400:
            print("    [WARNING] Fewer than 400 markers. Possible synchronisation dropout.")
            score -= 15
        else:
            print(f"    [PASS] Sufficient markers recorded ({total_markers}).")

        marker_counts = Counter(markers)
        missing_markers = [i for i in range(1, 20) if i not in marker_counts]
        if missing_markers:
            print(f"    [FAIL] Syllable class IDs {missing_markers} are absent. Class coverage is incomplete.")
            score -= 20
        else:
            print("    [PASS] All 19 syllable class IDs (1–19) are present.")

        # ==========================================
        # 3. SENSOR HEALTH (FLATLINE AND OUTLIER)
        # ==========================================
        print("\n[3] Sensor Health — 14 EEG Channels (Flatline and Noise Outlier)...")
        eeg_data = df[eeg_cols].values

        std_devs = np.std(eeg_data, axis=0)
        median_std = np.median(std_devs)

        bad_channels = []
        noisy_channels = []

        for i, ch in enumerate(eeg_cols):
            if std_devs[i] < 0.1:
                bad_channels.append(ch)
            elif std_devs[i] > (median_std * 5):
                noisy_channels.append(ch)

        if bad_channels:
            print(f"    [FAIL] Dead sensor(s) detected: {bad_channels}")
            score -= 30
        if noisy_channels:
            print(f"    [WARNING] Excessively noisy sensor(s) (outlier vs. ensemble): {noisy_channels}")
            score -= 10
        if not bad_channels and not noisy_channels:
            print("    [PASS] All sensors show healthy signal variance.")

        # ==========================================
        # 4. EXTREME AMPLITUDE SPIKES
        # ==========================================
        print("\n[4] Extreme Amplitude Spikes (> +/-200 uV)...")
        eeg_centered = eeg_data - np.mean(eeg_data, axis=0)
        extreme_spikes = np.sum(np.abs(eeg_centered) > 200)
        spike_ratio = (extreme_spikes / eeg_data.size) * 100

        print(f"    -> Extreme spike ratio: {spike_ratio:.3f}% of total samples.")
        if spike_ratio > 5.0:
            print("    [WARNING] High spike ratio — headset may have been frequently displaced.")
            score -= 15
        else:
            print("    [PASS] Signal is stable with negligible extreme artefacts.")

        # ==========================================
        # 5. ARTIFACT REJECTION RETENTION RATE
        # ==========================================
        print("\n[5] Simulated Filtering and Windowing (rejection threshold: +/-100 uV)...")
        filtered_eeg = self.processor.apply_filter(eeg_data)
        marker_indices = df.index[df[marker_col] > 0].tolist()

        total_expected_windows = 0
        total_clean_windows = 0

        for idx in marker_indices:
            slot_data = filtered_eeg[idx: idx + (5 * self.processor.fs)]
            if len(slot_data) >= 5 * self.processor.fs:
                total_expected_windows += 5
                clean_windows = self.processor.windowing_slot(slot_data)
                total_clean_windows += len(clean_windows)

        retention_rate = (total_clean_windows / total_expected_windows) * 100 if total_expected_windows > 0 else 0

        print(f"    -> Total windows expected: {total_expected_windows}")
        print(f"    -> Windows passing QC:     {total_clean_windows}")
        print(f"    -> CLEAN RETENTION RATE:   {retention_rate:.2f}%")

        if retention_rate < 60:
            score -= 20
        elif retention_rate < 80:
            score -= 5

        # ==========================================
        # 6. FINAL QUALITY GRADE
        # ==========================================
        print("\n" + "=" * 60)
        print(f" FINAL DATA QUALITY SCORE: {score}/100 ")
        print("=" * 60)

        if score >= 90:
            print("[GRADE A] EXCELLENT")
            print("Dataset is clean, free of excessive noise, and markers are complete.")
            print("=> ACTION: Move this file to the primary dataset directory for training.")
        elif score >= 75:
            print("[GRADE B] GOOD")
            print("Dataset contains minor artefacts (e.g., blink) but is suitable for training.")
            print("=> ACTION: Move this file to the primary dataset directory for training.")
        elif score >= 50:
            print("[GRADE C] WARNING")
            print("Data quality is marginal. Model learning may be impaired.")
            print("=> ACTION: Strongly consider repeating the recording session if feasible.")
        else:
            print("[GRADE F] REJECTED")
            print("Severely degraded data: dead sensors, synchronisation failures, or extreme noise.")
            print("=> ACTION: Do NOT use this file. Delete it and repeat the recording session.")

        print("=" * 60 + "\n")

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    TARGET_DIR = os.path.join(BASE_DIR, 'dataset', 'raw')

    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"\n[INFO] Target directory created: {TARGET_DIR}")
        print("    Place EmotivPRO .csv export files into the 'raw' folder and re-run.")
        sys.exit()

    csv_files = glob.glob(os.path.join(TARGET_DIR, "*.csv"))

    if not csv_files:
        print(f"\n[ERROR] No .csv files found in: {TARGET_DIR}")
        print("    Place EmotivPRO .csv export files into the folder and re-run.")
        sys.exit()

    # Analyse the most recently modified CSV file if multiple are present
    latest_csv = max(csv_files, key=os.path.getmtime)

    if len(csv_files) > 1:
        print(f"\n[INFO] Found {len(csv_files)} CSV files. Analysing the most recent: {os.path.basename(latest_csv)}")

    try:
        checker = DataQualityChecker(latest_csv)
        checker.run_qc()
    except Exception as e:
        print(f"\n[ERROR] A critical error occurred during data processing: {e}")

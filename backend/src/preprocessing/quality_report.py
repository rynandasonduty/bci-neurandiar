import os
import glob
import pandas as pd
import numpy as np
import contextlib
from signal_processor import SignalProcessor
from collections import Counter

def generate_global_report():
    RAW_DIR = "../../dataset/raw"
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

    if not csv_files:
        print("[ERROR] No CSV files found in dataset/raw/")
        return

    processor = SignalProcessor()
    channels = processor.eeg_channels

    print("\n" + "=" * 85)
    print(" BCI DATA ACQUISITION QUALITY REPORT — GLOBAL SUMMARY ")
    print("=" * 85)
    print(f"{'SUBJECT ID':<15} | {'CLEAN RETENTION':<15} | {'EXTREME SPIKES':<17} | {'TOTAL WINDOWS':<15} | {'GRADE':<6}")
    print("-" * 85)

    total_retention = []
    total_spikes = []
    total_windows_all = 0
    clean_windows_all = 0

    global_channel_noise = {ch: 0 for ch in channels}
    global_class_counts = Counter()

    for file in sorted(csv_files):
        subj_name = os.path.basename(file).split('.')[0]

        header_idx = 0
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                if 'EEG.AF3' in line or 'AF3' in line:
                    header_idx = i
                    break

        df = pd.read_csv(file, header=header_idx, low_memory=False)

        try:
            float(df.iloc[0][channels[0]])
        except ValueError:
            df = df.drop(0).reset_index(drop=True)

        for col in channels:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # DC-offset correction before spike detection to account for Emotiv baseline drift
        eeg_raw = df[channels].values
        eeg_centered = eeg_raw - np.mean(eeg_raw, axis=0)
        extreme_spikes = np.sum((eeg_centered > 200.0) | (eeg_centered < -200.0))
        spike_percentage = (extreme_spikes / eeg_raw.size) * 100

        eeg_filtered = processor.apply_filter(eeg_raw)

        for ch_idx, ch_name in enumerate(channels):
            violations = np.sum(np.abs(eeg_filtered[:, ch_idx]) > 100.0)
            global_channel_noise[ch_name] += violations

        marker_col = 'MarkerValueInt' if 'MarkerValueInt' in df.columns else 'Marker'
        df[marker_col] = pd.to_numeric(df[marker_col], errors='coerce').fillna(0)

        valid_markers = df[df[marker_col] > 0][marker_col]
        global_class_counts.update(valid_markers.value_counts().to_dict())

        marker_indices = df.index[df[marker_col] > 0].tolist()

        subj_clean_windows = 0
        subj_total_windows = 0

        for idx in marker_indices:
            slot_data = eeg_filtered[idx: idx + (5 * processor.fs)]
            if len(slot_data) == 5 * processor.fs:
                clean_wins = processor.windowing_slot(slot_data)
                subj_clean_windows += len(clean_wins)
                subj_total_windows += 5

        retention_rate = (subj_clean_windows / subj_total_windows) * 100 if subj_total_windows > 0 else 0.0

        total_retention.append(retention_rate)
        total_spikes.append(spike_percentage)
        total_windows_all += subj_total_windows
        clean_windows_all += subj_clean_windows

        if retention_rate >= 80:
            grade = "A (Gold)"
        elif retention_rate >= 60:
            grade = "B (Safe)"
        elif retention_rate >= 30:
            grade = "C (Noisy)"
        else:
            grade = "F (Reject)"

        print(f"{subj_name:<15} | {retention_rate:>13.1f}% | {spike_percentage:>15.2f}% | {subj_clean_windows:>6} / {subj_total_windows:<6} | {grade}")

    # --- GLOBAL SUMMARY ---
    noisiest_ch = max(global_channel_noise, key=global_channel_noise.get)

    counts = list(global_class_counts.values())
    if len(counts) > 0:
        min_class = min(counts)
        max_class = max(counts)
        ratio = min_class / max_class
        if ratio >= 0.8:
            balance_status = f"Well-balanced (class ratio {ratio:.2f})"
        elif ratio >= 0.5:
            balance_status = f"Moderately balanced (class ratio {ratio:.2f})"
        else:
            balance_status = f"Imbalanced (class ratio {ratio:.2f} — bias risk)"
    else:
        balance_status = "No class data available"

    avg_retention = np.mean(total_retention)
    avg_spikes = np.mean(total_spikes)
    min_ret = np.min(total_retention)
    max_ret = np.max(total_retention)

    print("-" * 85)
    print(" BASELINE DATASET SUMMARY:")
    print("-" * 85)
    print(f"[INFO] Subjects evaluated:          {len(csv_files)}")
    print(f"[INFO] Mean clean retention:        {avg_retention:.2f}%  (physiological artefact rate)")
    print(f"[INFO] Mean extreme spike rate:     {avg_spikes:.2f}%  (hardware/connection integrity)")
    print(f"[INFO] Total clean windows:         {clean_windows_all}  (available for EEGNet training)")
    print(f"[INFO] Inter-subject variability:   range {min_ret:.1f}% – {max_ret:.1f}%")
    print(f"[INFO] Noisiest channel:            {noisiest_ch}  (largest artefact contributor)")
    print(f"[INFO] Class balance:               {balance_status}")
    print("-" * 85)

if __name__ == "__main__":
    generate_global_report()

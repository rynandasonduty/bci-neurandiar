import os
import glob
import pandas as pd
import numpy as np
import contextlib
from signal_processor import SignalProcessor
from collections import Counter

def generate_individual_reports():
    RAW_DIR = "../../dataset/raw"
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

    if not csv_files:
        print("[ERROR] No CSV files found in dataset/raw/")
        return

    processor = SignalProcessor()
    channels = processor.eeg_channels

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

        eeg_raw = df[channels].values

        # DC-offset correction before spike detection (Emotiv EPOC X has non-zero baseline)
        eeg_centered = eeg_raw - np.mean(eeg_raw, axis=0)
        extreme_spikes = np.sum((eeg_centered > 200.0) | (eeg_centered < -200.0))
        spike_percentage = (extreme_spikes / eeg_raw.size) * 100

        eeg_filtered = processor.apply_filter(eeg_raw)

        ch_noise = {}
        for ch_idx, ch_name in enumerate(channels):
            ch_noise[ch_name] = np.sum(np.abs(eeg_filtered[:, ch_idx]) > 100.0)
        noisiest_ch = max(ch_noise, key=ch_noise.get)

        marker_col = 'MarkerValueInt' if 'MarkerValueInt' in df.columns else 'Marker'
        df[marker_col] = pd.to_numeric(df[marker_col], errors='coerce').fillna(0)

        marker_indices = df.index[df[marker_col] > 0].tolist()

        subj_clean_windows = 0
        subj_total_windows = 0

        for idx in marker_indices:
            slot_data = eeg_filtered[idx: idx + (5 * processor.fs)]
            if len(slot_data) == 5 * processor.fs:
                with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
                    clean_wins = processor.windowing_slot(slot_data)

                subj_clean_windows += len(clean_wins)
                subj_total_windows += 5

        retention_rate = (subj_clean_windows / subj_total_windows) * 100 if subj_total_windows > 0 else 0.0

        if retention_rate >= 80:
            grade = "A (Gold)"
        elif retention_rate >= 60:
            grade = "B (Safe)"
        elif retention_rate >= 30:
            grade = "C (Noisy)"
        else:
            grade = "F (Reject)"

        print("\n" + "=" * 50)
        print(f" SUBJECT DATA QUALITY REPORT: {subj_name}")
        print("=" * 50)
        print(f"[INFO] Grade:                {grade}")
        print(f"[INFO] Clean retention:      {retention_rate:.2f}%  ({subj_clean_windows} / {subj_total_windows} windows)")
        print(f"[INFO] Extreme spike rate:   {spike_percentage:.4f}%  (hardware integrity)")
        print(f"[INFO] Noisiest channel:     {noisiest_ch}  ({ch_noise[noisiest_ch]} threshold violations)")
        print("-" * 50)

if __name__ == "__main__":
    generate_individual_reports()

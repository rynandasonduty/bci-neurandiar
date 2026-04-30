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
        print("[-] Tidak ada file CSV ditemukan di dataset/raw/")
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
        
        # [PERBAIKAN BUG EMOTIV] Kurangi nilai rata-rata (DC Offset) sebelum mengecek spike
        eeg_centered = eeg_raw - np.mean(eeg_raw, axis=0)
        extreme_spikes = np.sum((eeg_centered > 200.0) | (eeg_centered < -200.0))
        spike_percentage = (extreme_spikes / eeg_raw.size) * 100
        
        eeg_filtered = processor.apply_filter(eeg_raw)
        
        # Cari saluran paling bising individu
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
            slot_data = eeg_filtered[idx : idx + (5 * processor.fs)]
            if len(slot_data) == 5 * processor.fs:
                # Bungkam output saat memproses window
                with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
                    clean_wins = processor.windowing_slot(slot_data)
                
                subj_clean_windows += len(clean_wins)
                subj_total_windows += 5 
                
        if subj_total_windows > 0:
            retention_rate = (subj_clean_windows / subj_total_windows) * 100
        else:
            retention_rate = 0.0
            
        if retention_rate >= 80: grade = "A (Emas)"
        elif retention_rate >= 60: grade = "B (Aman)"
        elif retention_rate >= 30: grade = "C (Bising)"
        else: grade = "F (Bahaya)"
            
        print("\n" + "="*50)
        print(f" KUALITAS DATA SUBJEK: {subj_name}")
        print("="*50)
        print(f"[*] Grade Data       : {grade}")
        print(f"[*] Retensi Bersih   : {retention_rate:.2f}% ({subj_clean_windows} / {subj_total_windows} Jendela)")
        print(f"[*] Lonjakan Alat    : {spike_percentage:.4f}% (Integritas Hardware)")
        print(f"[*] Saluran Terkotor : {noisiest_ch} ({ch_noise[noisiest_ch]} pelanggaran threshold)")
        print("-" * 50)

if __name__ == "__main__":
    generate_individual_reports()
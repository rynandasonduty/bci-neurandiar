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
        print("[-] Tidak ada file CSV ditemukan di dataset/raw/")
        return

    processor = SignalProcessor()
    channels = processor.eeg_channels
    
    print("\n" + "="*85)
    print(" 📊 LAPORAN KUALITAS AKUISISI DATA BCI GLOBAL (12 SUBJEK) ")
    print("="*85)
    print(f"{'ID SUBJEK':<15} | {'RETENSI BERSIH':<15} | {'LONJAKAN EKSTREM':<17} | {'TOTAL WINDOWS':<15} | {'GRADE':<6}")
    print("-" * 85)
    
    total_retention = []
    total_spikes = []
    total_windows_all = 0
    clean_windows_all = 0
    
    global_channel_noise = {ch: 0 for ch in channels}
    global_class_counts = Counter()
    
    for file in sorted(csv_files):
        subj_name = os.path.basename(file).split('.')[0]
        
        # 1. Cari Header
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
            
        # 2. Hitung Extreme Spikes (Hardware Check) dengan perbaikan bug DC Offset
        eeg_raw = df[channels].values
        eeg_centered = eeg_raw - np.mean(eeg_raw, axis=0) # Memusatkan sinyal ke titik 0
        extreme_spikes = np.sum((eeg_centered > 200.0) | (eeg_centered < -200.0))
        spike_percentage = (extreme_spikes / eeg_raw.size) * 100
        
        # 3. Filter Data
        eeg_filtered = processor.apply_filter(eeg_raw)
        
        for ch_idx, ch_name in enumerate(channels):
            violations = np.sum(np.abs(eeg_filtered[:, ch_idx]) > 100.0)
            global_channel_noise[ch_name] += violations

        # Penanda Suku Kata
        marker_col = 'MarkerValueInt' if 'MarkerValueInt' in df.columns else 'Marker'
        df[marker_col] = pd.to_numeric(df[marker_col], errors='coerce').fillna(0)
        
        valid_markers = df[df[marker_col] > 0][marker_col]
        global_class_counts.update(valid_markers.value_counts().to_dict())
        
        # 4. Windowing (Physiological Check)
        marker_indices = df.index[df[marker_col] > 0].tolist()
        
        subj_clean_windows = 0
        subj_total_windows = 0
        
        for idx in marker_indices:
            slot_data = eeg_filtered[idx : idx + (5 * processor.fs)]
            if len(slot_data) == 5 * processor.fs:
                clean_wins = processor.windowing_slot(slot_data)
                subj_clean_windows += len(clean_wins)
                subj_total_windows += 5 
                
        if subj_total_windows > 0:
            retention_rate = (subj_clean_windows / subj_total_windows) * 100
        else:
            retention_rate = 0.0
            
        total_retention.append(retention_rate)
        total_spikes.append(spike_percentage)
        total_windows_all += subj_total_windows
        clean_windows_all += subj_clean_windows
        
        # Penentuan Grade
        if retention_rate >= 80: grade = "A (Emas)"
        elif retention_rate >= 60: grade = "B (Aman)"
        elif retention_rate >= 30: grade = "C (Bising)"
        else: grade = "F (Bahaya)"
            
        print(f"{subj_name:<15} | {retention_rate:>13.1f}% | {spike_percentage:>15.2f}% | {subj_clean_windows:>6} / {subj_total_windows:<6} | {grade}")

    # --- KESIMPULAN AKHIR ---
    noisiest_ch = max(global_channel_noise, key=global_channel_noise.get)
    
    counts = list(global_class_counts.values())
    if len(counts) > 0:
        min_class = min(counts)
        max_class = max(counts)
        ratio = min_class / max_class
        if ratio >= 0.8:
            balance_status = f"Sangat Seimbang (Rasio Kelas {ratio:.2f})"
        elif ratio >= 0.5:
            balance_status = f"Cukup Seimbang (Rasio Kelas {ratio:.2f})"
        else:
            balance_status = f"Timpang / Imbalance (Rasio Kelas {ratio:.2f} - Risiko Bias)"
    else:
        balance_status = "Tidak ada data kelas"

    avg_retention = np.mean(total_retention)
    avg_spikes = np.mean(total_spikes)
    min_ret = np.min(total_retention)
    max_ret = np.max(total_retention)

    print("-" * 85)
    print(" 📈 KESIMPULAN BASELINE DATASET:")
    print("-" * 85)
    print(f"[*] Total Subjek Dievaluasi : {len(csv_files)} Subjek")
    print(f"[*] Rata-Rata Retensi Global: {avg_retention:.2f}% (Tingkat Kebersihan Fisiologis)")
    print(f"[*] Rata-Rata Lonjakan Alat : {avg_spikes:.2f}% (Integritas Hardware/Koneksi)")
    print(f"[*] Total Jendela Bersih    : {clean_windows_all} Jendela siap masuk ke EEGNet")
    print(f"[*] Variansi Antar-Subjek   : Sangat Tinggi (Terendah: {min_ret:.1f}%, Tertinggi: {max_ret:.1f}%)")
    print(f"[*] Saluran Paling Bising   : {noisiest_ch} (Penyumbang artefak terbesar)")
    print(f"[*] Keseimbangan Kelas      : {balance_status}")
    print("-" * 85)

if __name__ == "__main__":
    generate_global_report()
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
            raise FileNotFoundError(f"[!] File tidak ditemukan: {csv_filepath}")

    def run_qc(self):
        print("\n" + "="*60)
        print(f" 🕵️ INSPEKSI MUTU DATA: {os.path.basename(self.csv_filepath)} ")
        print("="*60)
        
        try:
            # 1. Cari baris yang mengandung nama kolom (Header) secara otomatis
            header_idx = 0
            with open(self.csv_filepath, 'r') as f:
                for i, line in enumerate(f):
                    if 'EEG.AF3' in line or 'AF3' in line:
                        header_idx = i
                        break
            
            # 2. Baca CSV mulai dari baris header tersebut
            df = pd.read_csv(self.csv_filepath, header=header_idx, low_memory=False)
            
            # 3. Hapus baris "Units" (misal: 'uV', 'Hz') di baris pertama data jika ada
            try:
                float(df.iloc[0][self.processor.eeg_channels[0]])
            except (ValueError, TypeError):
                df = df.iloc[1:].reset_index(drop=True)
                
            # 4. Paksa konversi kolom EEG menjadi angka (mengatasi error Mixed Types)
            for col in self.processor.eeg_channels:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Paksa konversi kolom Marker menjadi angka
            marker_col = 'MarkerValueInt' if 'MarkerValueInt' in df.columns else 'Marker'
            if marker_col in df.columns:
                df[marker_col] = pd.to_numeric(df[marker_col], errors='coerce').fillna(0)

        except Exception as e:
            print(f"[-] GAGAL MEMBACA CSV: File corrupt atau tidak valid. ({e})")
            return

        score = 100 # Nilai awal
        eeg_cols = self.processor.eeg_channels
              
        # ==========================================
        # 1. UJI MISSING VALUES (Kekosongan Data)
        # ==========================================
        print("\n[1] Uji Missing Values (Data Hilang / NaN)...")
        missing_count = df[eeg_cols].isnull().sum().sum()
        if missing_count > 0:
            print(f"    [-] BAHAYA: Ditemukan {missing_count} titik data kosong (NaN)!")
            score -= 20
        else:
            print("    [+] LULUS: Tidak ada data yang bolong akibat koneksi terputus.")

        # ==========================================
        # 2. UJI INTEGRITAS & KONSISTENSI MARKER
        # ==========================================
        print("\n[2] Uji Integritas & Keberagaman Marker...")
        marker_col = 'MarkerValueInt' if 'MarkerValueInt' in df.columns else 'Marker'
        if marker_col not in df.columns:
            print("    [-] GAGAL: Kolom LSL Marker tidak ditemukan!")
            return
            
        markers = df[df[marker_col] > 0][marker_col].astype(int).tolist()
        total_markers = len(markers)
        print(f"    -> Total Marker Terekam : {total_markers} (Target: 400)")
        
        if total_markers < 400:  # Ubah dari != 400 menjadi < 400
            print("    [-] PERINGATAN: Jumlah marker kurang dari 400. Sinkronisasi terputus.")
            score -= 15
        else:
            print(f"    [+] LULUS: Jumlah marker mencukupi (Terekam: {total_markers}).")

        # Cek Keberagaman (Apakah semua ID Suku Kata 1-19 terwakili?)
        marker_counts = Counter(markers)
        missing_markers = [i for i in range(1, 20) if i not in marker_counts]
        if missing_markers:
            print(f"    [-] BAHAYA: Marker ID {missing_markers} tidak pernah muncul! Keberagaman kelas rusak.")
            score -= 20
        else:
            print("    [+] LULUS: Seluruh 19 kelas suku kata (ID 1-19) terekam secara konsisten.")

        # ==========================================
        # 3. UJI KESEHATAN SENSOR (Flatline & Outlier)
        # ==========================================
        print("\n[3] Uji Kesehatan 14 Sensor EEG (Flatline & Noise Outlier)...")
        eeg_data = df[eeg_cols].values
        
        # Hitung Standar Deviasi per channel untuk melihat sebaran datanya
        std_devs = np.std(eeg_data, axis=0)
        median_std = np.median(std_devs)
        
        bad_channels = []
        noisy_channels = []
        
        for i, ch in enumerate(eeg_cols):
            if std_devs[i] < 0.1: # Flatline / Mati
                bad_channels.append(ch)
            elif std_devs[i] > (median_std * 5): # Noise berlebih (Outlier dibanding channel lain)
                noisy_channels.append(ch)

        if bad_channels:
            print(f"    [-] BAHAYA: Sensor mati terdeteksi: {bad_channels}")
            score -= 30
        if noisy_channels:
            print(f"    [-] PERINGATAN: Sensor sangat bising (Outlier): {noisy_channels}. (Mungkin kurang gel/saline).")
            score -= 10
        if not bad_channels and not noisy_channels:
            print("    [+] LULUS: Semua sensor memiliki fluktuasi sinyal yang sehat dan seimbang.")

        # ==========================================
        # 4. UJI LONJAKAN EKSTREM (Extreme Spikes)
        # ==========================================
        print("\n[4] Uji Lonjakan Sinyal Ekstrem (> ±200 µV)...")
        # Kurangi DC Offset rata-rata terlebih dahulu agar sinyal berada di titik 0
        eeg_centered = eeg_data - np.mean(eeg_data, axis=0)
        extreme_spikes = np.sum(np.abs(eeg_centered) > 200)
        spike_ratio = (extreme_spikes / eeg_data.size) * 100
        
        print(f"    -> Rasio Lonjakan Ekstrem: {spike_ratio:.3f}% dari total data.")
        if spike_ratio > 5.0:
            print("    [-] PERINGATAN: Terlalu banyak lonjakan ekstrem! Headset mungkin sering tergeser.")
            score -= 15
        else:
            print("    [+] LULUS: Sinyal sangat stabil, tidak ada anomali listrik berlebih.")

        # ==========================================
        # 5. UJI RETENSI ARTEFAK (Kualitas Data Bersih)
        # ==========================================
        print("\n[5] Uji Simulasi Filter & Pemotongan (Batas Moderat ±100 µV)...")
        filtered_eeg = self.processor.apply_filter(eeg_data)
        marker_indices = df.index[df[marker_col] > 0].tolist()
        
        total_expected_windows = 0
        total_clean_windows = 0
        
        for idx in marker_indices:
            slot_data = filtered_eeg[idx : idx + (5 * self.processor.fs)]
            if len(slot_data) >= 5 * self.processor.fs:
                total_expected_windows += 5
                clean_windows = self.processor.windowing_slot(slot_data)
                total_clean_windows += len(clean_windows)
                
        retention_rate = (total_clean_windows / total_expected_windows) * 100 if total_expected_windows > 0 else 0
        
        print(f"    -> Total Jendela Diekstrak : {total_expected_windows}")
        print(f"    -> Jendela Lulus Uji       : {total_clean_windows}")
        print(f"    -> TINGKAT RETENSI BERSIH  : {retention_rate:.2f}%")
        
        if retention_rate < 60:
            score -= 20
        elif retention_rate < 80:
            score -= 5

        # ==========================================
        # 6. RAPOR KESIMPULAN (FINAL GRADING)
        # ==========================================
        print("\n" + "="*60)
        print(f" HASIL AKHIR KUALITAS DATA (SKOR: {score}/100) ")
        print("="*60)
        
        if score >= 90:
            print("[GRADE A] EXCELLENT! ✨")
            print("Data ini sangat sempurna. Bebas dari noise berlebih, marker akurat, dan sangat direkomendasikan untuk melatih EEGNet!")
            print("=> TINDAKAN: Pindahkan file ini ke folder utama 'dataset/' untuk ditraining.")
        elif score >= 75:
            print("[GRADE B] GOOD! 👍")
            print("Data memiliki sedikit artefak (misal kedipan mata), namun masih sangat layak untuk masuk ke dalam dataset.")
            print("=> TINDAKAN: Pindahkan file ini ke folder utama 'dataset/' untuk ditraining.")
        elif score >= 50:
            print("[GRADE C] WARNING! ⚠️")
            print("Kualitas data meragukan. Fitur AI mungkin kesulitan belajar dari data ini.")
            print("=> TINDAKAN: Sangat disarankan untuk MENGULANG pengambilan data jika memungkinkan.")
        else:
            print("[GRADE F] REJECTED! ❌")
            print("DATA RUSAK PARAH! Terdapat sensor mati, koneksi terputus, atau noise luar biasa.")
            print("=> TINDAKAN: JANGAN GUNAKAN FILE INI! Hapus file dan ulang perekaman.")
            
        print("="*60 + "\n")

if __name__ == "__main__":
    # Path absolut menuju folder dataset/raw
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    TARGET_DIR = os.path.join(BASE_DIR, 'dataset', 'raw')
    
    # 1. Pastikan folder raw/ ada, jika belum buat otomatis
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"\n[!] Folder target baru telah dibuat: {TARGET_DIR}")
        print("    Silakan letakkan file hasil export .csv EmotivPRO ke dalam folder 'raw' tersebut.")
        print("    Lalu jalankan ulang skrip ini.")
        sys.exit()
        
    # 2. Cari semua file CSV di dalam folder raw/
    csv_files = glob.glob(os.path.join(TARGET_DIR, "*.csv"))
    
    if not csv_files:
        print(f"\n[-] GAGAL: Tidak ada file .csv yang ditemukan di dalam folder:")
        print(f"    {TARGET_DIR}")
        print("    Silakan letakkan file hasil export EmotivPRO ke folder tersebut terlebih dahulu.")
        sys.exit()
        
    # 3. Jika ada lebih dari 1 file CSV, ambil yang paling terbaru di-copy/dimodifikasi
    latest_csv = max(csv_files, key=os.path.getmtime)
    
    if len(csv_files) > 1:
        print(f"\n[!] Ditemukan {len(csv_files)} file CSV di folder raw/.")
        print(f"    Menganalisis file yang paling BARU: {os.path.basename(latest_csv)}")
        
    # 4. Jalankan Quality Checker
    try:
        checker = DataQualityChecker(latest_csv)
        checker.run_qc()
    except Exception as e:
        print(f"\n[X] Terjadi kesalahan kritis saat memproses data: {e}")
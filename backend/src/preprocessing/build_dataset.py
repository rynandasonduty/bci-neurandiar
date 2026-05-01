import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Impor dari root backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment
from preprocessing.signal_processor import SignalProcessor

SYLLABLE_CLASSES = {
    "MA": 0, "KAN": 1, "MI": 2, "NUM": 3, "BE": 4, "RAK": 5,
    "PI": 6, "PIS": 7, "MAN": 8, "DI": 9, "BO": 10, "SAN": 11,
    "LE": 12, "LAH": 13, "SA": 14, "KIT": 15, "TI": 16, "DUR": 17, "YANG": 18
}

class DatasetBuilder:
    def __init__(self, exp_id="E0_Baseline", processor_params=None, 
                 crop_time=None, use_augmentation=False, augmentation_params=None,
                 phase_filter="all", channels_to_use="all"):
        """
        - exp_id: Penanda eksperimen (E0_Baseline, E2_ICA, dll).
        - processor_params: Dict untuk mengatur bandpass, ICA, target_fs di SignalProcessor.
        - crop_time: Tuple (start_ms, end_ms). Jika None, gunakan windowing baseline 5 detik.
        - phase_filter: 'all', 'overt', atau 'imagined' (Untuk Eksperimen 6).
        - channels_to_use: 'all' atau list nama channel (Untuk Eksperimen 4).
        """
        print(f"\n[*] MENGINISIALISASI DATASET BUILDER UNTUK EKSPERIMEN: {exp_id}")
        
        # 1. Setup Direktori Dinamis dari config.py
        self.paths = setup_experiment(exp_id)
        self.raw_data_dir = self.paths["raw_data"]
        self.output_dir = self.paths["processed_data"]
        self.scaler_dir = self.paths["scalers"]
        
        # 2. Parameter Eksperimen
        self.crop_time = crop_time
        self.use_augmentation = use_augmentation
        self.augmentation_params = augmentation_params if augmentation_params else {}
        self.phase_filter = phase_filter.lower()
        
        # 3. Inisialisasi Signal Processor
        if processor_params is None:
            processor_params = {}
        self.processor = SignalProcessor(**processor_params)
        
        # 4. Filter Channel (Eksperimen 4: Channel Ablation)
        self.all_channels = self.processor.eeg_channels
        if channels_to_use == "all":
            self.selected_channels = self.all_channels
        else:
            self.selected_channels = [ch for ch in channels_to_use if ch in self.all_channels]
            print(f"[*] Channel Ablation Aktif. Hanya menggunakan: {self.selected_channels}")
            
        # Simpan indeks channel yang dipilih untuk slicing nanti
        self.channel_indices = [self.all_channels.index(ch) for ch in self.selected_channels]

    def parse_log_for_word_sequence(self, log_filepath):
        """Mengekstrak urutan kata dan FASE (Overt/Imagined) dari file log."""
        sequence = []
        with open(log_filepath, 'r') as file:
            for line in file:
                if "Menjalankan Trial" in line and "Kata:" in line:
                    # Contoh line: "Menjalankan Trial 1/100 (Blok 1) - Kata: Makan (Fase: overt)"
                    # PENTING: Struktur regex/split bergantung pada format log dari experiment_runner.py
                    # Asumsi format log menyimpan fase. Jika tidak, butuh adaptasi logika di sini.
                    try:
                        word = line.split("Kata: ")[1].split("(")[0].strip().upper()
                        # Jika log tidak mencatat fase eksplisit, asumsikan 100 pertama overt, 100 kedua imagined
                        phase = "overt" if "overt" in line.lower() else "imagined"
                        sequence.append({"word": word, "phase": phase})
                    except Exception:
                        pass
        return sequence

    def process_subject(self, subject_id, csv_filepath, log_filepath):
        print(f"[*] Memproses data subjek: {subject_id}")
        
        trial_sequence = self.parse_log_for_word_sequence(log_filepath)
        
        # Smart CSV Loader
        header_idx = 0
        with open(csv_filepath, 'r') as f:
            for i, line in enumerate(f):
                if 'EEG.AF3' in line or 'AF3' in line:
                    header_idx = i; break
        
        df = pd.read_csv(csv_filepath, header=header_idx, low_memory=False)
        try:
            float(df.iloc[0][self.all_channels[0]])
        except (ValueError, TypeError):
            df = df.iloc[1:].reset_index(drop=True)
            
        for col in self.all_channels:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        marker_col = 'MarkerValueInt' if 'MarkerValueInt' in df.columns else 'Marker'
        if marker_col not in df.columns:
            return [], []

        df[marker_col] = pd.to_numeric(df[marker_col], errors='coerce').fillna(0)

        # Proses Sinyal Dinamis
        eeg_data = df[self.all_channels].values
        filtered_eeg = self.processor.apply_filter(eeg_data)
        
        marker_indices = df.index[df[marker_col] > 0].tolist()
        
        X_clean_windows = [] 
        y_labels = [] 
               
        for i, idx in enumerate(marker_indices):
            marker_value = int(df.iloc[idx][marker_col])
            if marker_value < 1 or marker_value > 19: 
                continue
            
            # --- EKSPERIMEN 6: CROSS-MODALITY PHASE FILTERING ---
            # Cari tahu trial ini overt atau imagined (asumsi 1 kata = 2 marker berurutan)
            trial_idx = i // 2 
            if trial_idx < len(trial_sequence):
                trial_phase = trial_sequence[trial_idx]["phase"]
                if self.phase_filter != "all" and trial_phase != self.phase_filter:
                    continue # Lewati jika bukan fase yang dicari
                
            label_int = marker_value - 1
            slot_data = filtered_eeg[idx : idx + (5 * self.processor.fs)]
            
            # --- EKSPERIMEN 3: ERP CROPPING vs BASELINE WINDOWING ---
            if self.crop_time is not None:
                # Mode Eksperimen: Potong berdasarkan milidetik
                clean_windows = self.processor.extract_erp_window(
                    slot_data, start_ms=self.crop_time[0], end_ms=self.crop_time[1]
                )
            else:
                # Mode Baseline: Potong 5 jendela 1 detik
                clean_windows = self.processor.windowing_slot(slot_data)
            
            for window in clean_windows:
                # --- EKSPERIMEN 4: CHANNEL ABLATION ---
                # Hanya simpan kolom sensor yang diminta
                window_selected = window[:, self.channel_indices]
                
                # --- EKSPERIMEN 5: AUGMENTATION (JITTER + NOISE) ---
                if self.use_augmentation:
                    aug_window = self.processor.apply_augmentation(window_selected, **self.augmentation_params)
                    X_clean_windows.append(aug_window)
                    y_labels.append(label_int)
                    # Opsi: Sertakan juga versi asli tanpa augmentasi untuk memperbanyak data
                    X_clean_windows.append(window_selected)
                    y_labels.append(label_int)
                else:
                    X_clean_windows.append(window_selected)
                    y_labels.append(label_int)
                
        # Normalisasi Skala & Penyimpanan Scaler Spesifik Eksperimen
        if len(X_clean_windows) == 0:
            return [], []
            
        X_subj_array = np.array(X_clean_windows)
        N, T, C = X_subj_array.shape
        X_subj_2d = X_subj_array.reshape(-1, C)
        
        scaler = StandardScaler()
        X_subj_scaled_2d = scaler.fit_transform(X_subj_2d)
        X_subj_scaled = X_subj_scaled_2d.reshape(N, T, C)
        
        # Simpan scaler ke folder eksperimen yang bersangkutan
        scaler_path = os.path.join(self.scaler_dir, f"{subject_id}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        return X_subj_scaled.tolist(), y_labels

    def build_full_dataset(self, return_data=False):
        all_X, all_y = [], []
        log_files = glob.glob(os.path.join(self.raw_data_dir, "logs", "*_experiment_log.txt"))
        
        for log_path in log_files:
            filename = os.path.basename(log_path)
            subject_id = filename.replace("_experiment_log.txt", "")
            
            csv_files = glob.glob(os.path.join(self.raw_data_dir, f"{subject_id}*.csv"))
            if not csv_files: continue
                
            X_subj, y_subj = self.process_subject(subject_id, csv_files[0], log_path)
            all_X.extend(X_subj)
            all_y.extend(y_subj)
            
        X_tensor = np.array(all_X)
        y_tensor = np.array(all_y)
        
        # Simpan ke folder spesifik eksperimen
        np.save(os.path.join(self.output_dir, "X_features.npy"), X_tensor)
        np.save(os.path.join(self.output_dir, "y_labels.npy"), y_tensor)
        print(f"[SUCCESS] {len(y_tensor)} data diekstrak. Disimpan di: {self.output_dir}")
        
        if return_data:
            return X_tensor, y_tensor

# Testing blok
if __name__ == "__main__":
    builder = DatasetBuilder(exp_id="E0_Test")
    builder.build_full_dataset()
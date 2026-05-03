import os
import sys
import glob
import numpy as np
import pandas as pd

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
        
        # Direktori scaler dipertahankan di config, namun tidak digunakan di sini untuk menghindari data leakage
        
        # 2. Parameter Eksperimen
        self.crop_time = crop_time
        # Parameter augmentasi disimpan namun tidak dieksekusi di sini (ditunda ke fase post-split)
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
                    try:
                        word = line.split("Kata: ")[1].split("(")[0].strip().upper()
                        
                        # FIX: Logika deteksi fase yang lebih ketat untuk menghindari silent failure
                        if "overt" in line.lower():
                            phase = "overt"
                        elif "imagined" in line.lower():
                            phase = "imagined"
                        else:
                            print(f"[!] WARNING: Fase tidak terdeteksi secara eksplisit di baris log: {line.strip()}")
                            phase = "unknown"
                            
                        sequence.append({"word": word, "phase": phase})
                    except Exception:
                        pass
        return sequence

    def process_subject(self, subject_id, csv_filepath, log_filepath):
        print(f"[*] Memproses data mentah subjek: {subject_id}")
        
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

        # Proses Sinyal Dinamis (Bandpass, dll)
        eeg_data = df[self.all_channels].values
        filtered_eeg = self.processor.apply_filter(eeg_data)
        
        marker_indices = df.index[df[marker_col] > 0].tolist()
        
        X_clean_windows = [] 
        y_labels = [] 
        
        # FIX: Sinkronisasi marker menggunakan penghitung marker valid murni
        valid_marker_count = 0
               
        for idx in marker_indices:
            marker_value = int(df.iloc[idx][marker_col])
            
            # Abaikan marker yang tidak relevan dengan suku kata target
            if marker_value < 1 or marker_value > 19: 
                continue
            
            # --- EKSPERIMEN 6: CROSS-MODALITY PHASE FILTERING ---
            # Asumsi: 1 kata terdiri dari 2 marker suku kata.
            trial_idx = valid_marker_count // 2 
            if trial_idx < len(trial_sequence):
                trial_phase = trial_sequence[trial_idx]["phase"]
            else:
                trial_phase = "unknown"
                
            valid_marker_count += 1
            
            # Lewati ekstraksi jika fase tidak sesuai dengan resep eksperimen
            if self.phase_filter != "all" and trial_phase != self.phase_filter:
                continue 
                
            label_int = marker_value - 1
            slot_data = filtered_eeg[idx : idx + (5 * self.processor.fs)]
            
            # --- EKSPERIMEN 3: ERP CROPPING vs BASELINE WINDOWING ---
            if self.crop_time is not None:
                # Mode Eksperimen: Potong berdasarkan milidetik untuk N400
                clean_windows = self.processor.extract_erp_window(
                    slot_data, start_ms=self.crop_time[0], end_ms=self.crop_time[1]
                )
            else:
                # Mode Baseline: Potong 5 jendela 1 detik (Data Augmentation Temporal Natural)
                clean_windows = self.processor.windowing_slot(slot_data)
            
            for window in clean_windows:
                # --- EKSPERIMEN 4: CHANNEL ABLATION ---
                window_selected = window[:, self.channel_indices]
                
                # FIX: Augmentasi (Eksperimen 5) dihapus dari fase ini untuk mencegah Data Leakage.
                # Data diserahkan murni, augmentasi hanya akan diinjeksikan pada X_train di pipeline pelatihan.
                X_clean_windows.append(window_selected)
                y_labels.append(label_int)
                
        if len(X_clean_windows) == 0:
            return [], []
            
        X_subj_array = np.array(X_clean_windows)
        
        # FIX: Penghapusan StandardScaler dari sini.
        # Normalisasi wajib dilakukan HANYA setelah split train/val/test di data_utils.py
        return X_subj_array.tolist(), y_labels

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
        
        # Simpan tensor murni ke folder spesifik eksperimen
        np.save(os.path.join(self.output_dir, "X_features.npy"), X_tensor)
        np.save(os.path.join(self.output_dir, "y_labels.npy"), y_tensor)
        print(f"[SUCCESS] {len(y_tensor)} data diekstrak. Disimpan di: {self.output_dir}")
        
        if return_data:
            return X_tensor, y_tensor

# Testing blok
if __name__ == "__main__":
    builder = DatasetBuilder(exp_id="E0_Test")
    builder.build_full_dataset()
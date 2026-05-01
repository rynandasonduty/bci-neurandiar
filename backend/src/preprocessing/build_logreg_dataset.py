import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Impor dari root backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment
from preprocessing.signal_processor import SignalProcessor

WORD_CLASSES = {
    "MAKAN": 0, "MINUM": 1, "BERAK": 2, "PIPIS": 3, "MANDI": 4,
    "BOSAN": 5, "LELAH": 6, "SAKIT": 7, "TIDUR": 8, "SAYANG": 9
}

class LogRegDatasetBuilder:
    def __init__(self, exp_id="E0_Baseline", processor_params=None, 
                 crop_time=None, phase_filter="all", channels_to_use="all"):
        
        print(f"\n[*] MENGINISIALISASI LOGREG BUILDER UNTUK EKSPERIMEN: {exp_id}")
        
        # 1. Setup Direktori Dinamis
        self.paths = setup_experiment(exp_id)
        self.raw_data_dir = self.paths["raw_data"]
        self.output_dir = self.paths["processed_data"]
        self.scaler_dir = self.paths["scalers"]
        self.weights_dir = self.paths["weights"]
        
        # 2. Parameter Eksperimen
        self.crop_time = crop_time
        self.phase_filter = phase_filter.lower()
        
        # 3. Inisialisasi Signal Processor
        if processor_params is None:
            processor_params = {}
        self.processor = SignalProcessor(**processor_params)
        
        # 4. Filter Channel
        self.all_channels = self.processor.eeg_channels
        if channels_to_use == "all":
            self.selected_channels = self.all_channels
        else:
            self.selected_channels = [ch for ch in channels_to_use if ch in self.all_channels]
        self.channel_indices = [self.all_channels.index(ch) for ch in self.selected_channels]

        # 5. Load EEGNet Model SPESIFIK untuk Eksperimen ini
        model_path = os.path.join(self.weights_dir, f"eegnet_trained_{exp_id}.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[!] {model_path} tidak ditemukan. Latih EEGNet untuk {exp_id} terlebih dahulu!")
            
        print(f"[*] Memuat model EEGNet: {model_path}")
        self.eegnet = load_model(model_path)

    def parse_log_for_word_sequence(self, log_filepath):
        sequence = []
        with open(log_filepath, 'r') as file:
            for line in file:
                if "Menjalankan Trial" in line and "Kata:" in line:
                    try:
                        word = line.split("Kata: ")[1].split("(")[0].strip().upper()
                        phase = "overt" if "overt" in line.lower() else "imagined"
                        sequence.append({"word": word, "phase": phase})
                    except Exception:
                        pass
        return sequence

    def extract_probabilities(self, clean_windows, scaler):
        if len(clean_windows) == 0:
            return None
            
        windows_array = np.array(clean_windows)
        
        # Hanya gunakan channel yang dipilih
        windows_selected = windows_array[:, :, self.channel_indices]
        
        N, T, C = windows_selected.shape
        windows_2d = windows_selected.reshape(-1, C)
        
        try:
            windows_scaled_2d = scaler.transform(windows_2d)
        except Exception as e:
            print(f"[X] Gagal Scaling: {e}. Pastikan scaler cocok dengan channel ablation.")
            return None
            
        windows_scaled = windows_scaled_2d.reshape(N, T, C)
        
        X_input = np.transpose(windows_scaled, (0, 2, 1))
        X_input = np.expand_dims(X_input, axis=3)
        
        probs = self.eegnet.predict(X_input, verbose=0) 
        averaged_prob = np.mean(probs, axis=0)
        return averaged_prob

    def build_dataset(self):
        print("\n" + "="*50)
        print(" MEMBANGUN DATASET UNTUK REGRESI LOGISTIK ")
        print("="*50)
        
        X_word_features = []
        y_word_labels = []
        
        log_files = glob.glob(os.path.join(self.raw_data_dir, "logs", "*_experiment_log.txt"))
        
        for log_path in log_files:
            filename = os.path.basename(log_path)
            subject_id = filename.replace("_experiment_log.txt", "")
            
            csv_files = glob.glob(os.path.join(self.raw_data_dir, f"{subject_id}*.csv"))
            if not csv_files: continue
            
            scaler_path = os.path.join(self.scaler_dir, f"{subject_id}_scaler.pkl")
            if not os.path.exists(scaler_path): continue
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                
            trial_sequence = self.parse_log_for_word_sequence(log_path)
            
            header_idx = 0
            with open(csv_files[0], 'r') as f:
                for i, line in enumerate(f):
                    if 'EEG.AF3' in line or 'AF3' in line:
                        header_idx = i; break
            
            df = pd.read_csv(csv_files[0], header=header_idx, low_memory=False)
            try:
                float(df.iloc[0][self.all_channels[0]])
            except (ValueError, TypeError):
                df = df.iloc[1:].reset_index(drop=True)
                
            for col in self.all_channels:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            marker_col = 'MarkerValueInt' if 'MarkerValueInt' in df.columns else 'Marker'
            df[marker_col] = pd.to_numeric(df[marker_col], errors='coerce').fillna(0)
            
            filtered_eeg = self.processor.apply_filter(df[self.all_channels].values)
            marker_indices = df.index[df[marker_col] > 0].tolist()
            
            # Asumsi: Setiap 2 marker adalah Slot 1 dan Slot 2
            valid_trials = 0
            for i in range(0, len(marker_indices)-1, 2):
                idx_slot1 = marker_indices[i]
                idx_slot2 = marker_indices[i+1]
                
                trial_index = i // 2
                if trial_index >= len(trial_sequence): break
                    
                target_word = trial_sequence[trial_index]["word"]
                trial_phase = trial_sequence[trial_index]["phase"]
                
                if target_word not in WORD_CLASSES: continue
                if self.phase_filter != "all" and trial_phase != self.phase_filter: continue
                    
                data_slot1 = filtered_eeg[idx_slot1 : idx_slot1 + (5 * self.processor.fs)]
                data_slot2 = filtered_eeg[idx_slot2 : idx_slot2 + (5 * self.processor.fs)]
                
                if self.crop_time is not None:
                    clean_win1 = self.processor.extract_erp_window(data_slot1, self.crop_time[0], self.crop_time[1])
                    clean_win2 = self.processor.extract_erp_window(data_slot2, self.crop_time[0], self.crop_time[1])
                else:
                    clean_win1 = self.processor.windowing_slot(data_slot1)
                    clean_win2 = self.processor.windowing_slot(data_slot2)
                
                p1 = self.extract_probabilities(clean_win1, scaler)
                p2 = self.extract_probabilities(clean_win2, scaler)
                
                if p1 is not None and p2 is not None:
                    phi_features = np.concatenate((p1, p2))
                    X_word_features.append(phi_features)
                    y_word_labels.append(WORD_CLASSES[target_word])
                    valid_trials += 1
                    
            print(f"[+] Subjek {subject_id} diproses. (Ekstrak {valid_trials} trial kata)")

        if not X_word_features:
            print("[!] Peringatan: Tidak ada data kata yang berhasil diekstrak.")
            return

        X_tensor = np.array(X_word_features)
        y_tensor = np.array(y_word_labels)
        
        np.save(os.path.join(self.output_dir, "X_word_features.npy"), X_tensor)
        np.save(os.path.join(self.output_dir, "y_word_labels.npy"), y_tensor)
        print(f"[SUCCESS] Dataset Regresi Logistik disimpan di: {self.output_dir}")

if __name__ == "__main__":
    builder = LogRegDatasetBuilder(exp_id="E0_Test")
    builder.build_dataset()
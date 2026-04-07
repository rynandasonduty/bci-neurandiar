import os
import sys
import pandas as pd
import glob
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from signal_processor import SignalProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))


# Pemetaan 10 Kata Target ke Kelas Integer (0-9)
WORD_CLASSES = {
    "MAKAN": 0, "MINUM": 1, "BERAK": 2, "PIPIS": 3, "MANDI": 4,
    "BOSAN": 5, "LELAH": 6, "SAKIT": 7, "TIDUR": 8, "SAYANG": 9
}

class LogRegDatasetBuilder:
    def __init__(self, raw_data_dir="../../dataset", output_dir="../../dataset/processed", model_dir="../../dataset/models"):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.scaler_dir = os.path.join(raw_data_dir, "scalers")
        
        self.processor = SignalProcessor()
        
        # Memuat model EEGNet yang sudah dilatih
        model_path = os.path.join(model_dir, "eegnet_trained.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[!] {model_path} tidak ditemukan. Latih EEGNet terlebih dahulu!")
            
        print("[*] Memuat model EEGNet yang telah dilatih...")
        self.eegnet = load_model(model_path)
        print("[+] Model EEGNet berhasil dimuat.")

    def parse_log_for_word_sequence(self, log_filepath):
        word_sequence = []
        with open(log_filepath, 'r') as file:
            for line in file:
                if "Menjalankan Trial" in line and "Kata:" in line:
                    word = line.split("Kata: ")[1].split(" (Fase")[0].strip().upper()
                    word_sequence.append(word)
        return word_sequence

    def extract_probabilities(self, clean_windows, scaler):
        """Memasukkan 5 jendela ke EEGNet dan merata-ratakan probabilitasnya"""
        if len(clean_windows) == 0:
            return None
            
        # Bentuk awal: (5 Jendela, 256 Titik, 14 Channel)
        windows_array = np.array(clean_windows)
        N, T, C = windows_array.shape
        
        # Standarisasi menggunakan Scaler khusus subjek ini
        windows_2d = windows_array.reshape(-1, C)
        windows_scaled_2d = scaler.transform(windows_2d)
        windows_scaled = windows_scaled_2d.reshape(N, T, C)
        
        # Ubah bentuk untuk input EEGNet Keras: (Samples, Channels, Time, Depth)
        X_input = np.transpose(windows_scaled, (0, 2, 1))
        X_input = np.expand_dims(X_input, axis=3)
        
        # Prediksi 5 jendela
        probs = self.eegnet.predict(X_input, verbose=0) # Output shape: (5, 19)
        
        # Rata-ratakan 5 probabilitas tersebut menjadi 1 vektor stabil (19,)
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
            
            # Load Scaler Subjek
            scaler_path = os.path.join(self.scaler_dir, f"{subject_id}_scaler.pkl")
            if not os.path.exists(scaler_path): continue
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                
            word_sequence = self.parse_log_for_word_sequence(log_path)
            
            # --- MULAI SMART CSV LOADER ---
            header_idx = 0
            with open(csv_files[0], 'r') as f:
                for i, line in enumerate(f):
                    if 'EEG.AF3' in line or 'AF3' in line:
                        header_idx = i
                        break
            
            df = pd.read_csv(csv_files[0], header=header_idx, low_memory=False)
            
            try:
                float(df.iloc[0][self.processor.eeg_channels[0]])
            except (ValueError, TypeError):
                df = df.iloc[1:].reset_index(drop=True)
                
            for col in self.processor.eeg_channels:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # --- SELESAI SMART CSV LOADER ---
            
            marker_col = 'MarkerValueInt' if 'MarkerValueInt' in df.columns else 'Marker'
            df[marker_col] = pd.to_numeric(df[marker_col], errors='coerce').fillna(0)
            
            filtered_eeg = self.processor.apply_filter(df[self.processor.eeg_channels].values)
                                   
            # Kita proses secara berpasangan (Slot 1 dan Slot 2) untuk 1 Trial Kata
            marker_indices = df.index[df[marker_col] > 0].tolist()
            
            # Asumsi: Setiap 2 marker berurutan adalah Slot 1 dan Slot 2 dari 1 Trial
            for i in range(0, len(marker_indices)-1, 2):
                idx_slot1 = marker_indices[i]
                idx_slot2 = marker_indices[i+1]
                
                trial_index = i // 2
                if trial_index >= len(word_sequence): break
                    
                target_word = word_sequence[trial_index]
                if target_word not in WORD_CLASSES: continue
                    
                # Ekstrak Jendela
                data_slot1 = filtered_eeg[idx_slot1 : idx_slot1 + (5 * self.processor.fs)]
                data_slot2 = filtered_eeg[idx_slot2 : idx_slot2 + (5 * self.processor.fs)]
                
                clean_win1 = self.processor.windowing_slot(data_slot1)
                clean_win2 = self.processor.windowing_slot(data_slot2)
                
                # Masukkan ke EEGNet -> Dapatkan Probabilitas
                p1 = self.extract_probabilities(clean_win1, scaler)
                p2 = self.extract_probabilities(clean_win2, scaler)
                
                if p1 is not None and p2 is not None:
                    # Gabungkan (Concatenate) P1 dan P2 menjadi 38 dimensi
                    phi_features = np.concatenate((p1, p2))
                    
                    X_word_features.append(phi_features)
                    y_word_labels.append(WORD_CLASSES[target_word])
                    
            print(f"[+] Subjek {subject_id} diproses. (Mengekstrak {len(X_word_features)} trial kata berjalan)")

        X_tensor = np.array(X_word_features)
        y_tensor = np.array(y_word_labels)
        
        print("\n" + "="*50)
        print(" EKSTRAKSI FITUR KATA SELESAI ")
        print("="*50)
        print(f"Bentuk Fitur X (Word) : {X_tensor.shape} -> (Samples, 38 Probabilitas)")
        print(f"Bentuk Label y (Word) : {y_tensor.shape} -> (Samples,)")
        
        np.save(os.path.join(self.output_dir, "X_word_features.npy"), X_tensor)
        np.save(os.path.join(self.output_dir, "y_word_labels.npy"), y_tensor)
        print(f"[SUCCESS] Dataset Regresi Logistik disimpan di: {self.output_dir}/")

if __name__ == "__main__":
    builder = LogRegDatasetBuilder()
    builder.build_dataset()
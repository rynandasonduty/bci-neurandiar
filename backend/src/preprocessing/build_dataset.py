import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from signal_processor import SignalProcessor

# Pemetaan 19 Suku Kata ke Kelas Integer (0-18)
SYLLABLE_CLASSES = {
    "MA": 0, "KAN": 1, "MI": 2, "NUM": 3, "BE": 4, "RAK": 5,
    "PI": 6, "PIS": 7, "MAN": 8, "DI": 9, "BO": 10, "SAN": 11,
    "LE": 12, "LAH": 13, "SA": 14, "KIT": 15, "TI": 16, "DUR": 17, "YANG": 18
}

WORD_TO_SYLLABLES = {
    "MAKAN": ("MA", "KAN"), "MINUM": ("MI", "NUM"), "BERAK": ("BE", "RAK"),
    "PIPIS": ("PI", "PIS"), "MANDI": ("MAN", "DI"), "BOSAN": ("BO", "SAN"),
    "LELAH": ("LE", "LAH"), "SAKIT": ("SA", "KIT"), "TIDUR": ("TI", "DUR"),
    "SAYANG": ("SA", "YANG")
}

class DatasetBuilder:
    def __init__(self, raw_data_dir="../../dataset", output_dir="../../dataset/processed"):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        
        # Folder MLOps untuk menyimpan Scaler per subjek
        self.scaler_dir = os.path.join(raw_data_dir, "scalers")
        
        self.processor = SignalProcessor()
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.scaler_dir, exist_ok=True)

    def parse_log_for_word_sequence(self, log_filepath):
        word_sequence = []
        with open(log_filepath, 'r') as file:
            for line in file:
                if "Menjalankan Trial" in line and "Kata:" in line:
                    word = line.split("Kata: ")[1].split(" (Fase")[0].strip().upper()
                    word_sequence.append(word)
        return word_sequence

    def process_subject(self, subject_id, csv_filepath, log_filepath):
        print(f"[*] Memproses data subjek: {subject_id}")
        
        word_sequence = self.parse_log_for_word_sequence(log_filepath)
        df = pd.read_csv(csv_filepath)
        
        marker_col = 'MarkerValueInt' if 'MarkerValueInt' in df.columns else 'Marker'
        if marker_col not in df.columns:
            print(f"[X] Kolom marker tidak ditemukan di {csv_filepath}")
            return [], []

        # Terapkan Band-Pass Filter (Sinyal masih dalam skala Mikrovolt)
        eeg_data = df[self.processor.eeg_channels].values
        filtered_eeg = self.processor.apply_filter(eeg_data)
        
        marker_indices = df.index[df[marker_col] > 0].tolist()
        
        X_clean_windows = [] 
        y_labels = [] 
        
        for i, idx in enumerate(marker_indices):
            marker_value = int(df.iloc[idx][marker_col])
            
            # Validasi: Pastikan marker adalah ID Suku Kata (1 sampai 19)
            if marker_value < 1 or marker_value > 19: 
                continue
                
            # Keras/TensorFlow membutuhkan label dimulai dari 0 (0 sampai 18)
            # Karena LSL Marker kita mengirim 1-19, kita kurangi 1.
            label_int = marker_value - 1
            
            # Potong slot 5 detik
            slot_data = filtered_eeg[idx : idx + (5 * self.processor.fs)]
            
            # Windowing & Artifact Rejection (Batas +- 100 uV dieksekusi di sini)
            clean_windows = self.processor.windowing_slot(slot_data)
            
            for window in clean_windows:
                X_clean_windows.append(window)
                y_labels.append(label_int)
                
        # =======================================================
        # FASE MLOPS: STANDARD SCALER (Z-Score Normalization)
        # =======================================================
        if len(X_clean_windows) == 0:
            return [], []
            
        # Bentuk matriks: (Jumlah Jendela, 256 Titik Waktu, 14 Sensor)
        X_subj_array = np.array(X_clean_windows)
        N, T, C = X_subj_array.shape
        
        # Scikit-learn Scaler hanya menerima data 2D, kita ratakan sementara
        X_subj_2d = X_subj_array.reshape(-1, C)
        
        scaler = StandardScaler()
        X_subj_scaled_2d = scaler.fit_transform(X_subj_2d)
        
        # Kembalikan bentuknya ke 3D
        X_subj_scaled = X_subj_scaled_2d.reshape(N, T, C)
        
        # Simpan objek scaler untuk subjek ini agar bisa dipanggil saat Live Inference
        scaler_path = os.path.join(self.scaler_dir, f"{subject_id}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        print(f"[+] Scaler disimpan: {subject_id}_scaler.pkl")
        return X_subj_scaled.tolist(), y_labels

    def build_full_dataset(self):
        all_X, all_y = [], []
        log_files = glob.glob(os.path.join(self.raw_data_dir, "logs", "*_experiment_log.txt"))
        
        for log_path in log_files:
            filename = os.path.basename(log_path)
            subject_id = filename.replace("_experiment_log.txt", "")
            
            csv_pattern = os.path.join(self.raw_data_dir, f"{subject_id}*.csv")
            csv_files = glob.glob(csv_pattern)
            if not csv_files: continue
                
            X_subj, y_subj = self.process_subject(subject_id, csv_files[0], log_path)
            all_X.extend(X_subj)
            all_y.extend(y_subj)
            
        X_tensor = np.array(all_X)
        y_tensor = np.array(all_y)
        
        print("\n" + "="*50)
        print(" EKSTRAKSI & NORMALISASI DATASET SELESAI")
        print("="*50)
        print(f"Bentuk Fitur X : {X_tensor.shape}")
        
        np.save(os.path.join(self.output_dir, "X_features.npy"), X_tensor)
        np.save(os.path.join(self.output_dir, "y_labels.npy"), y_tensor)
        print(f"[+] Dataset siap latih disimpan di: {self.output_dir}/")

if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build_full_dataset()
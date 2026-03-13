import os
import glob
import numpy as np
import pandas as pd
from signal_processor import SignalProcessor

# Pemetaan 19 Suku Kata ke Kelas Integer (0-18) untuk EEGNet
SYLLABLE_CLASSES = {
    "MA": 0, "KAN": 1, "MI": 2, "NUM": 3, "BE": 4, "RAK": 5,
    "PI": 6, "PIS": 7, "MAN": 8, "DI": 9, "BO": 10, "SAN": 11,
    "LE": 12, "LAH": 13, "SA": 14, "KIT": 15, "TI": 16, "DUR": 17, "YANG": 18
}

# Kamus bantuan untuk memetakan Kata ke pasang Suku Katanya
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
        self.processor = SignalProcessor() # Memanggil mesin pembersih sinyal
        
        os.makedirs(self.output_dir, exist_ok=True)

    def parse_log_for_word_sequence(self, log_filepath):
        """
        Membaca file log eksperimen untuk mendapatkan urutan kata yang diuji.
        Mengembalikan daftar (list) kata target secara berurutan.
        """
        word_sequence = []
        with open(log_filepath, 'r') as file:
            for line in file:
                # Mencari baris yang mencatat jalannya trial
                if "Menjalankan Trial" in line and "Kata:" in line:
                    # Mengekstrak kata target dari string log
                    word = line.split("Kata: ")[1].split(" (Fase")[0].strip().upper()
                    word_sequence.append(word)
        return word_sequence

    def process_subject(self, subject_id, csv_filepath, log_filepath):
        """Memproses data satu subjek dari awal hingga akhir."""
        print(f"[*] Memproses data subjek: {subject_id}")
        
        # 1. Dapatkan urutan kata dari Log
        word_sequence = self.parse_log_for_word_sequence(log_filepath)
        
        # 2. Baca file CSV Mentah dari Emotiv
        df = pd.read_csv(csv_filepath)
        
        # Penyesuaian nama kolom marker standar EmotivPRO
        marker_col = 'MarkerValueInt' if 'MarkerValueInt' in df.columns else 'Marker'
        if marker_col not in df.columns:
            print(f"[X] Kolom marker tidak ditemukan di {csv_filepath}")
            return [], []

        # 3. Terapkan Band-Pass Filter (0.5 - 50 Hz) ke seluruh sinyal sekaligus
        eeg_data = df[self.processor.eeg_channels].values
        filtered_eeg = self.processor.apply_filter(eeg_data)
        
        # 4. Cari indeks waktu saat Marker 1 (Slot 1) dan Marker 2 (Slot 2) muncul
        marker_indices = df.index[df[marker_col] > 0].tolist()
        
        X_data = [] # Menyimpan matriks sinyal
        y_labels = [] # Menyimpan label kelas (0-18)
        
        trial_count = 0
        
        # 5. Iterasi melalui setiap marker yang ditemukan
        for i, idx in enumerate(marker_indices):
            marker_value = df.iloc[idx][marker_col]
            
            # Hitung ini berada di trial kata ke-berapa
            trial_index = i // 2 
            if trial_index >= len(word_sequence):
                break
                
            current_word = word_sequence[trial_index]
            syl1, syl2 = WORD_TO_SYLLABLES[current_word]
            
            # Tentukan suku kata mana yang sedang dieksekusi
            if marker_value == 1:
                target_syllable = syl1
            elif marker_value == 2:
                target_syllable = syl2
            else:
                continue # Abaikan marker lain jika ada
                
            label_int = SYLLABLE_CLASSES[target_syllable]
            
            # 6. Potong sinyal 5 detik ke depan
            slot_data = filtered_eeg[idx : idx + (5 * self.processor.fs)]
            
            # 7. Kirim ke SignalProcessor untuk di-windowing (1 detik) & pembersihan artefak
            clean_windows = self.processor.windowing_slot(slot_data)
            
            # 8. Simpan jendela yang lolos validasi beserta labelnya
            for window in clean_windows:
                X_data.append(window)
                y_labels.append(label_int)
                
        return X_data, y_labels

    def build_full_dataset(self):
        """Memproses seluruh subjek dan menggabungkannya menjadi 1 Dataset Tensor"""
        all_X = []
        all_y = []
        
        # Mencari semua file log di folder dataset
        log_files = glob.glob(os.path.join(self.raw_data_dir, "logs", "*_experiment_log.txt"))
        
        for log_path in log_files:
            # Ekstrak ID Subjek dari nama file
            filename = os.path.basename(log_path)
            subject_id = filename.replace("_experiment_log.txt", "")
            
            # Cari file CSV yang sesuai untuk subjek ini
            # (Misal format nama dari Emotiv: SUBJ01_*.csv)
            csv_pattern = os.path.join(self.raw_data_dir, f"{subject_id}*.csv")
            csv_files = glob.glob(csv_pattern)
            
            if not csv_files:
                print(f"[!] CSV untuk {subject_id} tidak ditemukan. Dilewati.")
                continue
                
            csv_path = csv_files[0] # Ambil file CSV pertama yang cocok
            
            # Proses subjek tersebut
            X_subj, y_subj = self.process_subject(subject_id, csv_path, log_path)
            
            all_X.extend(X_subj)
            all_y.extend(y_subj)
            
        # Konversi ke format Numpy Array (Tensor)
        X_tensor = np.array(all_X)
        y_tensor = np.array(all_y)
        
        print("\n" + "="*50)
        print(" EKSTRAKSI DATASET SELESAI")
        print("="*50)
        print(f"Bentuk (Shape) Fitur X : {X_tensor.shape}")
        print(f"Bentuk (Shape) Label y : {y_tensor.shape}")
        
        # Simpan ke file .npy
        np.save(os.path.join(self.output_dir, "X_features.npy"), X_tensor)
        np.save(os.path.join(self.output_dir, "y_labels.npy"), y_tensor)
        print(f"[+] Dataset berhasil disimpan di: {self.output_dir}/")

if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build_full_dataset()
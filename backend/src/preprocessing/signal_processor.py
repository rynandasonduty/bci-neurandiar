import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

class SignalProcessor:
    def __init__(self, fs=256, lowcut=0.5, highcut=50.0, order=4, artifact_threshold=100.0):
        """
        Inisialisasi parameter prapemrosesan sesuai Metodologi Bab 3.
        - fs: Sampling rate Emotiv EPOC X (256 Hz)
        - lowcut & highcut: Rentang Band-Pass Filter (0.5 - 50.0 Hz)
        - order: Orde filter Butterworth (Orde ke-4)
        - artifact_threshold: Batas amplitudo artefak EOG/EMG (±100 µV)
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.artifact_threshold = artifact_threshold
        
        # 14 Saluran (Channels) Emotiv EPOC X sesuai standar 10-20
        self.eeg_channels = [
            "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", 
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
        ]

    def _butter_bandpass(self):
        """Mendesain koefisien filter IIR Butterworth"""
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return b, a

    def apply_filter(self, data):
        """
        Menerapkan Band-Pass Filter 0.5 - 50 Hz pada data EEG.
        Menggunakan filtfilt (Zero-phase filtering) agar tidak terjadi pergeseran sinyal.
        """
        b, a = self._butter_bandpass()
        # Menerapkan filter di sepanjang sumbu waktu (axis=0)
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data

    def reject_artifacts(self, epoch_data):
        """
        Mengecek apakah dalam jendela 1 detik terdapat amplitudo yang melebihi ±100 µV.
        Mengembalikan True jika ada artefak (harus dibuang), False jika bersih.
        """
        max_amplitude = np.max(np.abs(epoch_data))
        if max_amplitude > self.artifact_threshold:
            return True
        return False

    def windowing_slot(self, slot_data):
        """
        Memotong 1 slot rekaman (5 detik) menjadi 5 jendela terpisah (masing-masing 1 detik).
        Satu detik = 256 titik data (karena fs=256 Hz).
        """
        windows = []
        window_size = self.fs * 1  # 256 baris data
        
        # Pastikan data slot cukup untuk 5 detik (5 * 256 = 1280 baris)
        expected_length = 5 * self.fs
        if len(slot_data) < expected_length:
            print(f"[!] Peringatan: Data slot kurang dari 5 detik ({len(slot_data)} baris). Dilewati.")
            return windows
            
        for i in range(5):
            # Memotong per 1 detik (0-256, 256-512, dst)
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_data = slot_data[start_idx:end_idx]
            
            # Pengecekan Artefak Otomatis
            if not self.reject_artifacts(window_data):
                windows.append(window_data)
            else:
                # Jika Anda ingin melihat log data yang dibuang, aktifkan print di bawah:
                # print(f"[-] Artefak terdeteksi pada jendela {i+1}. Jendela dibuang.")
                pass
                
        return windows

    def process_csv_file(self, csv_filepath):
        """
        Fungsi master untuk memproses satu file CSV utuh.
        (Ini akan sangat berguna setelah Anda mendapatkan data dari EmotivPRO nanti).
        """
        # 1. Baca Data
        df = pd.read_csv(csv_filepath)
        
        # Asumsi: Kolom marker bernama 'Marker' (akan disesuaikan nanti dengan format Emotiv)
        if 'Marker' not in df.columns:
            raise ValueError("Kolom 'Marker' tidak ditemukan di CSV.")
            
        # 2. Ambil hanya 14 saluran EEG
        eeg_data = df[self.eeg_channels].values
        
        # 3. Terapkan Band-Pass Filter ke seluruh data kontinu
        filtered_eeg = self.apply_filter(eeg_data)
        
        # 4. Cari index baris di mana Marker 1 (Slot 1) dan Marker 2 (Slot 2) ditekan
        marker_1_indices = df.index[df['Marker'] == 1].tolist()
        marker_2_indices = df.index[df['Marker'] == 2].tolist()
        
        all_clean_epochs = []
        
        # Proses Slot 1
        for idx in marker_1_indices:
            slot_data = filtered_eeg[idx : idx + (5 * self.fs)]
            clean_windows = self.windowing_slot(slot_data)
            all_clean_epochs.extend(clean_windows)
            
        # Proses Slot 2
        for idx in marker_2_indices:
            slot_data = filtered_eeg[idx : idx + (5 * self.fs)]
            clean_windows = self.windowing_slot(slot_data)
            all_clean_epochs.extend(clean_windows)
            
        return np.array(all_clean_epochs)
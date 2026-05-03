import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA

# KAMUS PITA FREKUENSI EEG (Untuk Eksperimen 7)
EEG_BANDS = {
    "broadband": (0.5, 50.0), # Baseline
    "theta": (4.0, 8.0),      # Relaksasi / Mengantuk
    "alpha": (8.0, 13.0),     # Relaksasi sadar / Mata tertutup
    "low_beta": (13.0, 20.0), # Fokus aktif / Imagined Speech
    "high_beta": (20.0, 30.0),# Kewaspadaan tinggi
    "gamma": (30.0, 50.0)     # Pemrosesan kognitif tingkat tinggi
}

class SignalProcessor:
    def __init__(self, fs=256, band="broadband", order=4, artifact_threshold=100.0, 
                 apply_ica=False, target_fs=None):
        """
        - band: Menggunakan preset pita frekuensi dari EEG_BANDS (Eksperimen 7)
        """
        self.fs = fs
        self.order = order
        self.artifact_threshold = artifact_threshold
        
        # Ekstraksi lowcut & highcut dari preset band yang dipilih
        if band in EEG_BANDS:
            self.lowcut, self.highcut = EEG_BANDS[band]
        else:
            self.lowcut, self.highcut = EEG_BANDS["broadband"]
            print(f"[!] Band '{band}' tidak dikenali. Menggunakan broadband 0.5-50Hz.")
        
        # Parameter Eksperimen
        self.apply_ica = apply_ica
        # FIX: Semua ukuran pemotongan dan pergeseran akan bergantung pada target_fs
        self.target_fs = target_fs if target_fs is not None else fs
        
        self.eeg_channels = [
            "EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7", "EEG.P7", "EEG.O1", 
            "EEG.O2", "EEG.P8", "EEG.T8", "EEG.FC6", "EEG.F4", "EEG.F8", "EEG.AF4"
        ]

    # =====================================================================
    # 1. FONDASI BASELINE 
    # =====================================================================

    def _butter_bandpass(self):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return b, a

    def apply_filter(self, data):
        b, a = self._butter_bandpass()
        filtered_data = filtfilt(b, a, data, axis=0)
        
        # [EKSPERIMEN 1] Independent Component Analysis (ICA)
        if self.apply_ica:
            ica = FastICA(n_components=14, random_state=42, max_iter=800)
            sources = ica.fit_transform(filtered_data)
            
            # FIX: Evaluasi neurofisiologis artefak menggunakan Kurtosis
            for i in range(sources.shape[1]):
                component_kurt = kurtosis(sources[:, i])
                # Jika kurtosis tinggi (> 3.0), itu adalah distribusi non-Gaussian (berkemungkinan besar mata/otot)
                if component_kurt > 3.0: 
                    sources[:, i] = 0
                    
            filtered_data = ica.inverse_transform(sources)
            
        return filtered_data

    def reject_artifacts(self, epoch_data):
        # Method ini digunakan SEBELUM Resampling
        max_amplitude = np.max(np.abs(epoch_data))
        if max_amplitude > self.artifact_threshold:
            return True
        return False

    def windowing_slot(self, slot_data):
        windows = []
        # Ukuran window saat ini masih berdasarkan self.fs (sebelum resample)
        window_size = self.fs * 1  
        
        expected_length = 5 * self.fs
        if len(slot_data) < expected_length:
            return windows
            
        for i in range(5):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_data = slot_data[start_idx:end_idx]
            
            if not self.reject_artifacts(window_data):
                # FIX: Aplikasikan Resampling JIKA target_fs berbeda
                if self.target_fs != self.fs:
                    target_length = self.target_fs * 1  # Untuk durasi 1 detik
                    window_data = resample(window_data, target_length, axis=0)
                windows.append(window_data)
                
        return windows

    def process_csv_file(self, csv_filepath):
        df = pd.read_csv(csv_filepath)
        if 'Marker' not in df.columns:
            raise ValueError("Kolom 'Marker' tidak ditemukan di CSV.")
            
        eeg_data = df[self.eeg_channels].values
        filtered_eeg = self.apply_filter(eeg_data)
        
        marker_1_indices = df.index[df['Marker'] == 1].tolist()
        marker_2_indices = df.index[df['Marker'] == 2].tolist()
        
        all_clean_epochs = []
        
        for idx in marker_1_indices:
            slot_data = filtered_eeg[idx : idx + (5 * self.fs)]
            clean_windows = self.windowing_slot(slot_data)
            all_clean_epochs.extend(clean_windows)
            
        for idx in marker_2_indices:
            slot_data = filtered_eeg[idx : idx + (5 * self.fs)]
            clean_windows = self.windowing_slot(slot_data)
            all_clean_epochs.extend(clean_windows)
            
        return np.array(all_clean_epochs)

    # =====================================================================
    # 2. SISTEM ADD-ON (KHUSUS EKSPERIMEN BARU)
    # =====================================================================

    def extract_erp_window(self, slot_data, start_ms=0, end_ms=1000):
        """[EKSPERIMEN 3] ERP Cropping & [EKSPERIMEN 2] Resampling"""
        start_idx = int((start_ms / 1000.0) * self.fs)
        end_idx = int((end_ms / 1000.0) * self.fs)
        
        if end_idx > len(slot_data):
            end_idx = len(slot_data)
            
        window_data = slot_data[start_idx:end_idx]
        
        if not self.reject_artifacts(window_data):
            # FIX: Aplikasikan Resampling JIKA target_fs berbeda
            if self.target_fs != self.fs:
                duration_sec = (end_ms - start_ms) / 1000.0
                target_length = int(duration_sec * self.target_fs)
                window_data = resample(window_data, target_length, axis=0)
            return [window_data]
        else:
            return []
            
    def apply_augmentation(self, window_data, add_noise=True, noise_factor=0.05, apply_jitter=True, jitter_ms=10):
        """[EKSPERIMEN 5] Data Augmentation: Noise & Jittering."""
        augmented_data = window_data.copy()
        
        if apply_jitter:
            # FIX: Mencegah temporal leakage, pergeseran jittering harus bergantung pada target_fs
            jitter_samples = int((jitter_ms / 1000.0) * self.target_fs)
            if jitter_samples > 0:
                direction = np.random.choice([-1, 1]) 
                shift = direction * jitter_samples
                
                if shift > 0: 
                    augmented_data[shift:, :] = augmented_data[:-shift, :]
                    augmented_data[:shift, :] = 0 
                elif shift < 0: 
                    shift = abs(shift)
                    augmented_data[:-shift, :] = augmented_data[shift:, :]
                    augmented_data[-shift:, :] = 0

        if add_noise:
            noise = np.random.normal(0, noise_factor * np.std(augmented_data), augmented_data.shape)
            augmented_data = augmented_data + noise
            
        return augmented_data

    # =====================================================================
    # 3. FITUR ML KLASIK
    # =====================================================================

    def extract_classical_features(self, window_data):
        """[EKSPERIMEN 8] Ekstraksi Fitur Manual untuk ML Tradisional."""
        features = []
        for ch_idx in range(window_data.shape[1]):
            channel_signal = window_data[:, ch_idx]
            
            features.append(np.mean(channel_signal))
            features.append(np.var(channel_signal))
            features.append(skew(channel_signal))
            features.append(kurtosis(channel_signal))
            
            activity = np.var(channel_signal)
            diff1 = np.diff(channel_signal)
            mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
            diff2 = np.diff(diff1)
            mobility_diff1 = np.sqrt(np.var(diff2) / np.var(diff1)) if np.var(diff1) > 0 else 0
            complexity = mobility_diff1 / mobility if mobility > 0 else 0
            
            features.extend([activity, mobility, complexity])
            
        return np.array(features)
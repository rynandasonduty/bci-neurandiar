import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA

# EEG frequency band definitions
EEG_BANDS = {
    "broadband": (0.5, 50.0),
    "theta":     (4.0, 8.0),
    "alpha":     (8.0, 13.0),
    "low_beta":  (13.0, 20.0),
    "high_beta": (20.0, 30.0),
    "gamma":     (30.0, 50.0)
}

class SignalProcessor:
    def __init__(self, fs=256, band="broadband", order=4, artifact_threshold=100.0,
                 apply_ica=False, target_fs=None):
        """
        EEG signal processing pipeline for the NEURANDIAR BCI system.

        Args:
            fs (int): Native sampling frequency of the EEG device (Hz).
            band (str): Frequency band key from EEG_BANDS (e.g., 'broadband', 'alpha').
            order (int): Butterworth filter order.
            artifact_threshold (float): Amplitude threshold (uV) above which epochs are rejected.
            apply_ica (bool): Whether to apply Independent Component Analysis for artifact removal.
            target_fs (int or None): Resampling target frequency (Hz). If None, no resampling is applied.
        """
        self.fs = fs
        self.order = order
        self.artifact_threshold = artifact_threshold

        if band in EEG_BANDS:
            self.lowcut, self.highcut = EEG_BANDS[band]
        else:
            self.lowcut, self.highcut = EEG_BANDS["broadband"]
            print(f"[WARNING] Band '{band}' not recognized. Defaulting to broadband (0.5-50 Hz).")

        self.apply_ica = apply_ica
        # All window sizes and shifts are computed relative to target_fs to support resampling
        self.target_fs = target_fs if target_fs is not None else fs

        self.eeg_channels = [
            "EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7", "EEG.P7", "EEG.O1",
            "EEG.O2", "EEG.P8", "EEG.T8", "EEG.FC6", "EEG.F4", "EEG.F8", "EEG.AF4"
        ]

    # =====================================================================
    # 1. CORE BASELINE PROCESSING
    # =====================================================================

    def _butter_bandpass(self):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return b, a

    def apply_filter(self, data):
        """Apply zero-phase Butterworth bandpass filter, with optional ICA artifact removal."""
        b, a = self._butter_bandpass()
        filtered_data = filtfilt(b, a, data, axis=0)

        # E1: Independent Component Analysis for eye/muscle artifact removal
        if self.apply_ica:
            ica = FastICA(n_components=14, random_state=42, max_iter=800)
            sources = ica.fit_transform(filtered_data)

            # Kurtosis-based artifact detection: high kurtosis indicates non-Gaussian
            # distributions consistent with eye movement or EMG contamination
            for i in range(sources.shape[1]):
                component_kurt = kurtosis(sources[:, i])
                if component_kurt > 3.0:
                    sources[:, i] = 0

            filtered_data = ica.inverse_transform(sources)

        return filtered_data

    def reject_artifacts(self, epoch_data):
        """Return True if any channel exceeds the amplitude rejection threshold."""
        max_amplitude = np.max(np.abs(epoch_data))
        if max_amplitude > self.artifact_threshold:
            return True
        return False

    def windowing_slot(self, slot_data):
        """
        Extract up to five non-overlapping 1-second windows from a 5-second recording slot.
        Windows exceeding the artifact threshold are discarded.
        Resampling is applied if target_fs differs from the native fs.
        """
        windows = []
        window_size = self.fs * 1  # 1-second window at native sampling rate

        expected_length = 5 * self.fs
        if len(slot_data) < expected_length:
            return windows

        for i in range(5):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_data = slot_data[start_idx:end_idx]

            if not self.reject_artifacts(window_data):
                # E2: Apply resampling if target_fs differs from native fs
                if self.target_fs != self.fs:
                    target_length = self.target_fs * 1
                    window_data = resample(window_data, target_length, axis=0)
                windows.append(window_data)

        return windows

    def process_csv_file(self, csv_filepath):
        df = pd.read_csv(csv_filepath)
        if 'Marker' not in df.columns:
            raise ValueError("Column 'Marker' not found in CSV file.")

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
    # 2. EXPERIMENT-SPECIFIC ADD-ONS
    # =====================================================================

    def extract_erp_window(self, slot_data, start_ms=0, end_ms=1000):
        """
        Extract a single ERP time window from a recording slot (E3: N400 cropping).
        Also applies resampling if target_fs differs from native fs (E2).

        Args:
            slot_data (np.ndarray): Raw filtered EEG epoch array (samples x channels).
            start_ms (int): Window start time in milliseconds post-stimulus.
            end_ms (int): Window end time in milliseconds post-stimulus.

        Returns:
            list: A list containing the extracted window array, or an empty list if rejected.
        """
        start_idx = int((start_ms / 1000.0) * self.fs)
        end_idx = int((end_ms / 1000.0) * self.fs)

        if end_idx > len(slot_data):
            end_idx = len(slot_data)

        window_data = slot_data[start_idx:end_idx]

        if not self.reject_artifacts(window_data):
            if self.target_fs != self.fs:
                duration_sec = (end_ms - start_ms) / 1000.0
                target_length = int(duration_sec * self.target_fs)
                window_data = resample(window_data, target_length, axis=0)
            return [window_data]
        else:
            return []

    def apply_augmentation(self, window_data, add_noise=True, noise_factor=0.05,
                           apply_jitter=True, jitter_ms=10):
        """
        Apply data augmentation to a single EEG window (E5: noise injection and temporal jittering).

        Args:
            window_data (np.ndarray): EEG window array (time x channels).
            add_noise (bool): Whether to add Gaussian noise scaled by the signal's standard deviation.
            noise_factor (float): Fraction of the signal std used as the noise standard deviation.
            apply_jitter (bool): Whether to apply a random temporal shift.
            jitter_ms (int): Maximum jitter magnitude in milliseconds.

        Returns:
            np.ndarray: Augmented EEG window of the same shape as the input.
        """
        augmented_data = window_data.copy()

        if apply_jitter:
            # Jitter shift magnitude is computed relative to target_fs for temporal consistency
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
    # 3. CLASSICAL ML FEATURE EXTRACTION
    # =====================================================================

    def extract_classical_features(self, window_data):
        """
        Extract per-channel time-domain and Hjorth features for classical ML classification (E8).

        Args:
            window_data (np.ndarray): EEG window array (time x channels).

        Returns:
            np.ndarray: Concatenated feature vector across all channels.
        """
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

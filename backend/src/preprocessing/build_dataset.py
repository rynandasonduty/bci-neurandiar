import os
import sys
import glob
import numpy as np
import pandas as pd

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
        Initialize the dataset builder for a given experiment configuration.

        Args:
            exp_id (str): Experiment identifier (e.g., 'E0_Baseline').
            processor_params (dict): Parameters forwarded to SignalProcessor (band, apply_ica, target_fs).
            crop_time (tuple): ERP extraction window (start_ms, end_ms), or None for 5-second baseline windowing.
            use_augmentation (bool): Reserved flag; augmentation is deferred to the post-split training phase.
            augmentation_params (dict): Parameters for SignalProcessor.apply_augmentation().
            phase_filter (str): Recording phase filter: 'all', 'overt', or 'imagined'.
            channels_to_use (str or list): Channel subset for E4 ablation, or 'all'.
        """
        print(f"\n[INFO] Initializing DatasetBuilder for experiment: {exp_id}")

        # 1. Resolve experiment directories from the Golden Standard path engine
        self.paths = setup_experiment(exp_id)
        self.raw_data_dir = self.paths["raw_data"]
        self.output_dir = self.paths["processed_data"]

        # 2. Store experiment parameters
        self.crop_time = crop_time
        # Augmentation parameters are stored but not executed here;
        # they are applied post-split in the training pipeline to prevent data leakage.
        self.use_augmentation = use_augmentation
        self.augmentation_params = augmentation_params if augmentation_params else {}
        self.phase_filter = phase_filter.lower()

        # 3. Initialize signal processor
        if processor_params is None:
            processor_params = {}
        self.processor = SignalProcessor(**processor_params)

        # 4. Resolve channel subset (E4: channel ablation)
        self.all_channels = self.processor.eeg_channels
        if channels_to_use == "all":
            self.selected_channels = self.all_channels
        else:
            self.selected_channels = [ch for ch in channels_to_use if ch in self.all_channels]
            print(f"[INFO] Channel ablation active. Using channels: {self.selected_channels}")

        self.channel_indices = [self.all_channels.index(ch) for ch in self.selected_channels]

    def parse_log_for_word_sequence(self, log_filepath):
        sequence = []
        current_phase = "unknown"

        with open(log_filepath, 'r') as file:
            for line in file:
                line_lower = line.lower()

                # Track the current recording phase from block header lines
                if "overt" in line_lower:
                    current_phase = "overt"
                elif "imagined" in line_lower:
                    current_phase = "imagined"

                # Record each trial entry with its associated phase label
                if "Menjalankan Trial" in line and "Kata:" in line:
                    try:
                        word = line.split("Kata: ")[1].split("(")[0].strip().upper()
                        sequence.append({"word": word, "phase": current_phase})
                    except Exception:
                        pass

        return sequence

    def process_subject(self, subject_id, csv_filepath, log_filepath):
        print(f"[INFO] Processing raw EEG data for subject: {subject_id}")
        
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

        # Apply bandpass filter (and optional ICA) to the full recording
        eeg_data = df[self.all_channels].values
        filtered_eeg = self.processor.apply_filter(eeg_data)

        marker_indices = df.index[df[marker_col] > 0].tolist()

        X_clean_windows = []
        y_labels = []

        # Marker counter used for phase synchronization (two syllable markers per word trial)
        valid_marker_count = 0

        for idx in marker_indices:
            marker_value = int(df.iloc[idx][marker_col])

            # Ignore markers outside the target syllable class range (1-19)
            if marker_value < 1 or marker_value > 19:
                continue

            # E6 phase filter: map valid marker index to trial phase from the log
            trial_idx = valid_marker_count // 2
            if trial_idx < len(trial_sequence):
                trial_phase = trial_sequence[trial_idx]["phase"]
            else:
                trial_phase = "unknown"

            valid_marker_count += 1

            # Skip epochs whose phase does not match the experiment's phase filter
            if self.phase_filter != "all" and trial_phase != self.phase_filter:
                continue

            label_int = marker_value - 1
            slot_data = filtered_eeg[idx : idx + (5 * self.processor.fs)]

            # E3: ERP window cropping vs. E0 baseline 5-second windowing
            if self.crop_time is not None:
                clean_windows = self.processor.extract_erp_window(
                    slot_data, start_ms=self.crop_time[0], end_ms=self.crop_time[1]
                )
            else:
                clean_windows = self.processor.windowing_slot(slot_data)

            for window in clean_windows:
                # E4: Apply channel subset selection
                window_selected = window[:, self.channel_indices]

                # Augmentation (E5) is deferred to the post-split training phase
                # to prevent data leakage from augmented samples into the test set.
                X_clean_windows.append(window_selected)
                y_labels.append(label_int)

        if len(X_clean_windows) == 0:
            return [], []

        X_subj_array = np.array(X_clean_windows)

        # StandardScaler is intentionally excluded here;
        # scaling must be applied post-split in data_utils.py to prevent leakage.
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

        np.save(os.path.join(self.output_dir, "X_features.npy"), X_tensor)
        np.save(os.path.join(self.output_dir, "y_labels.npy"), y_tensor)
        print(f"[INFO] {len(y_tensor)} samples extracted and saved to: {self.output_dir}")

        if return_data:
            return X_tensor, y_tensor

if __name__ == "__main__":
    builder = DatasetBuilder(exp_id="E0_Test")
    builder.build_full_dataset()
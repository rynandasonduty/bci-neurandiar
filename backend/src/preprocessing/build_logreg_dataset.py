import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment
from preprocessing.signal_processor import SignalProcessor
from utils.data_utils import three_way_split

WORD_CLASSES = {
    "MAKAN": 0, "MINUM": 1, "BERAK": 2, "PIPIS": 3, "MANDI": 4,
    "BOSAN": 5, "LELAH": 6, "SAKIT": 7, "TIDUR": 8, "SAYANG": 9
}

class LogRegDatasetBuilder:
    def __init__(self, exp_id="E0_Baseline", processor_params=None,
                 crop_time=None, phase_filter="all", channels_to_use="all"):

        print(f"\n[INFO] Initializing LogRegDatasetBuilder for experiment: {exp_id}")

        # 1. Resolve experiment directories
        self.paths = setup_experiment(exp_id)
        self.raw_data_dir = self.paths["raw_data"]
        self.output_dir = self.paths["processed_data"]
        self.weights_dir = self.paths["weights"]

        # 2. Experiment parameters
        self.crop_time = crop_time
        self.phase_filter = phase_filter.lower()

        # 3. Signal processor
        if processor_params is None:
            processor_params = {}
        self.processor = SignalProcessor(**processor_params)

        # 4. Channel selection
        self.all_channels = self.processor.eeg_channels
        if channels_to_use == "all":
            self.selected_channels = self.all_channels
        else:
            self.selected_channels = [ch for ch in channels_to_use if ch in self.all_channels]
        self.channel_indices = [self.all_channels.index(ch) for ch in self.selected_channels]

        # 5. Load the champion EEGNet model and its corresponding scaler
        model_path = os.path.join(self.weights_dir, f"eegnet_trained_{exp_id}.h5")
        scaler_path = os.path.join(self.weights_dir, f"scaler_{exp_id}.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"EEGNet model or scaler not found for experiment {exp_id}. "
                f"Run the EEGNet training pipeline first."
            )

        print(f"[INFO] Loading EEGNet model: {model_path}")
        self.eegnet = load_model(model_path)

        print(f"[INFO] Loading parent scaler: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def parse_log_for_word_sequence(self, log_filepath):
        """Parse the experiment log to extract the ordered word–phase trial sequence."""
        sequence = []
        current_phase = "unknown"

        with open(log_filepath, 'r') as file:
            for line in file:
                line_lower = line.lower()

                if "overt" in line_lower:
                    current_phase = "overt"
                elif "imagined" in line_lower:
                    current_phase = "imagined"

                if "Menjalankan Trial" in line and "Kata:" in line:
                    try:
                        word = line.split("Kata: ")[1].split("(")[0].strip().upper()
                        sequence.append({"word": word, "phase": current_phase})
                    except Exception:
                        pass

        return sequence

    def extract_probabilities(self, clean_windows):
        """
        Pass a list of clean EEG windows through the pretrained EEGNet and return
        the mean class probability vector across all windows in the slot.
        """
        if len(clean_windows) == 0:
            return None

        windows_array = np.array(clean_windows)

        windows_selected = windows_array[:, :, self.channel_indices]
        N, T, C = windows_selected.shape

        # Transpose to EEGNet input format: (N, channels, time, 1)
        X_input = np.transpose(windows_selected, (0, 2, 1))
        X_input = np.expand_dims(X_input, axis=3)

        X_input_flatten = X_input.reshape(N, -1)

        try:
            X_scaled_flatten = self.scaler.transform(X_input_flatten)
        except Exception as e:
            print(f"[ERROR] Scaling failed: {e}")
            return None

        X_scaled = X_scaled_flatten.reshape(N, C, T, 1)

        probs = self.eegnet.predict(X_scaled, verbose=0)
        averaged_prob = np.mean(probs, axis=0)
        return averaged_prob

    def build_dataset(self):
        print("\n" + "=" * 50)
        print(" BUILDING LOGISTIC REGRESSION DATASET ")
        print("=" * 50)

        X_word_features = []
        y_word_labels = []

        log_files = glob.glob(os.path.join(self.raw_data_dir, "logs", "*_experiment_log.txt"))

        for log_path in log_files:
            filename = os.path.basename(log_path)
            subject_id = filename.replace("_experiment_log.txt", "")

            csv_files = glob.glob(os.path.join(self.raw_data_dir, f"{subject_id}*.csv"))
            if not csv_files:
                continue

            trial_sequence = self.parse_log_for_word_sequence(log_path)

            header_idx = 0
            with open(csv_files[0], 'r') as f:
                for i, line in enumerate(f):
                    if 'EEG.AF3' in line or 'AF3' in line:
                        header_idx = i
                        break

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

            valid_trials = 0
            i = 0
            trial_counter = 0

            while i < len(marker_indices) - 1:
                idx_slot1 = marker_indices[i]
                marker_val1 = int(df.iloc[idx_slot1][marker_col])

                if marker_val1 < 1 or marker_val1 > 19:
                    i += 1
                    continue

                idx_slot2 = marker_indices[i + 1]
                marker_val2 = int(df.iloc[idx_slot2][marker_col])

                if marker_val2 < 1 or marker_val2 > 19:
                    i += 1
                    continue

                if trial_counter >= len(trial_sequence):
                    break

                target_word = trial_sequence[trial_counter]["word"]
                trial_phase = trial_sequence[trial_counter]["phase"]

                i += 2
                trial_counter += 1

                if target_word not in WORD_CLASSES:
                    continue
                if self.phase_filter != "all" and trial_phase != self.phase_filter:
                    continue

                data_slot1 = filtered_eeg[idx_slot1: idx_slot1 + (5 * self.processor.fs)]
                data_slot2 = filtered_eeg[idx_slot2: idx_slot2 + (5 * self.processor.fs)]

                if self.crop_time is not None:
                    clean_win1 = self.processor.extract_erp_window(data_slot1, self.crop_time[0], self.crop_time[1])
                    clean_win2 = self.processor.extract_erp_window(data_slot2, self.crop_time[0], self.crop_time[1])
                else:
                    clean_win1 = self.processor.windowing_slot(data_slot1)
                    clean_win2 = self.processor.windowing_slot(data_slot2)

                p1 = self.extract_probabilities(clean_win1)
                p2 = self.extract_probabilities(clean_win2)

                if p1 is not None and p2 is not None:
                    phi_features = np.concatenate((p1, p2))
                    X_word_features.append(phi_features)
                    y_word_labels.append(WORD_CLASSES[target_word])
                    valid_trials += 1

            print(f"[INFO] Subject {subject_id}: {valid_trials} word trials extracted.")

        if not X_word_features:
            print("[WARNING] No word trial data could be extracted.")
            return

        X_tensor = np.array(X_word_features)
        y_tensor = np.array(y_word_labels)

        X_w_train, X_w_val, X_w_test, y_w_train, y_w_val, y_w_test = three_way_split(X_tensor, y_tensor)

        # Save train split under the legacy name for backwards compatibility with run_master_experiments.py
        np.save(os.path.join(self.output_dir, "X_word_features.npy"), X_w_train)
        np.save(os.path.join(self.output_dir, "y_word_labels.npy"), y_w_train)
        np.save(os.path.join(self.output_dir, "X_word_val.npy"), X_w_val)
        np.save(os.path.join(self.output_dir, "y_word_val.npy"), y_w_val)
        np.save(os.path.join(self.output_dir, "X_word_test.npy"), X_w_test)
        np.save(os.path.join(self.output_dir, "y_word_test.npy"), y_w_test)

        print(f"[INFO] LogReg dataset saved: {len(X_w_train)} train | {len(X_w_val)} val | {len(X_w_test)} test")

if __name__ == "__main__":
    builder = LogRegDatasetBuilder(exp_id="E0_Test")
    builder.build_dataset()

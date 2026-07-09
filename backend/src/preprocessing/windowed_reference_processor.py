"""
backend/src/preprocessing/windowed_reference_processor.py

Control experiment support module for the P4 subsampling control.

Rebuilds the standard 1-second-windowed S3/E0_Baseline dataset (the same
structure used by the P3 SVM champion) by calling the existing, unmodified
SignalProcessor.apply_filter() and SignalProcessor.windowing_slot() public
methods. The resulting full feature/label set is used only as a source
pool for the same-size random subsampling performed by
run_p4_control_subsampled.py -- no model is trained on the full dataset
here.

This module is new and self-contained. It duplicates the marker/CSV parsing
loop from preprocessing.build_dataset.DatasetBuilder.process_subject()
locally, rather than importing DatasetBuilder, to avoid instantiating that
class (whose constructor resolves paths under the P1_Global paradigm by
default) and to keep this control experiment fully isolated from the
P1/P2/P3 pipeline modules, consistent with the P4 No-Windowing pilot's
full_epoch_processor.py.
"""
import numpy as np
import pandas as pd

from preprocessing.signal_processor import SignalProcessor


class WindowedReferenceDatasetBuilder:
    """
    Builds the standard windowed dataset (up to five 1-second windows per
    5-second slot, via SignalProcessor.windowing_slot()) for a single
    subject, under the E0_Baseline configuration: broadband 0.5-50 Hz,
    no ICA, no ERP cropping, no channel ablation, no augmentation.
    """

    def __init__(self, processor_params=None, phase_filter="all"):
        if processor_params is None:
            processor_params = {}
        self.processor = SignalProcessor(**processor_params)
        self.all_channels = self.processor.eeg_channels
        self.phase_filter = phase_filter.lower()

    def parse_log_for_word_sequence(self, log_filepath):
        """Parse the experiment log for the ordered (word, phase) trial sequence."""
        sequence = []
        current_phase = "unknown"

        with open(log_filepath, "r") as file:
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

    def process_subject(self, subject_id, csv_filepath, log_filepath):
        """
        Load one subject's raw CSV/log pair and extract standard 1-second
        windows via SignalProcessor.windowing_slot(). Returns (X_list, y_labels)
        where each element of X_list has shape (fs, channels).
        """
        print(f"[INFO][P4-Control] Processing raw EEG data for subject: {subject_id}")

        trial_sequence = self.parse_log_for_word_sequence(log_filepath)

        header_idx = 0
        with open(csv_filepath, "r") as f:
            for i, line in enumerate(f):
                if "EEG.AF3" in line or "AF3" in line:
                    header_idx = i
                    break

        df = pd.read_csv(csv_filepath, header=header_idx, low_memory=False)
        try:
            float(df.iloc[0][self.all_channels[0]])
        except (ValueError, TypeError):
            df = df.iloc[1:].reset_index(drop=True)

        for col in self.all_channels:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        marker_col = "MarkerValueInt" if "MarkerValueInt" in df.columns else "Marker"
        if marker_col not in df.columns:
            return [], []

        df[marker_col] = pd.to_numeric(df[marker_col], errors="coerce").fillna(0)

        eeg_data = df[self.all_channels].values
        filtered_eeg = self.processor.apply_filter(eeg_data)

        marker_indices = df.index[df[marker_col] > 0].tolist()

        X_windows = []
        y_labels = []
        valid_marker_count = 0

        for idx in marker_indices:
            marker_value = int(df.iloc[idx][marker_col])

            if marker_value < 1 or marker_value > 19:
                continue

            trial_idx = valid_marker_count // 2
            if trial_idx < len(trial_sequence):
                trial_phase = trial_sequence[trial_idx]["phase"]
            else:
                trial_phase = "unknown"

            valid_marker_count += 1

            if self.phase_filter != "all" and trial_phase != self.phase_filter:
                continue

            label_int = marker_value - 1
            slot_data = filtered_eeg[idx: idx + (5 * self.processor.fs)]

            clean_windows = self.processor.windowing_slot(slot_data)

            for window in clean_windows:
                X_windows.append(window)
                y_labels.append(label_int)

        if len(X_windows) == 0:
            return [], []

        return X_windows, y_labels

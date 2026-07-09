"""
backend/src/preprocessing/full_epoch_processor.py

P4 (No-Windowing) pilot paradigm: standalone epoch extraction that treats a
full 5-second recording slot as a single sample, without splitting it into
five 1-second windows as SignalProcessor.windowing_slot() does for P1/P2/P3.

This module is entirely new and self-contained. It does not modify
SignalProcessor, EEGFeatureExtractor, or preprocessing.build_dataset in any
way -- it only calls their existing public methods (SignalProcessor.__init__,
SignalProcessor.apply_filter, SignalProcessor.reject_artifacts). The marker
parsing / CSV loading logic below is intentionally duplicated (not imported
from build_dataset.DatasetBuilder) to keep the P4 pilot experiment fully
isolated from the P1/P2/P3 pipeline, per the experiment's isolation
requirements.
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import resample

from preprocessing.signal_processor import SignalProcessor


def extract_full_epoch(processor: SignalProcessor, slot_data: np.ndarray):
    """
    Extract one 5-second recording slot as a single unified epoch sample.

    Unlike SignalProcessor.windowing_slot(), which slices a slot into five
    1-second windows and rejects each independently, this function evaluates
    the amplitude artifact threshold once over the entire 5-second epoch and
    either keeps it whole or discards it.

    Args:
        processor (SignalProcessor): existing SignalProcessor instance,
            used only through its public fs/target_fs attributes and its
            public reject_artifacts() method.
        slot_data (np.ndarray): filtered raw slot, shape (samples, channels),
            where samples is expected to be 5 * processor.fs.

    Returns:
        np.ndarray or None: the full epoch, shape (5*fs_out, channels), or
        None if the slot is too short or fails artifact rejection.
    """
    expected_length = 5 * processor.fs
    if len(slot_data) < expected_length:
        return None

    epoch_data = slot_data[:expected_length]

    if processor.reject_artifacts(epoch_data):
        return None

    if processor.target_fs != processor.fs:
        target_length = processor.target_fs * 5
        epoch_data = resample(epoch_data, target_length, axis=0)

    return epoch_data


class FullEpochDatasetBuilder:
    """
    Builds a P4 (no-windowing) dataset: one sample per 5-second marker slot,
    extracted via extract_full_epoch() instead of SignalProcessor.windowing_slot().

    Mirrors the marker-parsing conventions of the existing pipeline (raw CSV
    with MarkerValueInt/Marker column, 19 syllable classes, experiment log
    phase tracking) so that P4 results remain comparable with P1/P2/P3, but
    reimplements the loop locally to avoid any dependency on P1/P2/P3-owned
    dataset-building code.
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
        Load one subject's raw CSV/log pair and extract full 5-second epochs
        (no windowing). Returns (X_list, y_labels) where each element of
        X_list has shape (5*fs_out, channels).
        """
        print(f"[INFO][P4] Processing raw EEG data for subject: {subject_id}")

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

        X_epochs = []
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

            epoch = extract_full_epoch(self.processor, slot_data)
            if epoch is not None:
                X_epochs.append(epoch)
                y_labels.append(label_int)

        if len(X_epochs) == 0:
            return [], []

        return X_epochs, y_labels

"""
backend/src/experiments_p4_p7/signal_processors_ext.py

SignalProcessor subclasses for P4 (No-Windowing) and P5 (Shifted Bandpass).
Both inherit everything from the unmodified SignalProcessor
(backend/src/preprocessing/signal_processor.py) and override only the one
behavior their experiment varies -- P4 the epoching, P5 the filter band.
"""
import os
import sys

from scipy.signal import resample

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.signal_processor import SignalProcessor


class FullEpochSignalProcessor(SignalProcessor):
    """P4: returns one full 5-second epoch per slot instead of five 1-second
    windows. Artifact rejection is applied once to the full epoch, not
    per sub-window. All other behavior (filtering, artifact threshold,
    resampling target) is inherited unchanged from SignalProcessor."""

    def windowing_slot(self, slot_data):
        expected_length = 5 * self.fs
        if len(slot_data) < expected_length:
            return []
        full_window = slot_data[:expected_length]
        if self.reject_artifacts(full_window):
            return []
        if self.target_fs != self.fs:
            target_length = self.target_fs * 5
            full_window = resample(full_window, target_length, axis=0)
        return [full_window]


class ShiftedBandSignalProcessor(SignalProcessor):
    """P5: shifts the bandpass filter to 15-65 Hz instead of the standard
    broadband 0.5-50 Hz. Windowing, artifact rejection, and resampling are
    all inherited unchanged from SignalProcessor -- only the cutoff
    frequencies used by apply_filter()'s internal _butter_bandpass() differ."""

    def __init__(self, fs=256, order=4, artifact_threshold=100.0,
                 apply_ica=False, target_fs=None):
        super().__init__(fs=fs, band="broadband", order=order,
                          artifact_threshold=artifact_threshold,
                          apply_ica=apply_ica, target_fs=target_fs)
        self.lowcut, self.highcut = 15.0, 65.0

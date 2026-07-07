"""
Offline Trial Reader
======================
Membaca satu trial nyata (satu kata utuh, terdiri dari dua epoch suku kata:
slot 1 dan slot 2) dari rekaman CSV mentah dan log eksperimen milik subjek
tertentu, untuk mode "Model Offline" pada demo inferensi.

Logika pemisahan marker dan windowing di modul ini SENGAJA mereplikasi
`preprocessing.build_dataset.DatasetBuilder.process_subject()` — pipeline
nyata yang benar-benar dipakai untuk membangun data latih seluruh model
champion (lihat `models/run_e8_classical.py`) — BUKAN
`SignalProcessor.process_csv_file()`. Fungsi tersebut mengharapkan kolom
'Marker' bernilai literal 1/2, padahal CSV mentah asli memakai kolom
'MarkerValueInt' dengan nilai 1-19 yang merepresentasikan ID suku kata;
fungsi tersebut tidak pernah dipanggil di jalur mana pun pada codebase ini
dan akan gagal bila dijalankan pada data asli.
"""

import os
import gc
import glob
import random

import numpy as np
import pandas as pd

from preprocessing.signal_processor import SignalProcessor


def _parse_log_trial_sequence(log_filepath):
    """Parse log eksperimen menjadi daftar {word, phase} terurut sesuai urutan trial."""
    sequence = []
    current_phase = "unknown"

    with open(log_filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
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


class OfflineTrialReader:
    """
    Mengekstrak epoch slot 1 dan slot 2 nyata untuk satu trial tertentu milik
    subjek, memakai parameter preprocessing yang identik dengan resep
    eksperimen yang dipakai melatih champion (E5_Data_Augmentation).

    Data ter-filter per subjek di-cache di memori supaya pemanggilan
    berulang untuk subjek yang sama (mis. beberapa trial berbeda, atau klik
    "Start Inference" berkali-kali pada demo) tidak mengulang pembacaan CSV
    dan filtering sinyal penuh dari awal.
    """

    def __init__(self, raw_data_dir, processor_params):
        self.raw_data_dir = raw_data_dir
        self.processor = SignalProcessor(**processor_params)
        self._session_cache = {}

    @staticmethod
    def _marker_column(df):
        return "MarkerValueInt" if "MarkerValueInt" in df.columns else "Marker"

    def _load_csv(self, csv_filepath):
        header_idx = 0
        header_columns = []
        with open(csv_filepath, "r") as f:
            for i, line in enumerate(f):
                if "EEG.AF3" in line or "AF3" in line:
                    header_idx = i
                    header_columns = [c.strip() for c in line.strip().split(",")]
                    break

        marker_col = "MarkerValueInt" if "MarkerValueInt" in header_columns else "Marker"
        # Load only the columns actually needed (14 EEG channels + marker), instead of
        # all ~59 EmotivPRO export columns (CQ/EQ/battery/etc.), and read in bounded
        # chunks rather than materialising the whole ~350MB file at once. This keeps
        # peak memory low and predictable when several subject CSVs are processed in
        # sequence under constrained system memory.
        usecols = [c for c in self.processor.eeg_channels + [marker_col] if c in header_columns]

        chunks = pd.read_csv(
            csv_filepath, header=header_idx, usecols=usecols, low_memory=False, chunksize=10_000
        )
        df = pd.concat(chunks, ignore_index=True)
        try:
            float(df.iloc[0][self.processor.eeg_channels[0]])
        except (ValueError, TypeError):
            df = df.iloc[1:].reset_index(drop=True)

        for col in self.processor.eeg_channels:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if marker_col not in df.columns:
            raise ValueError(f"Kolom marker tidak ditemukan pada: {csv_filepath}")
        df[marker_col] = pd.to_numeric(df[marker_col], errors="coerce").fillna(0)

        return df, marker_col

    def _load_subject(self, subject_id):
        """Baca, filter, dan cache sinyal EEG penuh + posisi marker milik satu subjek."""
        if subject_id in self._session_cache:
            return self._session_cache[subject_id]

        log_path = os.path.join(self.raw_data_dir, "logs", f"{subject_id}_experiment_log.txt")
        csv_candidates = glob.glob(os.path.join(self.raw_data_dir, f"{subject_id}*.csv"))
        if not os.path.exists(log_path) or not csv_candidates:
            raise FileNotFoundError(f"Data mentah untuk subjek {subject_id} tidak ditemukan.")

        df, marker_col = self._load_csv(csv_candidates[0])

        eeg_data = df[self.processor.eeg_channels].values
        filtered_eeg = self.processor.apply_filter(eeg_data)
        marker_indices = df.index[(df[marker_col] >= 1) & (df[marker_col] <= 19)].tolist()

        # Free the raw DataFrame and intermediate array explicitly (rather than waiting
        # for them to fall out of scope) so peak memory does not accumulate across
        # subjects when several ~350MB CSVs are processed back-to-back.
        del df, eeg_data
        gc.collect()

        trial_sequence = _parse_log_trial_sequence(log_path)

        session = {
            "filtered_eeg": filtered_eeg,
            "marker_indices": marker_indices,
            "trial_sequence": trial_sequence,
        }
        self._session_cache[subject_id] = session
        return session

    def list_valid_trials(self, subject_id):
        """Kembalikan daftar {word, phase} untuk trial yang punya pasangan marker lengkap di CSV."""
        session = self._load_subject(subject_id)
        n_valid_trials = len(session["marker_indices"]) // 2
        return session["trial_sequence"][:n_valid_trials]

    def _extract_first_clean_window(self, filtered_eeg, marker_idx):
        """Ambil window 1 detik pertama yang lolos rejeksi artefak dari slot 5 detik."""
        slot_data = filtered_eeg[marker_idx: marker_idx + (5 * self.processor.fs)]
        clean_windows = self.processor.windowing_slot(slot_data)
        if not clean_windows:
            return None
        return clean_windows[0]

    def read_trial(self, subject_id, trial_index=None):
        """
        Ekstrak satu trial nyata (dua epoch: slot 1 dan slot 2).

        Args:
            subject_id (str): ID subjek pada dataset (contoh: 'S3').
            trial_index (int atau None): Indeks trial berbasis nol. Jika
                None, trial valid dipilih secara acak; trial yang gagal lolos
                rejeksi artefak pada kedua slotnya dilewati secara otomatis
                dan diganti trial acak lain (tetap data nyata, tidak pernah
                direkayasa), sampai satu trial bersih ditemukan.

        Returns:
            dict: {
                "trial_index": int, "word": str, "phase": str,
                "epoch_slot1": np.ndarray (channels, time),
                "epoch_slot2": np.ndarray (channels, time),
            }
        """
        session = self._load_subject(subject_id)
        n_valid_trials = len(session["marker_indices"]) // 2

        if n_valid_trials == 0:
            raise ValueError(f"Tidak ditemukan pasangan marker valid untuk subjek {subject_id}.")

        if trial_index is not None:
            if not (0 <= trial_index < n_valid_trials):
                raise ValueError(f"trial_index {trial_index} di luar jangkauan (0-{n_valid_trials - 1}).")
            return self._build_trial_result(session, trial_index)

        # trial_index is None: coba trial acak, lewati yang tidak lolos rejeksi
        # artefak (tetap data nyata, hanya dipilih ulang secara acak).
        candidates = list(range(n_valid_trials))
        random.shuffle(candidates)
        last_error = None
        for candidate in candidates:
            try:
                return self._build_trial_result(session, candidate)
            except ValueError as e:
                last_error = e
                continue

        raise last_error or ValueError(
            f"Tidak ada trial dengan window bersih (bebas artefak) untuk subjek {subject_id}."
        )

    def _build_trial_result(self, session, trial_index):
        """Bangun dict hasil ekstraksi trial untuk satu trial_index yang sudah divalidasi."""
        marker_indices = session["marker_indices"]
        marker_idx_slot1 = marker_indices[trial_index * 2]
        marker_idx_slot2 = marker_indices[trial_index * 2 + 1]

        epoch_slot1 = self._extract_first_clean_window(session["filtered_eeg"], marker_idx_slot1)
        epoch_slot2 = self._extract_first_clean_window(session["filtered_eeg"], marker_idx_slot2)

        if epoch_slot1 is None or epoch_slot2 is None:
            raise ValueError(
                f"Trial {trial_index} tidak memiliki window bersih "
                f"(bebas artefak) pada salah satu atau kedua slot suku kata."
            )

        trial_sequence = session["trial_sequence"]
        if trial_index < len(trial_sequence):
            trial_meta = trial_sequence[trial_index]
        else:
            trial_meta = {"word": "UNKNOWN", "phase": "unknown"}

        return {
            "trial_index": trial_index,
            "word": trial_meta["word"],
            "phase": trial_meta["phase"],
            "epoch_slot1": epoch_slot1.T,  # (time, channels) -> (channels, time)
            "epoch_slot2": epoch_slot2.T,
        }

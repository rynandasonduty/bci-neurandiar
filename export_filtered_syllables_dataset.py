import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt

# ── PATH SETUP ────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent
RAW_DIR     = ROOT_DIR / "backend" / "dataset" / "raw"
LOG_DIR     = RAW_DIR / "logs"
OUTPUT_BASE = ROOT_DIR / "data_syllables_filtered"

# ── RECORDING CONSTANTS ───────────────────────────────────────────────────────
FS            = 256          # Emotiv EPOC X native sampling rate (Hz)
EPOCH_SECS    = 5            # Window to extract after each marker onset (seconds)
EPOCH_SAMPLES = FS * EPOCH_SECS   # 1280 samples per epoch

# ── FILTER PARAMETERS ────────────────────────────────────────────────────────
FILTER_LOWCUT  = 0.5         # Hz  — removes DC offset / slow drift
FILTER_HIGHCUT = 50.0        # Hz  — removes high-frequency noise above EEG band
FILTER_ORDER   = 4           # Butterworth order (zero-phase via filtfilt = effective order 8)

# Emotiv EPOC X channel names as they appear in the raw CSV
EEG_CHANNELS = [
    "EEG.AF3", "EEG.F7",  "EEG.F3",  "EEG.FC5",
    "EEG.T7",  "EEG.P7",  "EEG.O1",  "EEG.O2",
    "EEG.P8",  "EEG.T8",  "EEG.FC6", "EEG.F4",
    "EEG.F8",  "EEG.AF4",
]

# Marker ID → Syllable text (source: build_dataset.py SYLLABLE_CLASSES, offset +1)
MARKER_TO_SYLLABLE = {
    1:  "MA",   2:  "KAN",  3:  "MI",   4:  "NUM",
    5:  "BE",   6:  "RAK",  7:  "PI",   8:  "PIS",
    9:  "MAN",  10: "DI",   11: "BO",   12: "SAN",
    13: "LE",   14: "LAH",  15: "SA",   16: "KIT",
    17: "TI",   18: "DUR",  19: "YANG",
}


# ═════════════════════════════════════════════════════════════════════════════
# I. DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_raw_csv(csv_path: Path):
    """
    Load an Emotiv EPOC X raw CSV.

    File structure:
      Line 0: Metadata string (title, timestamps, headset info)
      Line 1: Column header (contains 'EEG.AF3', 'MarkerValueInt', etc.)
      Line 2+: Sample rows at 256 Hz

    Returns
    -------
    df         : pd.DataFrame with integer-indexed rows
    marker_col : str, name of the marker column ('MarkerValueInt' or 'Marker')
    """
    header_idx = 0
    with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if "EEG.AF3" in line or ",AF3," in line:
                header_idx = i
                break

    df = pd.read_csv(csv_path, header=header_idx, low_memory=False)

    try:
        float(df.iloc[0][EEG_CHANNELS[0]])
    except (ValueError, TypeError, KeyError):
        df = df.iloc[1:].reset_index(drop=True)

    marker_col = "MarkerValueInt" if "MarkerValueInt" in df.columns else "Marker"
    if marker_col not in df.columns:
        raise RuntimeError(
            f"No marker column found in {csv_path.name}. "
            f"Expected 'MarkerValueInt' or 'Marker'."
        )

    for ch in EEG_CHANNELS:
        if ch in df.columns:
            df[ch] = pd.to_numeric(df[ch], errors="coerce")

    df[marker_col] = pd.to_numeric(df[marker_col], errors="coerce").fillna(0).astype(int)

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    if "EEG.Counter" in df.columns:
        df["EEG.Counter"] = pd.to_numeric(df["EEG.Counter"], errors="coerce")

    return df.reset_index(drop=True), marker_col


# ═════════════════════════════════════════════════════════════════════════════
# II. BANDPASS FILTER  ← THE ONLY DIFFERENCE FROM THE RAW SCRIPT
# ═════════════════════════════════════════════════════════════════════════════

def apply_bandpass(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a zero-phase Butterworth bandpass filter to the 14 EEG channels
    in-place on the full continuous recording before any epoch slicing.

    Parameters
    ----------
    df : DataFrame containing the raw EEG columns (values ~4000 µV DC offset)

    Returns
    -------
    df : Same DataFrame with EEG columns replaced by filtered values (~0 µV baseline)

    Filter spec
    -----------
    Type   : Butterworth bandpass
    Band   : 0.5 – 50.0 Hz
    Order  : 4  (effective order = 8 due to zero-phase filtfilt)
    Method : scipy.signal.filtfilt (forward-backward → zero phase shift)
    """
    nyq  = 0.5 * FS
    low  = FILTER_LOWCUT  / nyq
    high = FILTER_HIGHCUT / nyq
    b, a = butter(FILTER_ORDER, [low, high], btype="band")

    present_channels = [ch for ch in EEG_CHANNELS if ch in df.columns]
    eeg_data = df[present_channels].values.astype(np.float64)

    # filtfilt requires the signal length to be > padlen = 3 * max(len(a), len(b))
    # For order-4 Butterworth: len(a) = len(b) = 5, padlen = 15
    # Full recording is ~900k samples — no risk of this condition failing
    filtered = filtfilt(b, a, eeg_data, axis=0)

    df = df.copy()
    df[present_channels] = filtered
    return df


# ═════════════════════════════════════════════════════════════════════════════
# III. LOG PARSING
# ═════════════════════════════════════════════════════════════════════════════

def parse_experiment_log(log_path: Path):
    """
    Parse the experiment log to build an ordered trial sequence.

    Returns
    -------
    list of dict with keys: trial_num, word, phase, slot_syllables
    """
    trials = []
    current_phase = "Unknown"
    current_trial = None

    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line_s = line.strip()
            ll = line_s.lower()

            if "fase 1" in ll or ("overt" in ll and "memulai" in ll):
                current_phase = "Overt"
                continue
            if "fase 2" in ll or ("imagined" in ll and "memulai" in ll):
                current_phase = "Imagined"
                continue

            if "Menjalankan Trial" in line_s and "Kata:" in line_s:
                if current_trial is not None:
                    trials.append(current_trial)
                try:
                    trial_num = int(line_s.split("Trial ")[1].split("/")[0].strip())
                    word = line_s.split("Kata: ")[1].split("(")[0].strip()
                except Exception:
                    trial_num = len(trials) + 1
                    word = "Unknown"
                current_trial = {
                    "trial_num":      trial_num,
                    "word":           word,
                    "phase":          current_phase,
                    "slot_syllables": [],
                }
                continue

            if "Inject Marker Slot" in line_s and current_trial is not None:
                try:
                    after_colon   = line_s.split("Slot")[1].split(":")[1].strip()
                    syllable_text = after_colon.split("(")[0].strip()
                    marker_id     = int(after_colon.split("ID:")[1].replace(")", "").strip())
                    current_trial["slot_syllables"].append((syllable_text, marker_id))
                except Exception:
                    pass

    if current_trial is not None:
        trials.append(current_trial)

    return trials


# ═════════════════════════════════════════════════════════════════════════════
# IV. EPOCH EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_epochs(df: pd.DataFrame, marker_col: str, trial_sequence: list):
    """
    Slice the filtered EEG DataFrame into per-marker epochs and attach metadata.
    Logic is identical to the raw script — filtering was already applied upstream.
    """
    valid_mask   = (df[marker_col] >= 1) & (df[marker_col] <= 19)
    marker_rows  = df.index[valid_mask].tolist()

    epochs           = []
    valid_marker_cnt = 0
    occurrence_ctr   = {}

    for row_idx in marker_rows:
        marker_val = int(df.at[row_idx, marker_col])

        trial_idx = valid_marker_cnt // 2
        slot_num  = (valid_marker_cnt % 2) + 1

        if trial_idx < len(trial_sequence):
            t         = trial_sequence[trial_idx]
            phase     = t["phase"]
            word      = t["word"]
            trial_num = t["trial_num"]
        else:
            phase     = "Unknown"
            word      = "Unknown"
            trial_num = -1

        valid_marker_cnt += 1

        syllable = MARKER_TO_SYLLABLE.get(marker_val, f"UNK{marker_val}")

        key = (syllable, phase)
        occurrence_ctr[key] = occurrence_ctr.get(key, 0) + 1
        occurrence_idx = occurrence_ctr[key]

        epoch_start = row_idx
        epoch_end   = min(row_idx + EPOCH_SAMPLES, len(df))
        actual_len  = epoch_end - epoch_start

        if actual_len < 64:
            print(f"    [WARN] Marker at row {row_idx} yielded only {actual_len} samples. Skipped.")
            continue

        epoch_df = df.iloc[epoch_start:epoch_end].copy()

        epoch_df["Sample_Index"]     = range(actual_len)
        epoch_df["Time_ms"]          = epoch_df["Sample_Index"] * (1000.0 / FS)
        epoch_df["Target_Syllable"]  = syllable
        epoch_df["Speech_Type"]      = phase
        epoch_df["Marker_Value"]     = marker_val
        epoch_df["Trial_Number"]     = trial_num
        epoch_df["Parent_Word"]      = word
        epoch_df["Slot"]             = slot_num
        epoch_df["Occurrence_Index"] = occurrence_idx

        epochs.append(epoch_df)

    return epochs


# ═════════════════════════════════════════════════════════════════════════════
# V. COLUMN SELECTION & EXPORT
# ═════════════════════════════════════════════════════════════════════════════

META_COLS = [
    "Target_Syllable",
    "Speech_Type",
    "Marker_Value",
    "Trial_Number",
    "Parent_Word",
    "Slot",
    "Occurrence_Index",
    "Sample_Index",
    "Time_ms",
]

def build_output_columns(df_sample: pd.DataFrame, marker_col: str) -> list:
    leading        = [c for c in ["Timestamp", "EEG.Counter"] if c in df_sample.columns]
    eeg_present    = [c for c in EEG_CHANNELS if c in df_sample.columns]
    marker_section = [marker_col] if marker_col in df_sample.columns else []
    return leading + eeg_present + marker_section + META_COLS


def export_subject(subject_id: str, csv_path: Path, log_path: Path):
    """
    Full pipeline for one subject:
      load → bandpass filter (whole recording) → parse log → slice epochs → export.
    """
    print(f"\n{'='*62}")
    print(f"  Processing Subject: {subject_id}")
    print(f"  CSV : {csv_path.name}")
    print(f"  LOG : {log_path.name}")
    print(f"{'='*62}")

    # 1. Load raw CSV
    print(f"  [1/5] Loading raw CSV ...")
    df, marker_col = load_raw_csv(csv_path)
    n_samples    = len(df)
    duration_min = n_samples / FS / 60
    present_ch   = [ch for ch in EEG_CHANNELS if ch in df.columns]

    # Show DC offset before filtering (mean of first 256 samples = first second)
    dc_before = df[present_ch].iloc[:FS].mean().mean()
    print(f"        {n_samples:,} samples loaded  ({duration_min:.1f} min at {FS} Hz)")
    print(f"        DC offset before filter: {dc_before:+.1f} µV  (expected ~4000 µV)")

    # 2. Apply bandpass filter to the full continuous recording
    print(f"  [2/5] Applying bandpass filter "
          f"({FILTER_LOWCUT}–{FILTER_HIGHCUT} Hz, order {FILTER_ORDER}, zero-phase) ...")
    df = apply_bandpass(df)

    dc_after = df[present_ch].iloc[:FS].mean().mean()
    print(f"        DC offset after filter : {dc_after:+.4f} µV  (expected ~0 µV)")

    # 3. Parse experiment log
    print(f"  [3/5] Parsing experiment log ...")
    trial_seq  = parse_experiment_log(log_path)
    n_overt    = sum(1 for t in trial_seq if t["phase"] == "Overt")
    n_imagined = sum(1 for t in trial_seq if t["phase"] == "Imagined")
    print(f"        {len(trial_seq)} trials found  ({n_overt} Overt, {n_imagined} Imagined)")

    # 4. Slice epochs from the already-filtered DataFrame
    print(f"  [4/5] Slicing filtered epochs ({EPOCH_SECS}s / {EPOCH_SAMPLES} samples each) ...")
    all_epochs = extract_epochs(df, marker_col, trial_seq)
    print(f"        {len(all_epochs)} epochs extracted")

    if not all_epochs:
        print(f"  [ERROR] No epochs extracted for {subject_id}. Skipping.")
        return 0

    out_cols = build_output_columns(all_epochs[0], marker_col)

    # 5. Group by syllable and export
    print(f"  [5/5] Exporting per-syllable CSV files ...")
    subj_dir = OUTPUT_BASE / f"Subject_{subject_id}"
    subj_dir.mkdir(parents=True, exist_ok=True)

    by_syllable = {}
    for ep in all_epochs:
        syl = ep["Target_Syllable"].iloc[0]
        by_syllable.setdefault(syl, []).append(ep)

    exported_files = 0
    for syllable in sorted(by_syllable.keys()):
        epochs_for_syl = by_syllable[syllable]
        combined       = pd.concat(epochs_for_syl, ignore_index=True)

        final_cols = [c for c in out_cols if c in combined.columns]
        combined   = combined[final_cols]

        n_trials      = len(epochs_for_syl)
        n_overt_ep    = sum(1 for ep in epochs_for_syl if ep["Speech_Type"].iloc[0] == "Overt")
        n_imagined_ep = n_trials - n_overt_ep

        out_path = subj_dir / f"Subj{subject_id}_{syllable}_Filtered.csv"
        combined.to_csv(out_path, index=False)

        print(f"        Exporting Subject {subject_id} - {syllable:<5} "
              f"... {n_trials:>3} trials "
              f"({n_overt_ep} Overt / {n_imagined_ep} Imagined), "
              f"{len(combined):>6} rows  -> {out_path.name}")
        exported_files += 1

    print(f"  Done. {exported_files} files saved to: {subj_dir}")
    return exported_files


# ═════════════════════════════════════════════════════════════════════════════
# VI. MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Filtered Syllable Dataset Exporter")
    print("  Output: data_syllables_filtered/Subject_SN/SubjSN_XXX_Filtered.csv")
    print(f"  Filter : Butterworth bandpass {FILTER_LOWCUT}–{FILTER_HIGHCUT} Hz, "
          f"order {FILTER_ORDER}, zero-phase (filtfilt)")
    print(f"  Epoch  : {EPOCH_SECS}s = {EPOCH_SAMPLES} samples @ {FS} Hz")
    print("=" * 62)

    if not RAW_DIR.exists():
        sys.exit(f"[FATAL] Raw data directory not found: {RAW_DIR}")
    if not LOG_DIR.exists():
        sys.exit(f"[FATAL] Log directory not found: {LOG_DIR}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    log_files = sorted(LOG_DIR.glob("S*_experiment_log.txt"))
    if not log_files:
        sys.exit(f"[FATAL] No experiment log files found in: {LOG_DIR}")

    total_exported = 0
    skipped        = []

    for log_path in log_files:
        subject_id = log_path.stem.replace("_experiment_log", "")

        csv_candidates = sorted(RAW_DIR.glob(f"{subject_id}.csv"))
        if not csv_candidates:
            print(f"\n[SKIP] No CSV found for {subject_id} (expected {subject_id}.csv)")
            skipped.append(subject_id)
            continue

        csv_path = csv_candidates[0]

        try:
            n = export_subject(subject_id, csv_path, log_path)
            total_exported += n
        except Exception as exc:
            print(f"\n[ERROR] Subject {subject_id} failed: {exc}")
            import traceback; traceback.print_exc()
            skipped.append(subject_id)

    print("\n" + "=" * 62)
    print("  EXPORT COMPLETE")
    print(f"  Total syllable files exported : {total_exported}")
    print(f"  Subjects processed            : {len(log_files) - len(skipped)}/{len(log_files)}")
    if skipped:
        print(f"  Skipped subjects              : {', '.join(skipped)}")
    print(f"  Output directory              : {OUTPUT_BASE}")
    print("=" * 62)


if __name__ == "__main__":
    main()

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# ── PATH SETUP ────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent
RAW_DIR     = ROOT_DIR / "backend" / "dataset" / "raw"
LOG_DIR     = RAW_DIR / "logs"
OUTPUT_BASE = ROOT_DIR / "data_syllables"

# ── RECORDING CONSTANTS ───────────────────────────────────────────────────────
FS             = 256          # Emotiv EPOC X native sampling rate (Hz)
EPOCH_SECS     = 5            # Window to extract after each marker onset (seconds)
EPOCH_SAMPLES  = FS * EPOCH_SECS   # 1280 samples per epoch

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
    # Locate the actual header row by scanning for the EEG channel name
    header_idx = 0
    with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if "EEG.AF3" in line or ",AF3," in line:
                header_idx = i
                break

    df = pd.read_csv(csv_path, header=header_idx, low_memory=False)

    # If the row immediately after the header is still non-numeric metadata, drop it
    try:
        float(df.iloc[0][EEG_CHANNELS[0]])
    except (ValueError, TypeError, KeyError):
        df = df.iloc[1:].reset_index(drop=True)

    # Resolve marker column name
    marker_col = "MarkerValueInt" if "MarkerValueInt" in df.columns else "Marker"
    if marker_col not in df.columns:
        raise RuntimeError(
            f"No marker column found in {csv_path.name}. "
            f"Expected 'MarkerValueInt' or 'Marker'."
        )

    # Force numeric types on EEG and marker columns
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
# II. LOG PARSING
# ═════════════════════════════════════════════════════════════════════════════

def parse_experiment_log(log_path: Path):
    """
    Parse the experiment log file to build an ordered trial sequence.

    The log records wall-clock events like:
      [11:14:53] === MEMULAI FASE 1: OVERT SPEECH ===
      [11:17:19] Menjalankan Trial 1/100 (Blok 1) - Kata: Mandi
      [11:17:20] Inject Marker Slot 1: MAN (ID: 9)
      [11:17:27] Inject Marker Slot 2: DI (ID: 10)

    Phase boundaries are detected from "FASE 1" / "FASE 2" header lines.
    Each trial entry contributes one dict to the returned list.

    Returns
    -------
    list of dict with keys: trial_num, word, phase, slot_syllables
      slot_syllables: list of (syllable_text, marker_id) in injection order
    """
    trials = []
    current_phase = "Unknown"
    current_trial = None

    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line_s = line.strip()
            ll = line_s.lower()

            # Detect phase transitions
            if "fase 1" in ll or ("overt" in ll and "memulai" in ll):
                current_phase = "Overt"
                continue
            if "fase 2" in ll or ("imagined" in ll and "memulai" in ll):
                current_phase = "Imagined"
                continue

            # Detect trial header: "Menjalankan Trial N/100 (Blok K) - Kata: WORD"
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
                    "trial_num":       trial_num,
                    "word":            word,
                    "phase":           current_phase,
                    "slot_syllables":  [],
                }
                continue

            # Detect marker injection lines: "Inject Marker Slot N: SYL (ID: M)"
            if "Inject Marker Slot" in line_s and current_trial is not None:
                try:
                    # "Inject Marker Slot 1: MAN (ID: 9)"
                    after_colon = line_s.split("Slot")[1].split(":")[1].strip()
                    syllable_text = after_colon.split("(")[0].strip()
                    marker_id_str = after_colon.split("ID:")[1].replace(")", "").strip()
                    marker_id = int(marker_id_str)
                    current_trial["slot_syllables"].append((syllable_text, marker_id))
                except Exception:
                    pass

    # Flush last trial
    if current_trial is not None:
        trials.append(current_trial)

    return trials


# ═════════════════════════════════════════════════════════════════════════════
# III. EPOCH EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_epochs(df: pd.DataFrame, marker_col: str, trial_sequence: list):
    """
    Slice the raw EEG DataFrame into per-marker epochs and attach metadata.

    Pairing strategy (mirrors build_dataset.py):
      - Collect all CSV rows where MarkerValueInt is in [1, 19] in order.
      - Every pair of consecutive markers belongs to one trial
        (Slot 1 = first syllable, Slot 2 = second syllable).
      - trial_idx = valid_marker_count // 2
      - Phase and word are looked up from trial_sequence[trial_idx].

    Parameters
    ----------
    df             : raw EEG DataFrame (unfiltered)
    marker_col     : name of the trigger column
    trial_sequence : list returned by parse_experiment_log()

    Returns
    -------
    List of epoch DataFrames, each with metadata columns appended.
    """
    # Locate all valid marker positions in the CSV (in row order)
    valid_mask  = (df[marker_col] >= 1) & (df[marker_col] <= 19)
    marker_rows = df.index[valid_mask].tolist()

    epochs          = []
    valid_marker_cnt = 0
    occurrence_ctr  = {}   # {syllable: count} for labelling repeated occurrences

    for row_idx in marker_rows:
        marker_val = int(df.at[row_idx, marker_col])

        # Map to trial
        trial_idx = valid_marker_cnt // 2
        slot_num  = (valid_marker_cnt % 2) + 1       # 1 or 2

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

        # Syllable label from marker value
        syllable = MARKER_TO_SYLLABLE.get(marker_val, f"UNK{marker_val}")

        # Occurrence counter per (syllable, phase)
        key = (syllable, phase)
        occurrence_ctr[key] = occurrence_ctr.get(key, 0) + 1
        occurrence_idx = occurrence_ctr[key]

        # ── RAW EPOCH SLICE (no filtering) ───────────────────────────────────
        epoch_start = row_idx
        epoch_end   = min(row_idx + EPOCH_SAMPLES, len(df))
        actual_len  = epoch_end - epoch_start

        if actual_len < 64:   # less than 250ms — discard as truncated
            print(f"    [WARN] Marker at row {row_idx} yielded only {actual_len} samples. Skipped.")
            continue

        epoch_df = df.iloc[epoch_start:epoch_end].copy()

        # ── METADATA COLUMNS ─────────────────────────────────────────────────
        epoch_df["Sample_Index"]     = range(actual_len)
        epoch_df["Time_ms"]          = epoch_df["Sample_Index"] * (1000.0 / FS)
        epoch_df["Target_Syllable"]  = syllable
        epoch_df["Speech_Type"]      = phase
        epoch_df["Marker_Value"]     = marker_val
        epoch_df["Trial_Number"]     = trial_num
        epoch_df["Parent_Word"]      = word
        epoch_df["Slot"]             = slot_num   # 1 = first syllable of word, 2 = second
        epoch_df["Occurrence_Index"] = occurrence_idx  # N-th time this syllable appears

        epochs.append(epoch_df)

    return epochs


# ═════════════════════════════════════════════════════════════════════════════
# IV. COLUMN SELECTION & EXPORT
# ═════════════════════════════════════════════════════════════════════════════

# Columns kept in the output CSV (in this order)
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
    """Return the ordered list of columns to keep in the output CSV."""
    leading = []
    for c in ["Timestamp", "EEG.Counter"]:
        if c in df_sample.columns:
            leading.append(c)

    eeg_present = [c for c in EEG_CHANNELS if c in df_sample.columns]

    # marker_col last of the 'raw' section so it is clearly visible
    marker_section = [marker_col] if marker_col in df_sample.columns else []

    return leading + eeg_present + marker_section + META_COLS


def export_subject(subject_id: str, csv_path: Path, log_path: Path):
    """
    Full pipeline for one subject: load → parse log → extract epochs → export.
    """
    print(f"\n{'='*62}")
    print(f"  Processing Subject: {subject_id}")
    print(f"  CSV : {csv_path.name}")
    print(f"  LOG : {log_path.name}")
    print(f"{'='*62}")

    # 1. Load raw CSV (no filtering)
    print(f"  [1/4] Loading raw CSV ...")
    df, marker_col = load_raw_csv(csv_path)
    n_samples = len(df)
    duration_min = n_samples / FS / 60
    print(f"        {n_samples:,} samples loaded  ({duration_min:.1f} min at {FS} Hz)")

    # 2. Parse experiment log
    print(f"  [2/4] Parsing experiment log ...")
    trial_seq = parse_experiment_log(log_path)
    n_overt    = sum(1 for t in trial_seq if t["phase"] == "Overt")
    n_imagined = sum(1 for t in trial_seq if t["phase"] == "Imagined")
    print(f"        {len(trial_seq)} trials found  "
          f"({n_overt} Overt, {n_imagined} Imagined)")

    # 3. Extract epochs
    print(f"  [3/4] Extracting raw epochs ({EPOCH_SECS}s / {EPOCH_SAMPLES} samples each) ...")
    all_epochs = extract_epochs(df, marker_col, trial_seq)
    print(f"        {len(all_epochs)} epochs extracted")

    if not all_epochs:
        print(f"  [ERROR] No epochs extracted for {subject_id}. Skipping.")
        return 0

    # Build output column order from the first epoch
    out_cols = build_output_columns(all_epochs[0], marker_col)

    # 4. Group by syllable and export
    print(f"  [4/4] Exporting per-syllable CSV files ...")
    subj_dir = OUTPUT_BASE / f"Subject_{subject_id}"
    subj_dir.mkdir(parents=True, exist_ok=True)

    # Group epochs by syllable
    by_syllable = {}
    for ep in all_epochs:
        syl = ep["Target_Syllable"].iloc[0]
        by_syllable.setdefault(syl, []).append(ep)

    exported_files = 0
    for syllable in sorted(by_syllable.keys()):
        epochs_for_syl = by_syllable[syllable]
        combined = pd.concat(epochs_for_syl, ignore_index=True)

        # Keep only the selected columns (intersect with what actually exists)
        final_cols = [c for c in out_cols if c in combined.columns]
        combined   = combined[final_cols]

        n_trials = len(epochs_for_syl)
        n_overt_ep    = sum(
            1 for ep in epochs_for_syl if ep["Speech_Type"].iloc[0] == "Overt"
        )
        n_imagined_ep = n_trials - n_overt_ep

        out_path = subj_dir / f"Subj{subject_id}_{syllable}_Raw.csv"
        combined.to_csv(out_path, index=False)

        print(f"        Exporting Subject {subject_id} - {syllable:<5} "
              f"... {n_trials:>3} trials "
              f"({n_overt_ep} Overt / {n_imagined_ep} Imagined), "
              f"{len(combined):>6} rows  -> {out_path.name}")
        exported_files += 1

    print(f"  Done. {exported_files} files saved to: {subj_dir}")
    return exported_files


# ═════════════════════════════════════════════════════════════════════════════
# V. MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Raw Syllable Dataset Exporter")
    print("  Output: data_syllables/Subject_SN/SubjSN_XXX_Raw.csv")
    print("  Signal processing: NONE (pure raw EEG)")
    print(f"  Epoch window    : {EPOCH_SECS}s = {EPOCH_SAMPLES} samples @ {FS} Hz")
    print("=" * 62)

    # Validate paths
    if not RAW_DIR.exists():
        sys.exit(f"[FATAL] Raw data directory not found: {RAW_DIR}")
    if not LOG_DIR.exists():
        sys.exit(f"[FATAL] Log directory not found: {LOG_DIR}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Discover all available subjects from log files
    log_files = sorted(LOG_DIR.glob("S*_experiment_log.txt"))
    if not log_files:
        sys.exit(f"[FATAL] No experiment log files found in: {LOG_DIR}")

    total_exported = 0
    skipped        = []

    for log_path in log_files:
        # Derive subject ID from log filename: "S1_experiment_log.txt" → "S1"
        subject_id = log_path.stem.replace("_experiment_log", "")

        # Locate matching CSV
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

    # Final summary
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

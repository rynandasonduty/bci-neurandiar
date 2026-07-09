"""
backend/src/experiments_p4_p7/verify_p6_phase_labels.py

Step 0.3 pre-flight verification for P6 (Transfer Overt->Imagined).

Confirms two things, read-only, before any P6 code relies on them:

  1. DatasetBuilder.process_subject (backend/src/preprocessing/build_dataset.py,
     UNMODIFIED) already gates samples on phase_filter='overt'/'imagined'.
     Verified by inspecting its live source via `inspect.getsource` -- this
     script never edits that file, it only reads what is already there.

  2. For all 12 subjects, the raw experiment log has a complete, balanced set
     of overt-phase and imagined-phase trials (100 each under the standard
     protocol -- TRIALS_PER_SUBJECT=200 split across two phases). Any subject
     with incomplete/imbalanced counts is flagged, since it directly bounds
     how much extra overt-phase training data P6 can add.

This script only reads *_experiment_log.txt files. It does not touch raw
CSVs, does not build any dataset, and does not write anything to disk.
"""
import os
import sys
import glob
import inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import RAW_DATA_DIR
from preprocessing.build_dataset import DatasetBuilder

EXPECTED_TRIALS_PER_PHASE = 100


def confirm_phase_filter_support():
    """Static, read-only confirmation that DatasetBuilder.process_subject
    gates samples by phase, by inspecting its unmodified source (never by
    editing it)."""
    source = inspect.getsource(DatasetBuilder.process_subject)
    supports = "phase_filter" in source and "trial_phase" in source
    supports_all_three = all(
        f'"{val}"' in source or f"'{val}'" in source for val in ("all", "overt", "imagined")
    ) or ("self.phase_filter != \"all\"" in source)

    print("[CHECK] DatasetBuilder.process_subject (unmodified source inspection)")
    print(f"        references phase_filter/trial_phase gating : {supports}")
    print(f"        gate is 'if phase_filter != all: skip'      : {'self.phase_filter != \"all\"' in source}")
    if not supports:
        print("[WARNING] Could not confirm phase_filter gating in DatasetBuilder.process_subject source.")
    return supports


def parse_log_phase_counts(log_filepath):
    """Count trials per phase from one subject's experiment log.

    Mirrors the parsing convention already used independently by
    DatasetBuilder.parse_log_for_word_sequence, full_epoch_processor.py,
    windowed_reference_processor.py, and offline_trial_reader.py. Kept as its
    own local copy here (rather than imported) to stay consistent with this
    experiment set's isolation requirement -- this is a verification script,
    not a dependency any P4-P7 run script relies on.
    """
    counts = {"overt": 0, "imagined": 0, "unknown": 0}
    current_phase = "unknown"

    with open(log_filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_lower = line.lower()
            if "overt" in line_lower:
                current_phase = "overt"
            elif "imagined" in line_lower:
                current_phase = "imagined"

            if "Menjalankan Trial" in line and "Kata:" in line:
                counts[current_phase] += 1

    return counts


def run_verification():
    log_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "logs", "*_experiment_log.txt")))
    subject_ids = [os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files]
    print(f"[INFO] Found {len(subject_ids)} subject log(s) under {os.path.join(RAW_DATA_DIR, 'logs')}")

    confirm_phase_filter_support()

    print(f"\n{'Subject':<10}{'Overt':>8}{'Imagined':>10}{'Unknown':>10}{'Status':>12}")
    print("-" * 50)

    results = {}
    flagged = []
    for subject_id, log_file in zip(subject_ids, log_files):
        counts = parse_log_phase_counts(log_file)
        results[subject_id] = counts
        balanced = (
            counts["overt"] == EXPECTED_TRIALS_PER_PHASE
            and counts["imagined"] == EXPECTED_TRIALS_PER_PHASE
        )
        status = "OK" if balanced else "FLAGGED"
        if not balanced:
            flagged.append(subject_id)
        print(f"{subject_id:<10}{counts['overt']:>8}{counts['imagined']:>10}{counts['unknown']:>10}{status:>12}")

    print("-" * 50)
    if flagged:
        print(f"\n[WARNING] {len(flagged)} subject(s) with incomplete/imbalanced phase data: {flagged}")
        print("[WARNING] This bounds the amount of extra overt-phase training data available to P6 for those subjects.")
    else:
        print(
            f"\n[OK] All {len(subject_ids)} subjects have the expected "
            f"{EXPECTED_TRIALS_PER_PHASE}/{EXPECTED_TRIALS_PER_PHASE} overt/imagined trial split."
        )

    return {"subject_ids": subject_ids, "counts": results, "flagged": flagged}


if __name__ == "__main__":
    run_verification()

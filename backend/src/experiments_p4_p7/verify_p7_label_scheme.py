"""
backend/src/experiments_p4_p7/verify_p7_label_scheme.py

Step 0.4 pre-flight verification for P7 (Coarse-to-Fine Hierarchical Decoding).

Confirms, read-only, against the UNMODIFIED SYLLABLE_CLASSES dict in
backend/src/preprocessing/build_dataset.py and against real trial data parsed
from all 12 subjects' raw experiment logs:

  1. SYLLABLE_CLASSES matches the expected 19-class mapping exactly.
  2. The vowel-group hierarchy (A/I/E/O) and the deterministic first-syllable
     -> word dictionary (ground truth given by the researcher, defined here
     only as fixed reference constants -- never derived or guessed) are
     internally consistent with SYLLABLE_CLASSES and cover exactly the 9
     first-syllable + 10 second-syllable = 19 total classes.
  3. The "each syllable label only ever appears in one slot" assumption that
     P7's label-filtering approach depends on: for every trial of every
     subject, slot 1 always carries the word's first syllable and slot 2
     always carries its second syllable, and no syllable label is ever seen
     in both slot positions anywhere in the real dataset.

This script only reads *_experiment_log.txt files and imports SYLLABLE_CLASSES
for comparison. It does not modify build_dataset.py, does not touch raw CSVs,
and does not write anything to disk.
"""
import os
import sys
import re
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import RAW_DATA_DIR, TARGET_WORDS
from preprocessing.build_dataset import SYLLABLE_CLASSES

# ---------------------------------------------------------------------------
# Ground truth given by the researcher (Langkah 0.4). Fixed reference
# constants only -- P4-P7 code must treat these as read-only truth, never
# derive or modify them.
# ---------------------------------------------------------------------------
EXPECTED_SYLLABLE_CLASSES = {
    "MA": 0, "KAN": 1, "MI": 2, "NUM": 3, "BE": 4, "RAK": 5,
    "PI": 6, "PIS": 7, "MAN": 8, "DI": 9, "BO": 10, "SAN": 11,
    "LE": 12, "LAH": 13, "SA": 14, "KIT": 15, "TI": 16, "DUR": 17, "YANG": 18,
}

# First-syllable -> vowel group. Group "U" has no members in this vocabulary
# and is intentionally absent (nothing to verify against).
VOWEL_GROUPS = {
    "A": ["MA", "MAN", "SA"],
    "I": ["MI", "PI", "TI"],
    "E": ["BE", "LE"],
    "O": ["BO"],
}
SYLLABLE_TO_VOWEL_GROUP = {syl: grp for grp, syls in VOWEL_GROUPS.items() for syl in syls}

# Word -> (slot1 syllable, slot2 syllable). SA is the only ambiguous first
# syllable (SAKIT vs SAYANG); every other first syllable maps deterministically.
WORD_TO_SYLLABLES = {
    "MAKAN": ("MA", "KAN"), "MINUM": ("MI", "NUM"), "BERAK": ("BE", "RAK"),
    "PIPIS": ("PI", "PIS"), "MANDI": ("MAN", "DI"), "BOSAN": ("BO", "SAN"),
    "LELAH": ("LE", "LAH"), "SAKIT": ("SA", "KIT"), "TIDUR": ("TI", "DUR"),
    "SAYANG": ("SA", "YANG"),
}
DETERMINISTIC_FIRST_SYLLABLE_TO_WORD = {
    syl1: word for word, (syl1, syl2) in WORD_TO_SYLLABLES.items() if syl1 != "SA"
}
SA_BRANCH_SECOND_SYLLABLE_TO_WORD = {"KIT": "SAKIT", "YANG": "SAYANG"}

SLOT_LINE_RE = re.compile(r"Inject Marker Slot (\d): (\w+) \(ID: (\d+)\)")


def verify_syllable_classes():
    print("[CHECK] SYLLABLE_CLASSES (build_dataset.py, unmodified) vs. expected mapping")
    match = SYLLABLE_CLASSES == EXPECTED_SYLLABLE_CLASSES
    print(f"        Exact match: {match}")
    if not match:
        only_in_code = {k: v for k, v in SYLLABLE_CLASSES.items() if EXPECTED_SYLLABLE_CLASSES.get(k) != v}
        only_in_expected = {k: v for k, v in EXPECTED_SYLLABLE_CLASSES.items() if SYLLABLE_CLASSES.get(k) != v}
        print(f"        [WARNING] Mismatch. In code but differs: {only_in_code}")
        print(f"        [WARNING] Expected but differs: {only_in_expected}")
    return match


def verify_hierarchy_consistency():
    print("\n[CHECK] Vowel-group hierarchy + word dictionary internal consistency")
    first_syllables_from_words = {syl1 for syl1, _ in WORD_TO_SYLLABLES.values()}
    second_syllables_from_words = {syl2 for _, syl2 in WORD_TO_SYLLABLES.values()}
    all_syllables_from_words = first_syllables_from_words | second_syllables_from_words

    ok = True
    if first_syllables_from_words != set(SYLLABLE_TO_VOWEL_GROUP.keys()):
        ok = False
        print(f"        [WARNING] Vowel-group coverage mismatch: {first_syllables_from_words} vs {set(SYLLABLE_TO_VOWEL_GROUP.keys())}")
    else:
        print(f"        Vowel groups cover exactly the 9 first syllables: True")

    if all_syllables_from_words != set(EXPECTED_SYLLABLE_CLASSES.keys()):
        ok = False
        print(f"        [WARNING] Word dictionary syllable coverage != SYLLABLE_CLASSES keys")
    else:
        print(f"        Word dictionary syllables cover exactly the 19 SYLLABLE_CLASSES entries: True")

    n_first, n_second = len(first_syllables_from_words), len(second_syllables_from_words)
    print(f"        First-syllable classes: {n_first} (expected 9) | Second-syllable classes: {n_second} (expected 10)")
    ok = ok and (n_first == 9) and (n_second == 10)

    if set(TARGET_WORDS_UPPER := {w.upper() for w in TARGET_WORDS}) != set(WORD_TO_SYLLABLES.keys()):
        ok = False
        print(f"        [WARNING] config.TARGET_WORDS != WORD_TO_SYLLABLES keys: {TARGET_WORDS_UPPER} vs {set(WORD_TO_SYLLABLES.keys())}")
    else:
        print(f"        Matches config.TARGET_WORDS (10 words): True")

    return ok


def parse_trial_slot_pairs(log_filepath):
    """Extract (word, slot1_syllable, slot1_marker_id, slot2_syllable,
    slot2_marker_id) for every complete trial in one subject's real log.

    Uses the literal on-disk format present in all 12 real subject logs
    ('Menjalankan Trial N/100 (Blok B) - Kata: WORD' followed by two
    'Inject Marker Slot k: SYL (ID: n)' lines) rather than the newer English
    strings emitted by the current acquisition/experiment_runner.py source,
    since every real log on disk predates that rename.
    """
    trials = []
    pending_word = None
    slot1 = None

    with open(log_filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Menjalankan Trial" in line and "Kata:" in line:
                try:
                    pending_word = line.split("Kata: ")[1].split("(")[0].strip().upper()
                except Exception:
                    pending_word = None
                slot1 = None
                continue

            m = SLOT_LINE_RE.search(line)
            if not m or pending_word is None:
                continue
            slot_num, syl, marker_id = int(m.group(1)), m.group(2), int(m.group(3))
            if slot_num == 1:
                slot1 = (syl, marker_id)
            elif slot_num == 2 and slot1 is not None:
                trials.append({
                    "word": pending_word,
                    "slot1_syllable": slot1[0], "slot1_marker_id": slot1[1],
                    "slot2_syllable": syl, "slot2_marker_id": marker_id,
                })
                pending_word, slot1 = None, None

    return trials


def verify_slot_assignment_from_real_data():
    print("\n[CHECK] Slot1=first-syllable / Slot2=second-syllable assignment, verified against real trial data")

    log_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "logs", "*_experiment_log.txt")))
    subject_ids = [os.path.basename(f).replace("_experiment_log.txt", "") for f in log_files]

    seen_as_slot1, seen_as_slot2 = set(), set()
    total_trials = 0
    mismatches = []
    marker_id_offset_errors = []

    for subject_id, log_file in zip(subject_ids, log_files):
        trials = parse_trial_slot_pairs(log_file)
        total_trials += len(trials)
        for t in trials:
            seen_as_slot1.add(t["slot1_syllable"])
            seen_as_slot2.add(t["slot2_syllable"])

            expected = WORD_TO_SYLLABLES.get(t["word"])
            if expected is None or (t["slot1_syllable"], t["slot2_syllable"]) != expected:
                mismatches.append((subject_id, t["word"], t["slot1_syllable"], t["slot2_syllable"], expected))

            # marker_id (1-indexed, as physically injected) must equal
            # SYLLABLE_CLASSES[syllable] + 1 (label_int = marker_value - 1
            # is exactly how build_dataset.py.process_subject decodes it).
            for syl_key, id_key in (("slot1_syllable", "slot1_marker_id"), ("slot2_syllable", "slot2_marker_id")):
                syl, marker_id = t[syl_key], t[id_key]
                expected_id = SYLLABLE_CLASSES.get(syl, -999) + 1
                if marker_id != expected_id:
                    marker_id_offset_errors.append((subject_id, t["word"], syl, marker_id, expected_id))

    print(f"        Subjects parsed: {len(subject_ids)} | Total trials parsed: {total_trials} (expected 12 x 200 = 2400)")
    print(f"        Distinct syllables ever seen in slot 1: {len(seen_as_slot1)} (expected 9)")
    print(f"        Distinct syllables ever seen in slot 2: {len(seen_as_slot2)} (expected 10)")

    overlap = seen_as_slot1 & seen_as_slot2
    print(f"        Syllables seen in BOTH slot 1 and slot 2 (must be empty): {overlap if overlap else 'EMPTY (as expected)'}")

    if mismatches:
        print(f"        [WARNING] {len(mismatches)} trial(s) where slot1/slot2 syllables didn't match WORD_TO_SYLLABLES:")
        for m in mismatches[:10]:
            print(f"                  {m}")
    else:
        print(f"        Word -> (slot1, slot2) matched WORD_TO_SYLLABLES for all {total_trials} trials: True")

    if marker_id_offset_errors:
        print(f"        [WARNING] {len(marker_id_offset_errors)} marker ID(s) inconsistent with SYLLABLE_CLASSES + 1:")
        for e in marker_id_offset_errors[:10]:
            print(f"                  {e}")
    else:
        print(f"        Marker IDs consistent with SYLLABLE_CLASSES[syl] + 1 for all trials: True")

    ok = (not overlap) and (not mismatches) and (not marker_id_offset_errors) and (len(seen_as_slot1) == 9) and (len(seen_as_slot2) == 10)
    return ok, {"subject_ids": subject_ids, "total_trials": total_trials, "mismatches": mismatches}


def run_verification():
    r1 = verify_syllable_classes()
    r2 = verify_hierarchy_consistency()
    r3, detail = verify_slot_assignment_from_real_data()

    print("\n" + "=" * 60)
    all_ok = r1 and r2 and r3
    if all_ok:
        print("[OK] P7 label scheme fully verified against build_dataset.py and real trial data.")
        print("     Label-based slot filtering (no raw-signal re-extraction) is safe to use.")
    else:
        print("[WARNING] One or more P7 label scheme checks did not pass -- review output above")
        print("          before relying on label-based slot filtering in run_p7_coarse_to_fine.py.")
    print("=" * 60)

    return {"syllable_classes_match": r1, "hierarchy_consistent": r2, "slot_assignment_verified": r3, **detail}


if __name__ == "__main__":
    run_verification()

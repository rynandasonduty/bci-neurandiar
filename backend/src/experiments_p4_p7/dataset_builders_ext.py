"""
backend/src/experiments_p4_p7/dataset_builders_ext.py

DatasetBuilder subclasses for P4 (No-Windowing) and P5 (Shifted Bandpass),
plus shared helpers used across P4/P5/P7:
  - the automatic feature-group selection rule (spot-check winner logic)
  - P7's sub-model label filters (coarse/fine_A/fine_I/fine_E/sa_branch)

P6 needs neither a SignalProcessor nor a DatasetBuilder subclass: its
variable is training-data *composition* (imagined-only vs. imagined+overt),
which the unmodified DatasetBuilder already supports natively via its
existing `phase_filter` parameter. P6 calls DatasetBuilder directly with
phase_filter='overt' / 'imagined' -- see run_p6_transfer_overt_imagined.py.

P7 also uses the unmodified DatasetBuilder as-is (identical to P1-P3
methodology) -- only its post-hoc label filtering is new, provided here.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment
from preprocessing.build_dataset import DatasetBuilder
from experiments_p4_p7.signal_processors_ext import (
    FullEpochSignalProcessor,
    ShiftedBandSignalProcessor,
)


class NoWindowDatasetBuilder(DatasetBuilder):
    """P4: swaps in FullEpochSignalProcessor so process_subject() (inherited
    unchanged) extracts one 5-second epoch per marker slot instead of five
    1-second windows. No other method is overridden -- CSV loading, log
    parsing, phase filtering, and channel selection all come from the
    unmodified DatasetBuilder."""

    def __init__(self, exp_id="E0_Baseline", processor_params=None,
                 phase_filter="all", channels_to_use="all", pilar="P4_NoWindowing"):
        super().__init__(
            exp_id=exp_id, processor_params=processor_params, crop_time=None,
            use_augmentation=False, augmentation_params=None,
            phase_filter=phase_filter, channels_to_use=channels_to_use,
        )
        # DatasetBuilder.__init__ always resolves paths under pilar="P1_Global"
        # (it takes no pilar parameter). Redirect to this paradigm's own tree
        # so build_full_dataset(), if ever called, would not write there.
        self.paths = setup_experiment(exp_id, pilar=pilar)
        self.raw_data_dir = self.paths["raw_data"]
        self.output_dir = self.paths["processed_data"]

        # eeg_channels is identical between SignalProcessor and
        # FullEpochSignalProcessor (both inherit it from __init__ unchanged),
        # so self.channel_indices computed above by super().__init__() using
        # the original processor instance remains valid after this swap.
        self.processor = FullEpochSignalProcessor(**(processor_params or {}))


class ShiftedBandDatasetBuilder(DatasetBuilder):
    """P5: swaps in ShiftedBandSignalProcessor (15-65 Hz bandpass) in place
    of the standard SignalProcessor. Same pattern as NoWindowDatasetBuilder;
    windowing, phase filtering, and channel selection are unchanged."""

    def __init__(self, exp_id="E0_Baseline", processor_params=None,
                 phase_filter="all", channels_to_use="all", pilar="P5_ShiftedBandpass"):
        super().__init__(
            exp_id=exp_id, processor_params=processor_params, crop_time=None,
            use_augmentation=False, augmentation_params=None,
            phase_filter=phase_filter, channels_to_use=channels_to_use,
        )
        self.paths = setup_experiment(exp_id, pilar=pilar)
        self.raw_data_dir = self.paths["raw_data"]
        self.output_dir = self.paths["processed_data"]

        self.processor = ShiftedBandSignalProcessor(**(processor_params or {}))


# ---------------------------------------------------------------------------
# Automatic feature-group selection rule (P4, P5, and P7's coarse stage).
# ---------------------------------------------------------------------------

def select_winning_feature_group(spotcheck_results, n_classes=19):
    """
    Pick the winning feature group from a 5-way spot-check, automatically,
    so the pipeline never waits on a manual decision.

    Rule (see README Section 12 / paradigm reports for the full rationale):
      1. Highest test accuracy wins outright.
      2. Tie-break: among candidates within 1 percentage point of the top
         score, prefer 'barlow' if it is one of them; otherwise prefer the
         tied candidate with the highest class coverage.
      3. If the winner does not beat chance level, this is flagged as a
         warning but never blocks the pipeline -- the full-scale stage still
         runs automatically; a human decides later whether the result is
         thesis-worthy.

    Args:
        spotcheck_results (dict): {feature_group: {"test_accuracy": float in
            [0, 1], "n_classes_covered": int}, ...} for the 5 spot-checked
            groups ('time', 'hjorth', 'barlow', 'band_ratio', 'all').
        n_classes (int): number of classes in this classification problem
            (19 for P4/P5's full syllable set, 4 for P7's coarse stage),
            used only to compute the chance-level warning threshold.

    Returns:
        dict: {"winner", "reason", "chance_level_pct", "below_chance_warning",
               "table"} -- fully self-contained record for the paradigm report.
    """
    chance_pct = 100.0 / n_classes
    accs_pct = {g: r["test_accuracy"] * 100.0 for g, r in spotcheck_results.items()}
    top_acc = max(accs_pct.values())
    tied = [g for g, a in accs_pct.items() if (top_acc - a) < 1.0]

    if len(tied) == 1:
        winner, reason = tied[0], "highest test accuracy, no tie within 1pp"
    elif "barlow" in tied:
        winner = "barlow"
        reason = f"tie-break within 1pp among {sorted(tied)}; barlow preferred (robust/consistent across P1-P3)"
    else:
        winner = max(tied, key=lambda g: spotcheck_results[g]["n_classes_covered"])
        reason = (
            f"tie-break within 1pp among {sorted(tied)}, barlow not among tied candidates; "
            f"fallback to highest class coverage ({spotcheck_results[winner]['n_classes_covered']} classes)"
        )

    below_chance = accs_pct[winner] <= chance_pct
    table = [
        {
            "feature_group": g,
            "test_accuracy_pct": accs_pct[g],
            "n_classes_covered": spotcheck_results[g]["n_classes_covered"],
        }
        for g in spotcheck_results
    ]

    return {
        "winner": winner,
        "reason": reason,
        "chance_level_pct": chance_pct,
        "below_chance_warning": bool(below_chance),
        "table": table,
    }


# ---------------------------------------------------------------------------
# P7 sub-model label filters. Ground truth per Langkah 0.4 / verified against
# real trial data in verify_p7_label_scheme.py -- fixed reference constants,
# not derived or configurable.
# ---------------------------------------------------------------------------

# First-syllable label_int -> vowel group letter (from SYLLABLE_CLASSES).
VOWEL_GROUP_OF_LABEL = {
    0: "A", 8: "A", 14: "A",   # MA, MAN, SA
    2: "I", 6: "I", 16: "I",   # MI, PI, TI
    4: "E", 12: "E",           # BE, LE
    10: "O",                   # BO (single member, no fine-stage model)
}
VOWEL_GROUP_TO_ID = {"A": 0, "I": 1, "E": 2, "O": 3}
ID_TO_VOWEL_GROUP = {v: k for k, v in VOWEL_GROUP_TO_ID.items()}

# Label sets each sub-model is trained on, filtered from the SAME shared
# three_way_split of the standard 19-class E0/Barlow dataset.
SUBMODEL_LABEL_SETS = {
    "coarse": {0, 2, 4, 6, 8, 10, 12, 14, 16},  # 9 first-syllable classes
    "fine_A": {0, 8, 14},                        # MA, MAN, SA
    "fine_I": {2, 6, 16},                        # MI, PI, TI
    "fine_E": {4, 12},                           # BE, LE
    "sa_branch": {15, 18},                       # KIT, YANG (second syllable, SA branch only)
}

# Deterministic first-syllable -> word dictionary (no ML needed). SA is
# intentionally absent: it is the one ambiguous branch requiring sa_branch.
DETERMINISTIC_FIRST_SYLLABLE_TO_WORD = {
    "MA": "MAKAN", "MI": "MINUM", "BE": "BERAK", "PI": "PIPIS", "MAN": "MANDI",
    "BO": "BOSAN", "LE": "LELAH", "TI": "TIDUR",
}
SA_BRANCH_SECOND_SYLLABLE_TO_WORD = {"KIT": "SAKIT", "YANG": "SAYANG"}

LABEL_TO_SYLLABLE = {
    0: "MA", 1: "KAN", 2: "MI", 3: "NUM", 4: "BE", 5: "RAK", 6: "PI", 7: "PIS",
    8: "MAN", 9: "DI", 10: "BO", 11: "SAN", 12: "LE", 13: "LAH", 14: "SA",
    15: "KIT", 16: "TI", 17: "DUR", 18: "YANG",
}


def filter_split_by_labels(X, y, label_set):
    """Filter an (X, y) split down to samples whose label is in label_set.

    Order-preserving boolean mask -- no re-splitting or shuffling. This is
    what lets every P7 sub-model be derived from the SAME shared
    three_way_split rather than an independent random split per sub-model.
    """
    mask = np.isin(y, list(label_set))
    return X[mask], y[mask]


def map_labels_to_vowel_group_ids(y):
    """Map first-syllable label_int values to vowel-group ids (0=A,1=I,2=E,3=O)
    for training/evaluating the `coarse` sub-model."""
    return np.array([VOWEL_GROUP_TO_ID[VOWEL_GROUP_OF_LABEL[int(v)]] for v in y])

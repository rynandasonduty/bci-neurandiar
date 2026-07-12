"""
backend/src/experiments_p4_p7/fair_comparison/_common.py

Shared paths, constants, and small helpers for the P3-vs-P6 fair-comparison
analysis scripts (Plan B prompt, Fase 1 / Poin 0). Every helper here loads
already-trained artefacts (.pkl models, .npy test splits, .json results) and
performs inference/analysis only -- nothing in this package calls .fit() on
anything. See backend/reports/P4_P7_Experiments/P7_CoarseToFine_report.md
for the paradigm this analyses (on-disk folder name is still
"P7_CoarseToFine" / tag "P7_*", predating the P6 rename used in the thesis).
"""
import os
import sys
import json
import pickle

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from config import MODELS_DIR, RAW_DATA_DIR  # noqa: E402
from experiments_p4_p7.dataset_builders_ext import (  # noqa: E402
    SUBMODEL_LABEL_SETS, VOWEL_GROUP_TO_ID, ID_TO_VOWEL_GROUP, LABEL_TO_SYLLABLE,
    VOWEL_GROUP_OF_LABEL, DETERMINISTIC_FIRST_SYLLABLE_TO_WORD, SA_BRANCH_SECOND_SYLLABLE_TO_WORD,
)

WEIGHTS_ROOT = os.path.join(MODELS_DIR, "weights")
WEIGHTS_P3 = os.path.join(WEIGHTS_ROOT, "P3_SVM")
WEIGHTS_P4 = os.path.join(WEIGHTS_ROOT, "P4_NoWindowing")
WEIGHTS_P5 = os.path.join(WEIGHTS_ROOT, "P5_ShiftedBandpass")
WEIGHTS_P6 = os.path.join(WEIGHTS_ROOT, "P7_CoarseToFine")  # on-disk name predates the P6 rename
P6_FULLSCALE = os.path.join(WEIGHTS_P6, "Fullscale_12Subj")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

T_TABLES_DIR = os.path.abspath(os.path.join(
    _SRC_DIR, '..', '..', 'notebooks', 'reports', 'data_export_claude'
))

ALL_SUBJECTS = [f"S{i}" for i in range(1, 13)]

P3_CHAMPION_SUBJECT = "S3"
P3_CHAMPION_EXP = "E5_Data_Augmentation"
P3_CHAMPION_FEATURE_GROUP = "barlow"

P6_CHAMPION_SUBJECT = "S3"  # ATURAN #0 point 4: chosen for highest first-syllable e2e, not S9's peak full-word e2e

SUBMODEL_NAMES = ("coarse", "fine_A", "fine_I", "fine_E", "sa_branch")
FINE_SUBMODEL_OF_GROUP = {"A": "fine_A", "I": "fine_I", "E": "fine_E"}

EMOTIV_CH = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
FEAT_PER_CH = {'time': 4, 'hjorth': 3, 'barlow': 2, 'band_ratio': 3, 'all': 12}
FEAT_SUBNAMES = {
    'time': ['mean', 'var', 'skew', 'kurt'],
    'hjorth': ['activity', 'mobility', 'complexity'],
    'barlow': ['amp', 'freq'],
    'band_ratio': ['alpha/theta', 'beta/alpha', 'gamma/beta'],
    'all': ['mean', 'var', 'skew', 'kurt', 'activity', 'mobility', 'complexity',
            'amp', 'freq', 'alpha/theta', 'beta/alpha', 'gamma/beta'],
}


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[SAVED] {path}")
    return path


def p6_submodel_paths(subject_id, submodel, fullscale_root=None):
    """(model_path, scaler_path, xtest_path, ytest_path) for one P6 (on-disk
    P7_CoarseToFine) sub-model's Stage B full-scale artefacts. Xtest/ytest
    are ALREADY feature-extracted (Barlow) and scaled by that sub-model's
    OWN fitted scaler -- do not re-apply the scaler before predict()."""
    root = fullscale_root or P6_FULLSCALE
    d = os.path.join(root, submodel)
    tag = f"P7_{submodel}"
    return (
        os.path.join(d, f"SVM_{tag}_barlow_{subject_id}.pkl"),
        os.path.join(d, f"scaler_{tag}_barlow_{subject_id}.pkl"),
        os.path.join(d, f"Xtest_{tag}_barlow_{subject_id}.npy"),
        os.path.join(d, f"ytest_{tag}_barlow_{subject_id}.npy"),
    )


def p3_champion_paths(subject_id, exp_id=P3_CHAMPION_EXP, feature_group=P3_CHAMPION_FEATURE_GROUP):
    """Xtest/ytest here are ALREADY scaled by fit_and_apply_scaler at
    training time -- call model.predict(X) directly, never re-apply
    scaler.transform() first (see memory: bci-svm-xtest-prescaled)."""
    d = os.path.join(WEIGHTS_P3, exp_id)
    return (
        os.path.join(d, f"SVM_{feature_group}_{exp_id}_{subject_id}.pkl"),
        os.path.join(d, f"scaler_SVM_{feature_group}_{exp_id}_{subject_id}.pkl"),
        os.path.join(d, f"Xtest_SVM_{feature_group}_{exp_id}_{subject_id}.npy"),
        os.path.join(d, f"ytest_SVM_{feature_group}_{exp_id}_{subject_id}.npy"),
    )


def p6_results_json_path(subject_id, fullscale_root=None):
    root = fullscale_root or P6_FULLSCALE
    return os.path.join(root, f"results_{subject_id}.json")


def check_exists(*paths):
    """Return the subset of paths that do NOT exist (empty list = all present)."""
    return [p for p in paths if not os.path.exists(p)]


def load_classical_classifier(model_path):
    """Load a ClassicalClassifier-saved SVM pickle (the pipeline object
    itself, per ClassicalClassifier.save_model/load_model)."""
    return load_pickle(model_path)


def p6_load_bundles(subject_id, fullscale_root=None):
    """Load all 5 P6 (on-disk P7_CoarseToFine) sub-model bundles for one
    subject, ready for predict_first_syllable()/predict_word_for_trial().
    Deferred import of run_p7_coarse_to_fine so callers that only need path
    helpers from this module don't pay its import cost."""
    from experiments_p4_p7.run_p7_coarse_to_fine import SubModelBundle, E0_PROCESSOR_PARAMS
    results = load_json(p6_results_json_path(subject_id, fullscale_root))
    coarse_feat = results.get("winning_coarse_feature_group", "barlow")
    fs = E0_PROCESSOR_PARAMS["target_fs"]

    m, s, _, _ = p6_submodel_paths(subject_id, "coarse", fullscale_root)
    coarse_bundle = SubModelBundle(m, s, coarse_feat, fs=fs)

    fine_bundles = {}
    for grp, name in FINE_SUBMODEL_OF_GROUP.items():
        m, s, _, _ = p6_submodel_paths(subject_id, name, fullscale_root)
        fine_bundles[grp] = SubModelBundle(m, s, "barlow", fs=fs)

    m, s, _, _ = p6_submodel_paths(subject_id, "sa_branch", fullscale_root)
    sa_bundle = SubModelBundle(m, s, "barlow", fs=fs)

    return coarse_bundle, fine_bundles, sa_bundle

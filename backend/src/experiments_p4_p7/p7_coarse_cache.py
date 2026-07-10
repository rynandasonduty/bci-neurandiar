"""
backend/src/experiments_p4_p7/p7_coarse_cache.py

Caching layer for P7's coarse sub-model ablation work (run_p7_coarse_ablation.py,
run_p7_coarse_combined.py, run_p7_postprocessing.py). Building the coarse
sub-model's shared split from raw CSV (DatasetBuilder + three_way_split +
label filtering) is the single most expensive step in the ablation pipeline
-- reading and windowing 12 subjects' raw EEG is the same order of cost as
the original P7 Stage B run itself. Caching it ONCE per subject, on disk,
means every downstream variant script (and every resumed/re-run process)
reuses the same raw split and Barlow features without re-touching the CSVs.

Three cache layers are saved per subject, all BEFORE scaling (scaling must
always be fit fresh per training-set composition, since some variants
enrich/alter the training set -- see run_p7_coarse_ablation.py's Varian B):

1. Raw 3D epochs (samples, channels, time), coarse-filtered (9
   first-syllable classes) -- X_{split}_3d_coarse. Needed by any variant
   that touches the RAW signal before feature extraction (Varian B's
   augmentation, Varian D's alternate feature group).
2. Labels for those same epochs, in TWO parallel forms: y_{split}_coarse
   (mapped to vowel-group ids, 0=A/1=I/2=E/3=O -- the actual training
   target for the coarse classifier, used directly by every ablation
   variant) and y_{split}_coarse_syllable (the original, unmapped 9-class
   first-syllable label, e.g. 0/2/4/6/8/10/12/14/16 -- needed by
   run_p7_postprocessing.py to score confidence-gated coarse->fine
   predictions against actual syllable identity, not just vowel group).
   Both arrays share the same sample order/mask, so index i in either
   corresponds to the same epoch as index i in X_{split}_3d_coarse.
3. Barlow features extracted from those same raw epochs, UNSCALED --
   X_{split}_feat_barlow. Needed by every variant that reuses Barlow
   features as-is (Varian A, C, E, and the unaugmented portion of Varian B).

Uses the exact same DatasetBuilder call (via run_p7_coarse_to_fine.py's
build_standard_e0_split_raw, imported read-only) and three_way_split(seed=42)
that run_p7_coarse_to_fine.py's Stage B already used for the `coarse`
sub-model, then the same coarse label-filtering
(dataset_builders_ext.filter_split_by_labels + map_labels_to_vowel_group_ids)
-- so the cached split is identical to what the existing `coarse` baseline
(P7_CoarseToFine/Fullscale_12Subj/coarse/) was trained on. Neither
run_p7_coarse_to_fine.py nor dataset_builders_ext.py is modified, only
imported.

Cache location: backend/models/weights/P7_CoarseToFine/_cache/{subject_id}/
(built via the existing, unmodified setup_experiment() helper, treating
"_cache" as a sibling experiment id under the P7_CoarseToFine pilar --
same convention every other P4-P7 stage directory already uses).
"""
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.extract_eeg_features import EEGFeatureExtractor
from utils.data_utils import three_way_split

from experiments_p4_p7 import run_p7_coarse_to_fine as p7base
from experiments_p4_p7.dataset_builders_ext import (
    SUBMODEL_LABEL_SETS, filter_split_by_labels, map_labels_to_vowel_group_ids,
)

CACHE_EXP_ID = "_cache"

# split -> (3D-array key, vowel-group-id label key, raw-syllable label key, Barlow-feature key)
_SPLIT_KEYS = {
    "train": ("X_train_3d_coarse", "y_train_coarse", "y_train_coarse_syllable", "X_train_feat_barlow"),
    "val":   ("X_val_3d_coarse",   "y_val_coarse",   "y_val_coarse_syllable",   "X_val_feat_barlow"),
    "test":  ("X_test_3d_coarse",  "y_test_coarse",  "y_test_coarse_syllable",  "X_test_feat_barlow"),
}
# Flat list of every array name this module caches, for load/exists checks.
ALL_CACHE_KEYS = [key for keys in _SPLIT_KEYS.values() for key in keys]


def get_cache_dir(subject_id):
    cache_root = p7base.setup_experiment(CACHE_EXP_ID, pilar=p7base.PILAR)["weights"]
    d = os.path.join(cache_root, subject_id)
    os.makedirs(d, exist_ok=True)
    return d


def _cache_paths(subject_id):
    d = get_cache_dir(subject_id)
    return {key: os.path.join(d, f"{key}.npy") for key in ALL_CACHE_KEYS}


def load_cached_coarse(subject_id):
    """Returns a dict of all 12 cached arrays (3 splits x {X_3d_coarse,
    y_coarse, y_coarse_syllable, X_feat_barlow}) if the full cache exists
    for this subject, else None (cache miss -- caller should build it, see
    build_and_cache_coarse / get_or_build_cached_coarse)."""
    paths = _cache_paths(subject_id)
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    return {key: np.load(p) for key, p in paths.items()}


def build_and_cache_coarse(subject_id):
    """Rebuilds the coarse sub-model's shared split from raw CSV (same
    DatasetBuilder + three_way_split(seed=42) call as
    run_p7_coarse_to_fine.py's Stage B), filters to the 9 coarse classes,
    maps labels to vowel-group ids, extracts unscaled Barlow features, and
    saves all of it to disk. Returns the same dict shape as
    load_cached_coarse(), or None if no raw epochs could be built for this
    subject (mirrors build_standard_e0_split_raw's own None-on-missing-data
    contract, e.g. missing raw CSV/log for that subject)."""
    X_3d, y = p7base.build_standard_e0_split_raw(subject_id)
    if X_3d is None:
        return None

    X_train_3d, X_val_3d, X_test_3d, y_train, y_val, y_test = three_way_split(
        X_3d, y, random_state=p7base.SPLIT_RANDOM_STATE
    )
    raw_splits = {"train": (X_train_3d, y_train), "val": (X_val_3d, y_val), "test": (X_test_3d, y_test)}

    coarse_labels = SUBMODEL_LABEL_SETS["coarse"]
    extractor = EEGFeatureExtractor(fs=p7base.E0_PROCESSOR_PARAMS["target_fs"])
    paths = _cache_paths(subject_id)
    result = {}

    for split, (X_split_3d, y_split) in raw_splits.items():
        x3d_key, y_key, y_syl_key, feat_key = _SPLIT_KEYS[split]

        X_filtered_3d, y_filtered_syllable = filter_split_by_labels(X_split_3d, y_split, coarse_labels)
        y_filtered_vowel = map_labels_to_vowel_group_ids(y_filtered_syllable)

        feat = extractor.transform(X_filtered_3d, groups=["barlow"])
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        np.save(paths[x3d_key], X_filtered_3d)
        np.save(paths[y_key], y_filtered_vowel)
        np.save(paths[y_syl_key], y_filtered_syllable)
        np.save(paths[feat_key], feat)

        result[x3d_key] = X_filtered_3d
        result[y_key] = y_filtered_vowel
        result[y_syl_key] = y_filtered_syllable
        result[feat_key] = feat

    return result


def get_or_build_cached_coarse(subject_id, force_rebuild=False):
    """Main entry point for downstream scripts: load from disk if present,
    else build once and persist. Returns None only if no raw epochs exist
    for this subject at all (genuine missing-data case, not a cache miss)."""
    if not force_rebuild:
        cached = load_cached_coarse(subject_id)
        if cached is not None:
            return cached
    return build_and_cache_coarse(subject_id)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pre-warm the P7 coarse sub-model's raw-split + Barlow-feature cache for one "
                     "or more subjects. Optional standalone step -- run_p7_coarse_ablation.py also "
                     "builds the cache lazily on first use per subject if it isn't warmed yet, so "
                     "running this script first only matters if you want the raw-CSV-read cost paid "
                     "up front instead of during the first ablation variant."
    )
    parser.add_argument("--subjects", nargs="+", default=None,
                         help="Restrict to specific subject IDs (e.g. --subjects S1 S2). "
                              "Default: all 12 subjects, auto-discovered.")
    parser.add_argument("--force-rebuild", action="store_true",
                         help="Rebuild even if a cache already exists for a subject.")
    args = parser.parse_args()

    subject_ids = args.subjects or p7base.discover_subject_ids()
    for subject_id in subject_ids:
        print(f"[INFO][P7-Cache] {subject_id}: building/loading cache...")
        cached = get_or_build_cached_coarse(subject_id, force_rebuild=args.force_rebuild)
        if cached is None:
            print(f"[WARNING][P7-Cache] {subject_id}: no raw epochs found -- skipped.")
        else:
            print(f"[INFO][P7-Cache] {subject_id}: cache ready "
                  f"(n_train={len(cached['y_train_coarse'])}, n_val={len(cached['y_val_coarse'])}, "
                  f"n_test={len(cached['y_test_coarse'])}).")

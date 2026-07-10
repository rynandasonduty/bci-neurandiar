"""
backend/src/experiments_p4_p7/run_p7_coarse_improve.py

P7 coarse sub-model improvement: two isolated variants trained ONLY for the
`coarse` sub-model. fine_A/fine_I/fine_E/sa_branch are NEVER retrained here
-- they are loaded read-only from the existing
P7_CoarseToFine/Fullscale_12Subj/{fine_A,fine_I,fine_E,sa_branch}/ artifacts
produced by run_p7_coarse_to_fine.py's Stage B. The coarse stage's 9
first-syllable classes collapse into 4 unbalanced vowel groups
(A={MA,MAN,SA}=3, I={MI,PI,TI}=3, E={BE,LE}=2, O={BO}=1 syllables) and is the
pipeline's bottleneck -- a coarse misclassification automatically breaks the
fine stage below it, so this is the highest-leverage place to intervene.

Varian A -- Class-weight balanced (`coarse_e0_balanced/`):
    WeightedClassicalClassifier (classical_models_ext.py -- new file,
    classical_models.py itself is never touched) adds class_weight='balanced'
    to the SVC constructor, otherwise identical to the existing coarse
    baseline: same E0 dataset, same three_way_split(seed=42), same feature
    group, same C=10.

Varian B -- E5 augmentation (`coarse_e5_augmented/`):
    Plain (unweighted) ClassicalClassifier. Training data enriched via
    SignalProcessor.apply_augmentation() (called, never modified) using
    EXPERIMENT_RECIPES["E5_Data_Augmentation"]'s augmentation_params from
    models/run_subject_dependent.py (read-only reference -- copied as a
    constant below rather than imported, to avoid pulling in that module's
    mlflow/tensorflow dependency chain just for a dict literal):
    {"add_noise": True, "noise_factor": 0.05, "apply_jitter": True,
    "jitter_ms": 10}. Applied to TRAINING data only, after the split -- val
    and test are the same untouched epochs used everywhere else in P7.

Both variants are derived from EXACTLY the same per-subject
three_way_split(seed=42) on the standard E0/19-class dataset that
run_p7_coarse_to_fine.py's Stage B already used (identical DatasetBuilder
call + identical seed + identical raw data => identical split), then
filtered down to the 9 coarse label classes with the same
dataset_builders_ext.filter_split_by_labels / map_labels_to_vowel_group_ids
helpers Stage B already uses.

After both variants are trained for all requested subjects, the winning
coarse candidate (baseline E0 vs. Varian A vs. Varian B) is selected
automatically by mean test accuracy (see select_winning_coarse_variant's
docstring for the exact tie-break rule), and both end-to-end metrics
(first-syllable, full-word) are recomputed using the winning coarse
sub-model combined with the EXISTING fine_A/fine_I/fine_E/sa_branch
sub-models -- never retrained.

NOTE (researcher instruction): this script performs real SVM training on raw
12-subject EEG data and is meant to be run on the lab PC, the same way
run_p7_coarse_to_fine.py itself was -- not on the memory-constrained dev
machine this code was written on. It was verified by static review and a
module-import check only; it has not been executed end-to-end here.

Usage:
    cd backend/src/experiments_p4_p7
    python run_p7_coarse_improve.py                 # all 12 subjects
    python run_p7_coarse_improve.py --subjects S1 S2
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.extract_eeg_features import EEGFeatureExtractor
from models.classical_models import ClassicalClassifier
from utils.data_utils import three_way_split, fit_and_apply_scaler
from preprocessing.signal_processor import SignalProcessor

from experiments_p4_p7 import run_p7_coarse_to_fine as p7base
from experiments_p4_p7.classical_models_ext import WeightedClassicalClassifier
from experiments_p4_p7.dataset_builders_ext import (
    SUBMODEL_LABEL_SETS, filter_split_by_labels, map_labels_to_vowel_group_ids,
)

VARIANT_A_DIR = "coarse_e0_balanced"
VARIANT_B_DIR = "coarse_e5_augmented"

# EXPERIMENT_RECIPES["E5_Data_Augmentation"]["augmentation_params"],
# models/run_subject_dependent.py -- read-only reference, copied verbatim.
E5_AUGMENTATION_PARAMS = {"add_noise": True, "noise_factor": 0.05, "apply_jitter": True, "jitter_ms": 10}


def augment_training_epochs(X_3d, proc, aug_params):
    """Apply SignalProcessor.apply_augmentation() to each training epoch.

    X_3d is (samples, channels, time) -- the convention used throughout
    P4-P7 (see EEGFeatureExtractor.transform). apply_augmentation expects
    (time, channels), so each sample is transposed in and back out."""
    aug_list = [proc.apply_augmentation(sample.T, **aug_params).T for sample in X_3d]
    return np.array(aug_list, dtype=X_3d.dtype)


def train_variant_a(Xtr_3d, ytr, Xva_3d, yva, Xte_3d, yte, extractor, output_dir, subject_id, feat_group):
    """Varian A: WeightedClassicalClassifier (class_weight='balanced'), same
    unaugmented coarse split as the baseline."""
    os.makedirs(output_dir, exist_ok=True)
    groups = [feat_group]

    Xtr_feat = np.nan_to_num(extractor.transform(Xtr_3d, groups=groups), nan=0.0, posinf=0.0, neginf=0.0)
    Xva_feat = np.nan_to_num(extractor.transform(Xva_3d, groups=groups), nan=0.0, posinf=0.0, neginf=0.0)
    Xte_feat = np.nan_to_num(extractor.transform(Xte_3d, groups=groups), nan=0.0, posinf=0.0, neginf=0.0)

    scaler_path = os.path.join(output_dir, f"scaler_P7_coarse_e0_balanced_{feat_group}_{subject_id}.pkl")
    Xtr_s, Xva_s, Xte_s, _ = fit_and_apply_scaler(Xtr_feat, Xva_feat, Xte_feat, save_path=scaler_path)

    np.save(os.path.join(output_dir, f"Xtest_P7_coarse_e0_balanced_{feat_group}_{subject_id}.npy"), Xte_s)
    np.save(os.path.join(output_dir, f"ytest_P7_coarse_e0_balanced_{feat_group}_{subject_id}.npy"), yte)

    model = WeightedClassicalClassifier(model_type='svm', C=10)
    model.train(Xtr_s, ytr)
    val_acc = model.evaluate(Xva_s, yva)
    test_acc = model.evaluate(Xte_s, yte)

    model_path = os.path.join(output_dir, f"SVM_P7_coarse_e0_balanced_{feat_group}_{subject_id}.pkl")
    model.save_model(model_path)

    y_pred = model.pipeline.predict(Xte_s)
    classes_covered = sorted(set(np.asarray(yte)[y_pred == yte].tolist()))

    return {
        "variant": "E0_balanced", "feature_group": feat_group, "subject_id": subject_id,
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "val_accuracy": float(val_acc), "test_accuracy": float(test_acc),
        "n_classes_covered": len(classes_covered), "classes_covered": classes_covered,
        "model_path": model_path, "scaler_path": scaler_path,
    }


def train_variant_b(Xtr_3d, ytr, Xva_3d, yva, Xte_3d, yte, extractor, output_dir, subject_id, feat_group):
    """Varian B: plain ClassicalClassifier, coarse training set enriched with
    E5-recipe augmented copies (noise + jitter). val/test are untouched."""
    os.makedirs(output_dir, exist_ok=True)
    groups = [feat_group]

    proc = SignalProcessor(target_fs=p7base.E0_PROCESSOR_PARAMS["target_fs"])
    X_aug_3d = augment_training_epochs(Xtr_3d, proc, E5_AUGMENTATION_PARAMS)
    X_train_enriched_3d = np.concatenate([Xtr_3d, X_aug_3d], axis=0)
    y_train_enriched = np.concatenate([ytr, ytr], axis=0)

    # Shuffle after concatenation, mirroring EXPERIMENT_RECIPES["E5_Data_Augmentation"]'s
    # own pattern in run_subject_dependent.py. SVC's solution does not depend
    # on training-sample order, so this is purely a fidelity-to-precedent
    # step, not a correctness requirement.
    shuffle_idx = np.random.permutation(len(y_train_enriched))
    X_train_enriched_3d = X_train_enriched_3d[shuffle_idx]
    y_train_enriched = y_train_enriched[shuffle_idx]

    Xtr_feat = np.nan_to_num(extractor.transform(X_train_enriched_3d, groups=groups), nan=0.0, posinf=0.0, neginf=0.0)
    Xva_feat = np.nan_to_num(extractor.transform(Xva_3d, groups=groups), nan=0.0, posinf=0.0, neginf=0.0)
    Xte_feat = np.nan_to_num(extractor.transform(Xte_3d, groups=groups), nan=0.0, posinf=0.0, neginf=0.0)

    scaler_path = os.path.join(output_dir, f"scaler_P7_coarse_e5_augmented_{feat_group}_{subject_id}.pkl")
    Xtr_s, Xva_s, Xte_s, _ = fit_and_apply_scaler(Xtr_feat, Xva_feat, Xte_feat, save_path=scaler_path)

    np.save(os.path.join(output_dir, f"Xtest_P7_coarse_e5_augmented_{feat_group}_{subject_id}.npy"), Xte_s)
    np.save(os.path.join(output_dir, f"ytest_P7_coarse_e5_augmented_{feat_group}_{subject_id}.npy"), yte)

    model = ClassicalClassifier(model_type='svm', C=10)
    model.train(Xtr_s, y_train_enriched)
    val_acc = model.evaluate(Xva_s, yva)
    test_acc = model.evaluate(Xte_s, yte)

    model_path = os.path.join(output_dir, f"SVM_P7_coarse_e5_augmented_{feat_group}_{subject_id}.pkl")
    model.save_model(model_path)

    y_pred = model.pipeline.predict(Xte_s)
    classes_covered = sorted(set(np.asarray(yte)[y_pred == yte].tolist()))

    return {
        "variant": "E5_augmented", "feature_group": feat_group, "subject_id": subject_id,
        "n_train_original": int(len(ytr)), "n_train_augmented_total": int(len(y_train_enriched)),
        "n_val": int(len(yva)), "n_test": int(len(yte)),
        "val_accuracy": float(val_acc), "test_accuracy": float(test_acc),
        "n_classes_covered": len(classes_covered), "classes_covered": classes_covered,
        "model_path": model_path, "scaler_path": scaler_path,
    }


def select_winning_coarse_variant(per_subject):
    """Automatic selection among 3 coarse-stage candidates: baseline (E0,
    unweighted -- already trained by run_p7_coarse_to_fine.py, not retrained
    here), Varian A (E0 + class_weight='balanced'), Varian B (E5
    augmentation, unweighted).

    Follows the same principle as dataset_builders_ext.select_winning_
    feature_group(): highest mean test accuracy wins outright; candidates
    within 1 percentage point of the top score are treated as tied, and the
    tie is broken by preferring the simplest/most standard candidate
    (baseline over either variant, Varian A over Varian B if the two
    variants are tied with each other and both beat baseline) -- the same
    "prefer the robust/conservative default when scores are statistically
    indistinguishable" philosophy that function applies to feature-group
    identity, applied here to variant complexity instead.
    """
    subjects = sorted(per_subject.keys())
    baseline_accs = [per_subject[s]["baseline_e0_test_accuracy"] * 100.0 for s in subjects]
    a_accs = [per_subject[s]["variant_a_e0_balanced"]["test_accuracy"] * 100.0 for s in subjects]
    b_accs = [per_subject[s]["variant_b_e5_augmented"]["test_accuracy"] * 100.0 for s in subjects]

    means = {
        "baseline_e0": float(np.mean(baseline_accs)),
        "variant_a_e0_balanced": float(np.mean(a_accs)),
        "variant_b_e5_augmented": float(np.mean(b_accs)),
    }
    stds = {
        "baseline_e0": float(np.std(baseline_accs, ddof=1)) if len(baseline_accs) > 1 else 0.0,
        "variant_a_e0_balanced": float(np.std(a_accs, ddof=1)) if len(a_accs) > 1 else 0.0,
        "variant_b_e5_augmented": float(np.std(b_accs, ddof=1)) if len(b_accs) > 1 else 0.0,
    }

    top_val = max(means.values())
    tied = [name for name, val in means.items() if (top_val - val) < 1.0]
    priority = ["baseline_e0", "variant_a_e0_balanced", "variant_b_e5_augmented"]

    if len(tied) == 1:
        winner = tied[0]
        reason = f"highest mean test accuracy ({means[winner]:.2f}%), no tie within 1pp"
    else:
        winner = min(tied, key=priority.index)
        tied_means = {n: round(means[n], 2) for n in tied}
        reason = (f"tie-break within 1pp among {sorted(tied)} (means: {tied_means}); preferring the "
                  f"simplest/most standard candidate among the tied set (baseline > class-weight "
                  f"balancing > data augmentation), same conservative-default principle as "
                  f"dataset_builders_ext.select_winning_feature_group()'s barlow tie-break")

    return {
        "winner": winner,
        "reason": reason,
        "means_pct": means,
        "stds_pct": stds,
        "n_subjects": len(subjects),
        "per_candidate_pct": {
            "baseline_e0": baseline_accs,
            "variant_a_e0_balanced": a_accs,
            "variant_b_e5_augmented": b_accs,
        },
        "subjects": subjects,
    }


def _winning_coarse_bundle_paths(winner, fullscale_root, feat_group, subject_id):
    if winner == "baseline_e0":
        d = os.path.join(fullscale_root, "coarse")
        tag = "P7_coarse"
    elif winner == "variant_a_e0_balanced":
        d = os.path.join(fullscale_root, VARIANT_A_DIR)
        tag = "P7_coarse_e0_balanced"
    elif winner == "variant_b_e5_augmented":
        d = os.path.join(fullscale_root, VARIANT_B_DIR)
        tag = "P7_coarse_e5_augmented"
    else:
        raise ValueError(f"Unknown coarse variant winner: {winner!r}")
    return (
        os.path.join(d, f"SVM_{tag}_{feat_group}_{subject_id}.pkl"),
        os.path.join(d, f"scaler_{tag}_{feat_group}_{subject_id}.pkl"),
    )


def _load_fine_and_sa_bundles(fullscale_root, subject_id):
    fine_bundles = {}
    for letter, name in (("A", "fine_A"), ("I", "fine_I"), ("E", "fine_E")):
        d = os.path.join(fullscale_root, name)
        model_path = os.path.join(d, f"SVM_P7_{name}_barlow_{subject_id}.pkl")
        scaler_path = os.path.join(d, f"scaler_P7_{name}_barlow_{subject_id}.pkl")
        fine_bundles[letter] = p7base.SubModelBundle(model_path, scaler_path, "barlow",
                                                       fs=p7base.E0_PROCESSOR_PARAMS["target_fs"])

    sa_dir = os.path.join(fullscale_root, "sa_branch")
    sa_model_path = os.path.join(sa_dir, f"SVM_P7_sa_branch_barlow_{subject_id}.pkl")
    sa_scaler_path = os.path.join(sa_dir, f"scaler_P7_sa_branch_barlow_{subject_id}.pkl")
    sa_bundle = p7base.SubModelBundle(sa_model_path, sa_scaler_path, "barlow",
                                       fs=p7base.E0_PROCESSOR_PARAMS["target_fs"])
    return fine_bundles, sa_bundle


def recompute_end_to_end_metrics(per_subject, selection, fullscale_root, feat_group, e2e_inputs):
    """Recompute first-syllable + full-word end-to-end accuracy using the
    winning coarse sub-model, combined with the EXISTING (never retrained)
    fine_A/fine_I/fine_E/sa_branch sub-models.

    e2e_inputs: {subject_id: (X_test_3d_full_19class, y_test_full_19class)}
    captured from the SAME three_way_split rebuilt while training the
    variants above -- reused here instead of re-reading raw CSVs a second
    time for the same subject.
    """
    winner = selection["winner"]
    print(f"\n[INFO][P7-Improve] Recomputing end-to-end metrics using winning coarse variant: {winner}")

    e2e_results = {}
    for subject_id in selection["subjects"]:
        e2e_json_path = os.path.join(fullscale_root, f"improved_e2e_{winner}_{subject_id}.json")
        if os.path.exists(e2e_json_path):
            print(f"[SKIP][P7-Improve] Improved e2e for {subject_id} already computed.")
            with open(e2e_json_path) as f:
                e2e_results[subject_id] = json.load(f)
            continue

        if winner == "baseline_e0":
            # Coarse sub-model is literally unchanged from the original P7
            # Stage B run -- reuse its already-computed e2e figures verbatim
            # instead of redundantly recomputing an identical result.
            with open(os.path.join(fullscale_root, f"results_{subject_id}.json")) as f:
                original = json.load(f)
            result = {
                "subject_id": subject_id, "coarse_variant": winner,
                "first_syllable_e2e": original["first_syllable_e2e"],
                "full_word_e2e": original["full_word_e2e"],
                "note": "baseline coarse sub-model won the selection (unchanged) -- e2e figures reused "
                        "verbatim from the original P7_CoarseToFine Stage B run, not recomputed.",
            }
        else:
            X_test_3d, y_test = e2e_inputs[subject_id]
            coarse_model_path, coarse_scaler_path = _winning_coarse_bundle_paths(
                winner, fullscale_root, feat_group, subject_id
            )
            coarse_bundle = p7base.SubModelBundle(coarse_model_path, coarse_scaler_path, feat_group,
                                                    fs=p7base.E0_PROCESSOR_PARAMS["target_fs"])
            fine_bundles, sa_bundle = _load_fine_and_sa_bundles(fullscale_root, subject_id)

            first_syl_e2e = p7base.compute_first_syllable_e2e_accuracy(X_test_3d, y_test, coarse_bundle, fine_bundles)
            full_word_e2e = p7base.compute_full_word_e2e_accuracy(subject_id, coarse_bundle, fine_bundles, sa_bundle)

            result = {
                "subject_id": subject_id, "coarse_variant": winner,
                "first_syllable_e2e": first_syl_e2e, "full_word_e2e": full_word_e2e,
            }

        e2e_results[subject_id] = result
        with open(e2e_json_path, "w") as f:
            json.dump(result, f, indent=2)
        fw = result["full_word_e2e"]
        fw_str = f"{fw['accuracy']*100:.2f}%" if fw.get("available") else "n/a"
        print(f"[INFO][P7-Improve] {subject_id}: improved first-syllable e2e "
              f"{result['first_syllable_e2e']['accuracy']*100:.2f}% | full-word e2e {fw_str}")

    return e2e_results


def write_improvement_report(per_subject, selection, e2e_results, fullscale_root, feat_group):
    """Append new sections to the EXISTING P7_CoarseToFine_report.md (never
    truncated/overwritten -- the original Stage A/B sections stay intact)."""
    report_path = p7base.REPORT_PATH
    existing = ""
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            existing = f.read().rstrip("\n") + "\n"

    lines = []
    lines.append("\n---\n")
    lines.append("## P7 Coarse Sub-model Improvement (Varian A/B)")
    lines.append("")
    lines.append(
        "Two isolated variants trained ONLY for the `coarse` sub-model (`fine_A`/`fine_I`/`fine_E`/"
        "`sa_branch` untouched, reused as-is from the results above). Same shared "
        f"`three_way_split(seed=42)` and feature group (`{feat_group}`) as the original coarse "
        "baseline -- only the classifier (Varian A) or the training data (Varian B) changes."
    )
    lines.append("")
    lines.append(
        "- **Varian A (`coarse_e0_balanced/`):** `WeightedClassicalClassifier` "
        "(`classical_models_ext.py`) -- identical SVM/split/feature group, `class_weight='balanced'` "
        "added to the SVC to counter unbalanced vowel-group class sizes (A=3, I=3, E=2, O=1 syllables)."
    )
    lines.append(
        "- **Varian B (`coarse_e5_augmented/`):** plain `ClassicalClassifier`, coarse training data "
        "enriched via `SignalProcessor.apply_augmentation()` with "
        "`EXPERIMENT_RECIPES[\"E5_Data_Augmentation\"]`'s parameters "
        "(`add_noise=True, noise_factor=0.05, apply_jitter=True, jitter_ms=10`), applied AFTER the "
        "split, training data only -- val/test are the untouched baseline epochs. Note: "
        "`apply_augmentation`'s noise/jitter are unseeded (same as the existing E5 precedent in "
        "`run_subject_dependent.py`), so Varian B's exact numbers may vary marginally between runs."
    )
    lines.append("")
    lines.append("### Per-subject coarse test accuracy: baseline vs. Varian A vs. Varian B")
    lines.append("")
    lines.append("| Subject | Baseline E0 (%) | Varian A: balanced (%) | Varian B: E5-augmented (%) |")
    lines.append("|---|---|---|---|")
    for subj in selection["subjects"]:
        r = per_subject[subj]
        lines.append(f"| {subj} | {r['baseline_e0_test_accuracy']*100:.2f} | "
                      f"{r['variant_a_e0_balanced']['test_accuracy']*100:.2f} | "
                      f"{r['variant_b_e5_augmented']['test_accuracy']*100:.2f} |")
    lines.append("")
    m, s = selection["means_pct"], selection["stds_pct"]
    lines.append(f"**Mean coarse test accuracy (n={selection['n_subjects']}):** "
                 f"Baseline E0 {m['baseline_e0']:.2f}% (std {s['baseline_e0']:.2f}pp) | "
                 f"Varian A balanced {m['variant_a_e0_balanced']:.2f}% (std {s['variant_a_e0_balanced']:.2f}pp) | "
                 f"Varian B E5-augmented {m['variant_b_e5_augmented']:.2f}% (std {s['variant_b_e5_augmented']:.2f}pp)")
    lines.append("")
    lines.append(f"**Automatic selection (coarse sub-model, improved):** `{selection['winner']}` -- {selection['reason']}")
    lines.append("")

    lines.append("### End-to-end Metrics (Improved Coarse Stage)")
    lines.append("")
    lines.append(f"Coarse sub-model used for these figures: **`{selection['winner']}`** (auto-selected "
                  "above). Shown side-by-side with the original (baseline E0, unweighted) end-to-end "
                  "figures from the section above, for direct before/after comparison.")
    lines.append("")
    lines.append("| Subject | First-syllable e2e -- before (%) | First-syllable e2e -- after (%) | "
                  "Full-word e2e -- before (%) | Full-word e2e -- after (%) |")
    lines.append("|---|---|---|---|---|")
    before_first, after_first, before_word, after_word = [], [], [], []
    for subj in selection["subjects"]:
        with open(os.path.join(fullscale_root, f"results_{subj}.json")) as f:
            original = json.load(f)
        before_fs = original["first_syllable_e2e"]["accuracy"] * 100
        after_fs = e2e_results[subj]["first_syllable_e2e"]["accuracy"] * 100
        before_fw_avail = original["full_word_e2e"].get("available")
        after_fw_avail = e2e_results[subj]["full_word_e2e"].get("available")
        before_fw = f"{original['full_word_e2e']['accuracy']*100:.2f}" if before_fw_avail else "n/a"
        after_fw = f"{e2e_results[subj]['full_word_e2e']['accuracy']*100:.2f}" if after_fw_avail else "n/a"
        lines.append(f"| {subj} | {before_fs:.2f} | {after_fs:.2f} | {before_fw} | {after_fw} |")
        before_first.append(before_fs)
        after_first.append(after_fs)
        if before_fw_avail:
            before_word.append(original["full_word_e2e"]["accuracy"] * 100)
        if after_fw_avail:
            after_word.append(e2e_results[subj]["full_word_e2e"]["accuracy"] * 100)
    lines.append("")
    lines.append(f"**Mean first-syllable e2e:** before {np.mean(before_first):.4f}% -> "
                 f"after {np.mean(after_first):.4f}% (delta {np.mean(after_first) - np.mean(before_first):+.4f}pp, n={len(before_first)})")
    if before_word and after_word:
        lines.append(f"**Mean full-word e2e:** before {np.mean(before_word):.4f}% -> "
                     f"after {np.mean(after_word):.4f}% (delta {np.mean(after_word) - np.mean(before_word):+.4f}pp, n={len(after_word)})")
    if selection["winner"] == "baseline_e0":
        lines.append("")
        lines.append("_Baseline E0 won the automatic selection -- the coarse sub-model is unchanged, so "
                     "the 'before'/'after' end-to-end figures above are identical by construction._")

    new_content = "\n".join(lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(existing + new_content)
    print(f"[INFO][P7-Improve] Appended improvement sections to {report_path}")


def run_coarse_improvement(subject_ids=None):
    print(f"\n{'=' * 70}\n P7 Coarse Sub-model Improvement -- Varian A (balanced) vs. Varian B (E5-augmented)\n{'=' * 70}")

    fullscale_root = p7base.setup_experiment(p7base.FULLSCALE_STAGE_DIR, pilar=p7base.PILAR)["weights"]
    variant_a_dir = os.path.join(fullscale_root, VARIANT_A_DIR)
    variant_b_dir = os.path.join(fullscale_root, VARIANT_B_DIR)

    selection_path = os.path.join(fullscale_root, "feature_selection_decision.json")
    if not os.path.exists(selection_path):
        raise RuntimeError(
            f"[P7-Improve] Coarse feature-group selection not found at {selection_path}. "
            f"Run `run_p7_coarse_to_fine.py` (Stage A+B) first to establish the baseline this "
            f"script improves on."
        )
    with open(selection_path) as f:
        feat_group = json.load(f)["winner"]
    print(f"[INFO][P7-Improve] Using coarse feature group from Stage A selection: {feat_group}")

    if subject_ids is None:
        subject_ids = p7base.discover_subject_ids()

    extractor = EEGFeatureExtractor(fs=p7base.E0_PROCESSOR_PARAMS["target_fs"])

    per_subject = {}
    e2e_inputs = {}

    for subject_id in subject_ids:
        baseline_json = os.path.join(fullscale_root, f"results_{subject_id}.json")
        if not os.path.exists(baseline_json):
            print(f"[WARNING][P7-Improve] No existing P7 baseline for {subject_id} -- skipping "
                  f"(run `run_p7_coarse_to_fine.py --stage b` first).")
            continue
        with open(baseline_json) as f:
            baseline_result = json.load(f)
        baseline_test_acc = baseline_result["sub_models"]["coarse"]["test_accuracy"]

        print(f"\n[INFO][P7-Improve] {subject_id}: rebuilding shared E0 split "
              f"(same DatasetBuilder call + seed=42 as the original Stage B run)...")
        X_3d, y = p7base.build_standard_e0_split_raw(subject_id)
        if X_3d is None:
            print(f"[WARNING][P7-Improve] No epochs extracted for {subject_id}; skipping.")
            continue

        X_train_3d, X_val_3d, X_test_3d, y_train, y_val, y_test = three_way_split(
            X_3d, y, random_state=p7base.SPLIT_RANDOM_STATE
        )
        e2e_inputs[subject_id] = (X_test_3d, y_test)

        comparison_json = os.path.join(fullscale_root, f"coarse_variant_comparison_{subject_id}.json")
        if os.path.exists(comparison_json):
            print(f"[SKIP][P7-Improve] {subject_id} variant comparison already exists.")
            with open(comparison_json) as f:
                per_subject[subject_id] = json.load(f)
            continue

        Xtr_3d, ytr_raw = filter_split_by_labels(X_train_3d, y_train, SUBMODEL_LABEL_SETS["coarse"])
        Xva_3d, yva_raw = filter_split_by_labels(X_val_3d, y_val, SUBMODEL_LABEL_SETS["coarse"])
        Xte_3d, yte_raw = filter_split_by_labels(X_test_3d, y_test, SUBMODEL_LABEL_SETS["coarse"])
        ytr = map_labels_to_vowel_group_ids(ytr_raw)
        yva = map_labels_to_vowel_group_ids(yva_raw)
        yte = map_labels_to_vowel_group_ids(yte_raw)

        result_a = train_variant_a(Xtr_3d, ytr, Xva_3d, yva, Xte_3d, yte, extractor,
                                    variant_a_dir, subject_id, feat_group)
        result_b = train_variant_b(Xtr_3d, ytr, Xva_3d, yva, Xte_3d, yte, extractor,
                                    variant_b_dir, subject_id, feat_group)

        print(f"[INFO][P7-Improve] {subject_id}: baseline {baseline_test_acc*100:.2f}% | "
              f"Varian A (balanced) {result_a['test_accuracy']*100:.2f}% | "
              f"Varian B (E5-aug) {result_b['test_accuracy']*100:.2f}%")

        combined = {
            "subject_id": subject_id, "feature_group": feat_group,
            "baseline_e0_test_accuracy": float(baseline_test_acc),
            "variant_a_e0_balanced": result_a,
            "variant_b_e5_augmented": result_b,
        }
        per_subject[subject_id] = combined
        with open(comparison_json, "w") as f:
            json.dump(combined, f, indent=2)

    if not per_subject:
        raise RuntimeError("[P7-Improve] No subjects processed -- nothing to select/report.")

    selection = select_winning_coarse_variant(per_subject)
    print(f"\n[INFO][P7-Improve] Winning coarse variant: {selection['winner']} -- {selection['reason']}")

    e2e_results = recompute_end_to_end_metrics(per_subject, selection, fullscale_root, feat_group, e2e_inputs)

    write_improvement_report(per_subject, selection, e2e_results, fullscale_root, feat_group)
    return per_subject, selection, e2e_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P7 coarse sub-model improvement: Varian A (balanced) + Varian B (E5 "
                     "augmentation), auto-select winner, recompute end-to-end metrics."
    )
    parser.add_argument("--subjects", nargs="+", default=None,
                         help="Restrict to specific subject IDs (e.g. --subjects S1 S2). "
                              "Default: all 12 subjects, auto-discovered.")
    args = parser.parse_args()
    run_coarse_improvement(subject_ids=args.subjects)

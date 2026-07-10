"""
backend/src/experiments_p4_p7/run_p7_coarse_combined.py

P7 coarse sub-model ablation, Fase 2: automatic combination of whichever
Fase 1 factors (run_p7_coarse_ablation.py) proved individually helpful,
into ONE final coarse model per subject.

Automatic inclusion rule: a Fase 1 variant is included in the combined
model if (mean 12-subject test accuracy) - (mean baseline test accuracy)
> 1.0 percentage point (INCLUSION_THRESHOLD_PP, same constant as
run_p7_coarse_ablation.py's preview). Varian E (calibration) is the one
exception: it is ALWAYS included structurally (wrapped at the outer layer)
regardless of whether it clears the threshold, because its purpose is
supporting Fase 3's confidence-gated post-processing
(run_p7_postprocessing.py), not raising raw accuracy by itself -- applying
an accuracy-only inclusion rule to a calibration-quality feature would be
a category error, not a stricter standard. This exception is recorded
explicitly in the report, not hidden.

The combined model composes whichever of A (class_weight='balanced'), B
(E5 augmentation), C (per-subject tuned C), D (feature group 'all') passed
the threshold -- e.g. if only A and C passed: class_weight='balanced' +
that subject's own Fase-1-tuned C + Barlow features (D not included, so
the default feature group) + calibration wrapper (always). If NONE of
A-D pass, the combined model reduces to the plain baseline wrapped only in
calibration.

fine_A/fine_I/fine_E/sa_branch are still untouched -- this script only
ever (re)trains the `coarse` sub-model. Reuses p7_coarse_cache.py's cached
raw split + Barlow features exactly like Fase 1, so no raw CSV is re-read
here either (fresh feature extraction is only needed when Varian D and/or
B's augmentation are part of the included combination).

Also defines SoftPredictBundle, a small generic loaded-model+scaler bundle
exposing predict_proba (not just a hard label) for single-epoch inference
-- used both for this script's own combined coarse model AND (read-only,
in run_p7_postprocessing.py) for the existing fine_A/fine_I/fine_E
sub-models, neither of which run_p7_coarse_to_fine.py's own
SubModelBundle supports (it only exposes a hard predict_single()).
run_p7_coarse_to_fine.py itself is never modified.

Usage:
    cd backend/src/experiments_p4_p7
    python run_p7_coarse_combined.py                 # all 12 subjects
    python run_p7_coarse_combined.py --subjects S1 S2
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np

from scipy.stats import wilcoxon
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.extract_eeg_features import EEGFeatureExtractor
from utils.data_utils import fit_and_apply_scaler
from preprocessing.signal_processor import SignalProcessor

from experiments_p4_p7 import run_p7_coarse_to_fine as p7base
from experiments_p4_p7 import run_p7_coarse_ablation as ablation
from experiments_p4_p7.p7_coarse_cache import get_or_build_cached_coarse

PILAR = ablation.PILAR
FULLSCALE_STAGE_DIR = ablation.FULLSCALE_STAGE_DIR
REPORTS_DIR = ablation.REPORTS_DIR
REPORT_PATH = ablation.PHASE1_REPORT_PATH  # Fase 2 appends to the same running ablation-study doc
COMBINED_DIR_NAME = "coarse_final_combined"


class SoftPredictBundle:
    """Generic loaded-model+scaler bundle exposing both hard (predict) and
    soft (predict_proba) single-epoch inference. Works for anything
    pickled as a plain sklearn Pipeline OR a fitted CalibratedClassifierCV
    -- both expose predict()/predict_proba()/classes_ identically."""

    def __init__(self, model_path, scaler_path, feat_group, fs=256):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.feat_group = feat_group
        self.extractor = EEGFeatureExtractor(fs=fs)

    def _features_single(self, epoch_2d):
        """epoch_2d: (channels, time)."""
        X_3d = epoch_2d[np.newaxis, :, :]
        groups = None if self.feat_group == "all" else [self.feat_group]
        features = self.extractor.transform(X_3d, groups=groups)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(features)

    def predict_single(self, epoch_2d):
        return int(self.model.predict(self._features_single(epoch_2d))[0])

    def predict_proba_single(self, epoch_2d):
        """Returns (proba_vector, classes_) for one epoch."""
        proba = self.model.predict_proba(self._features_single(epoch_2d))[0]
        return proba, self.model.classes_


def select_included_factors(phase1_summary):
    """Applies the >1pp inclusion rule to A/B/C/D; E is always True
    (structural inclusion, see module docstring)."""
    deltas = phase1_summary["deltas_pp_vs_baseline"]
    included = {label: bool(deltas[label] > ablation.INCLUSION_THRESHOLD_PP) for label in ablation.VARIANT_ORDER}
    included["E_calibrated"] = True  # always structural, independent of its own delta
    return included


def train_combined_model(subject_id, cached, included, chosen_c_per_subject, extractor, proc, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    feat_group = "all" if included["D_feat_all"] else "barlow"
    class_weight = "balanced" if included["A_balanced"] else None
    C_value = chosen_c_per_subject.get(subject_id, ablation.DEFAULT_C) if included["C_tuned"] else ablation.DEFAULT_C
    use_augment = included["B_augmented"]

    X_train_3d, ytr = cached["X_train_3d_coarse"], cached["y_train_coarse"]
    X_val_3d, yva = cached["X_val_3d_coarse"], cached["y_val_coarse"]
    X_test_3d, yte = cached["X_test_3d_coarse"], cached["y_test_coarse"]

    if use_augment:
        X_aug_3d = ablation.augment_training_epochs(X_train_3d, proc, ablation.E5_AUGMENTATION_PARAMS)
        X_train_3d_final = np.concatenate([X_train_3d, X_aug_3d], axis=0)
        y_train_final = np.concatenate([ytr, ytr], axis=0)
        shuffle_idx = np.random.permutation(len(y_train_final))
        X_train_3d_final = X_train_3d_final[shuffle_idx]
        y_train_final = y_train_final[shuffle_idx]
    else:
        X_train_3d_final, y_train_final = X_train_3d, ytr

    groups = [feat_group]
    # Cache only covers unaugmented Barlow features -- reuse it directly
    # only when neither augmentation nor an alternate feature group
    # changed the training data; otherwise fresh extraction is unavoidable.
    if feat_group == "barlow" and not use_augment:
        Xtr_feat = cached["X_train_feat_barlow"]
    else:
        Xtr_feat = extractor.transform(X_train_3d_final, groups=groups)
        Xtr_feat = np.nan_to_num(Xtr_feat, nan=0.0, posinf=0.0, neginf=0.0)

    if feat_group == "barlow":
        Xva_feat, Xte_feat = cached["X_val_feat_barlow"], cached["X_test_feat_barlow"]
    else:
        Xva_feat = np.nan_to_num(extractor.transform(X_val_3d, groups=groups), nan=0.0, posinf=0.0, neginf=0.0)
        Xte_feat = np.nan_to_num(extractor.transform(X_test_3d, groups=groups), nan=0.0, posinf=0.0, neginf=0.0)

    scaler_path = os.path.join(output_dir, f"scaler_P7_coarse_final_combined_{feat_group}_{subject_id}.pkl")
    Xtr_s, Xva_s, Xte_s, _ = fit_and_apply_scaler(Xtr_feat, Xva_feat, Xte_feat, save_path=scaler_path)

    base_pipeline = ablation.baseline_svc_pipeline(C=C_value, class_weight=class_weight)
    calibrated = CalibratedClassifierCV(base_pipeline, cv=5, method='sigmoid')
    calibrated.fit(Xtr_s, y_train_final)

    val_acc = float(accuracy_score(yva, calibrated.predict(Xva_s)))
    test_pred = calibrated.predict(Xte_s)
    test_proba = calibrated.predict_proba(Xte_s)
    test_acc = float(accuracy_score(yte, test_pred))
    test_brier = ablation.multiclass_brier_score(yte, test_proba, calibrated.classes_)
    test_ece = ablation.expected_calibration_error(yte, test_proba, calibrated.classes_)

    np.save(os.path.join(output_dir, f"Xtest_P7_coarse_final_combined_{feat_group}_{subject_id}.npy"), Xte_s)
    np.save(os.path.join(output_dir, f"ytest_P7_coarse_final_combined_{feat_group}_{subject_id}.npy"), yte)

    model_path = os.path.join(output_dir, f"CalibratedSVM_P7_coarse_final_combined_{feat_group}_{subject_id}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(calibrated, f)

    return {
        "subject_id": subject_id, "feature_group": feat_group, "C": C_value,
        "class_weight": class_weight, "augmented": use_augment,
        "val_accuracy": val_acc, "test_accuracy": test_acc,
        "test_brier_score": test_brier, "test_ece": test_ece,
        "classes_covered": ablation.classes_covered_from_predictions(yte, test_pred),
        "model_path": model_path, "scaler_path": scaler_path,
    }


def summarize_phase2(per_subject_combined, phase1_summary):
    subjects = sorted(per_subject_combined.keys())
    baseline_accs = [phase1_summary["per_candidate_pct"]["baseline"][phase1_summary["subjects"].index(s)]
                      for s in subjects]
    combined_accs = [per_subject_combined[s]["test_accuracy"] * 100.0 for s in subjects]

    mean_baseline = float(np.mean(baseline_accs))
    mean_combined = float(np.mean(combined_accs))
    delta_pp = mean_combined - mean_baseline

    if len(subjects) >= ablation.MIN_PAIRS_FOR_WILCOXON:
        diffs = np.array(combined_accs) - np.array(baseline_accs)
        p_value = None if np.all(diffs == 0) else float(wilcoxon(combined_accs, baseline_accs)[1])
    else:
        p_value = None

    return {
        "n_subjects": len(subjects), "subjects": subjects,
        "mean_baseline_pct": mean_baseline, "mean_combined_pct": mean_combined,
        "delta_pp": delta_pp, "wilcoxon_p_vs_baseline": p_value,
        "per_subject_combined_pct": dict(zip(subjects, combined_accs)),
        "per_subject_baseline_pct": dict(zip(subjects, baseline_accs)),
    }


def append_phase2_report(included, phase1_summary, per_subject_combined, phase2_summary):
    lines = []
    lines.append("\n---\n")
    lines.append("## Fase 2 -- Kombinasi Otomatis")
    lines.append("")
    lines.append(f"Aturan inklusi: sebuah Varian Fase 1 diikutsertakan jika "
                 f"(rerata akurasi 12 subjek) - (rerata baseline) > {ablation.INCLUSION_THRESHOLD_PP:.1f} pp.")
    lines.append("")
    lines.append("| Varian | Delta vs baseline (pp) | Diikutsertakan? | Alasan |")
    lines.append("|---|---|---|---|")
    for label in ablation.VARIANT_ORDER:
        delta = phase1_summary["deltas_pp_vs_baseline"][label]
        is_in = included[label]
        if label == "E_calibrated":
            reason = "SELALU disertakan secara struktural (kalibrasi, bukan untuk akurasi mentah) -- lihat catatan di bawah"
        else:
            reason = f"delta {delta:+.2f}pp {'> ' if is_in else '<= '}{ablation.INCLUSION_THRESHOLD_PP:.1f}pp"
        lines.append(f"| {label} | {delta:+.2f} | {'YA' if is_in else 'tidak'} | {reason} |")
    lines.append("")
    lines.append(
        "**Catatan Varian E:** disertakan di lapisan terluar model final terlepas dari lolos/tidaknya "
        "ambang 1pp di atas, karena fungsinya mendukung Fase 3 (confidence gating), bukan menaikkan "
        "akurasi mentah -- ini BUKAN pelanggaran aturan inklusi, melainkan aturan yang berbeda "
        "untuk tujuan yang berbeda, dijelaskan eksplisit di sini supaya tidak disalahpahami."
    )
    lines.append("")

    included_desc = []
    if included["A_balanced"]:
        included_desc.append("class_weight='balanced'")
    if included["B_augmented"]:
        included_desc.append("E5 augmentation (training data)")
    if included["C_tuned"]:
        included_desc.append("per-subject tuned C")
    if included["D_feat_all"]:
        included_desc.append("feature group 'all' (ganti dari Barlow default)")
    included_desc.append("dibungkus CalibratedClassifierCV (selalu)")
    lines.append(f"**Komposisi model kombinasi final:** {' + '.join(included_desc)}.")
    lines.append("")

    lines.append("### Tabel akhir: baseline -> tiap varian individual -> kombinasi final")
    lines.append("")
    lines.append("| Kandidat | Mean akurasi (%) | Delta vs baseline (pp) | Wilcoxon p |")
    lines.append("|---|---|---|---|")
    lines.append(f"| baseline | {phase1_summary['means_pct']['baseline']:.2f} | +0.00 | -- |")
    for label in ablation.VARIANT_ORDER:
        p = phase1_summary["wilcoxon_p_vs_baseline"][label]
        p_str = f"{p:.4f}" if p is not None else "n/a"
        lines.append(f"| {label} | {phase1_summary['means_pct'][label]:.2f} | "
                     f"{phase1_summary['deltas_pp_vs_baseline'][label]:+.2f} | {p_str} |")
    p2_str = f"{phase2_summary['wilcoxon_p_vs_baseline']:.4f}" if phase2_summary['wilcoxon_p_vs_baseline'] is not None else "n/a"
    lines.append(f"| **kombinasi final** | **{phase2_summary['mean_combined_pct']:.2f}** | "
                 f"**{phase2_summary['delta_pp']:+.2f}** | **{p2_str}** |")
    lines.append("")

    lines.append("### Per-subject: kombinasi final")
    lines.append("")
    lines.append("| Subject | Baseline (%) | Kombinasi Final (%) | Delta (pp) | Feature | C | class_weight | Augmented |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for subj in phase2_summary["subjects"]:
        r = per_subject_combined[subj]
        base_acc = phase2_summary["per_subject_baseline_pct"][subj]
        comb_acc = phase2_summary["per_subject_combined_pct"][subj]
        lines.append(f"| {subj} | {base_acc:.2f} | {comb_acc:.2f} | {comb_acc - base_acc:+.2f} | "
                     f"{r['feature_group']} | {r['C']} | {r['class_weight'] or '-'} | "
                     f"{'yes' if r['augmented'] else 'no'} |")
    lines.append("")

    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO][P7-Combined] Fase 2 section appended to {REPORT_PATH}")


def run_combined_phase2(subject_ids=None):
    print(f"\n{'=' * 70}\n P7 Coarse Sub-model Ablation -- Fase 2 (Kombinasi Otomatis)\n{'=' * 70}")

    fullscale_root = p7base.setup_experiment(FULLSCALE_STAGE_DIR, pilar=PILAR)["weights"]
    summary_path = ablation.phase1_summary_path(fullscale_root)
    if not os.path.exists(summary_path):
        raise RuntimeError(
            f"[P7-Combined] Fase 1 summary not found at {summary_path}. Run "
            f"`run_p7_coarse_ablation.py` first to establish which factors help."
        )
    with open(summary_path) as f:
        phase1_summary = json.load(f)

    included = select_included_factors(phase1_summary)
    print(f"[INFO][P7-Combined] Included factors: {included}")

    output_dir = os.path.join(fullscale_root, COMBINED_DIR_NAME)
    chosen_c_per_subject = phase1_summary["chosen_c_per_subject"]

    if subject_ids is None:
        subject_ids = phase1_summary["subjects"]

    extractor = EEGFeatureExtractor(fs=p7base.E0_PROCESSOR_PARAMS["target_fs"])
    proc = SignalProcessor(target_fs=p7base.E0_PROCESSOR_PARAMS["target_fs"])

    per_subject_combined = {}
    for subject_id in subject_ids:
        result_json = os.path.join(output_dir, f"results_{subject_id}.json")
        if os.path.exists(result_json):
            print(f"[SKIP][P7-Combined] {subject_id} combined model already exists.")
            with open(result_json) as f:
                per_subject_combined[subject_id] = json.load(f)
            continue

        print(f"[INFO][P7-Combined] {subject_id}: loading/building coarse cache...")
        cached = get_or_build_cached_coarse(subject_id)
        if cached is None:
            print(f"[WARNING][P7-Combined] No raw epochs available for {subject_id}; skipping.")
            continue

        result = train_combined_model(subject_id, cached, included, chosen_c_per_subject,
                                       extractor, proc, output_dir)
        per_subject_combined[subject_id] = result
        os.makedirs(output_dir, exist_ok=True)
        with open(result_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO][P7-Combined] {subject_id}: combined test acc {result['test_accuracy']*100:.2f}% "
              f"(feature={result['feature_group']}, C={result['C']}, "
              f"class_weight={result['class_weight']}, augmented={result['augmented']})")

    if not per_subject_combined:
        raise RuntimeError("[P7-Combined] No subjects processed -- nothing to summarize/report.")

    phase2_summary = summarize_phase2(per_subject_combined, phase1_summary)
    with open(os.path.join(fullscale_root, "phase2_summary.json"), "w") as f:
        json.dump({"included_factors": included, **phase2_summary}, f, indent=2)

    append_phase2_report(included, phase1_summary, per_subject_combined, phase2_summary)
    return per_subject_combined, phase2_summary, included


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P7 coarse sub-model ablation Fase 2: combine whichever Fase 1 factors passed "
                     "the >1pp threshold into one final coarse model per subject (E always included)."
    )
    parser.add_argument("--subjects", nargs="+", default=None,
                         help="Restrict to specific subject IDs (e.g. --subjects S1 S2). "
                              "Default: all subjects present in the Fase 1 summary.")
    args = parser.parse_args()
    run_combined_phase2(subject_ids=args.subjects)

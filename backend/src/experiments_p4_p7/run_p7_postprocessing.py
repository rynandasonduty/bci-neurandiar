"""
backend/src/experiments_p4_p7/run_p7_postprocessing.py

P7 coarse sub-model ablation, Fase 3: post-processing strategies on top of
the Fase 2 combined+calibrated coarse model (run_p7_coarse_combined.py),
WITHOUT retraining anything. The Fase 2 combined model, Varian A, Varian C
(run_p7_coarse_ablation.py), and fine_A/fine_I/fine_E (existing,
run_p7_coarse_to_fine.py's Stage B) are all loaded read-only via
SoftPredictBundle (run_p7_coarse_combined.py).

Strategi 1 -- Confidence-gated Coarse->Fine:
    For each epoch, if the combined coarse model's top predicted
    probability is below a threshold, compare the joint probability
    P(coarse_group) x P(top fine syllable | group) for the TOP-2 candidate
    vowel groups instead of committing blindly to the coarse top-1 choice.
    Group O (single-member, deterministic "BO") has no fine model, so its
    "fine" probability is always 1.0 by construction. Threshold in
    {0.4, 0.5, 0.6, 0.7, 0.8} is tuned per subject on the VALIDATION split
    (never test), then evaluated ONCE on the test split with the winning
    threshold, keeping test genuinely held out.

Strategi 2 -- Ensemble Voting:
    Majority vote among three independently-trained coarse predictions per
    test epoch -- Varian A (balanced), Varian C (tuned), and the Fase 2
    combined model -- each resolved to a syllable via the same fine
    models. Ties are broken by whichever tied syllable has the higher mean
    coarse-group probability across its voters.

Strategi 3 -- Final Selection:
    All three strategies -- (1) combined model alone (plain hierarchical,
    no gating), (2) confidence-gated, (3) ensemble voting -- are compared
    at the END-TO-END first-syllable accuracy level (not raw coarse
    accuracy), per subject, then averaged across subjects. The GLOBAL
    winner (highest mean across all subjects) is selected automatically
    and is what run_p7_final_integration.py uses -- this is a single
    pipeline-wide decision, not a per-subject one, consistent with how
    Fase 2's factor-inclusion decision is also global.

All predictions here operate on cached raw epochs
(p7_coarse_cache.py's X_val_3d_coarse / X_test_3d_coarse, including the
raw 9-class syllable labels y_{split}_coarse_syllable needed to score
first-syllable identity, not just vowel group) -- no raw CSV is read in
this script.

Usage:
    cd backend/src/experiments_p4_p7
    python run_p7_postprocessing.py                 # all subjects with Fase 2 results
    python run_p7_postprocessing.py --subjects S1 S2
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments_p4_p7 import run_p7_coarse_to_fine as p7base
from experiments_p4_p7 import run_p7_coarse_ablation as ablation
from experiments_p4_p7 import run_p7_coarse_combined as combined
from experiments_p4_p7.dataset_builders_ext import ID_TO_VOWEL_GROUP
from experiments_p4_p7.p7_coarse_cache import get_or_build_cached_coarse

PILAR = ablation.PILAR
FULLSCALE_STAGE_DIR = ablation.FULLSCALE_STAGE_DIR
REPORTS_DIR = ablation.REPORTS_DIR
REPORT_PATH = ablation.PHASE1_REPORT_PATH  # Fase 3 appends to the same running ablation-study doc

BO_LABEL = 10  # dataset_builders_ext.LABEL_TO_SYLLABLE[10] == "BO" -- group O's single, deterministic syllable
THRESHOLD_GRID = [0.4, 0.5, 0.6, 0.7, 0.8]
STRATEGY_NAMES = ["combined_only", "confidence_gated", "ensemble_voting"]


def _fine_bundle_paths(fullscale_root, name, subject_id):
    d = os.path.join(fullscale_root, name)
    model_path = os.path.join(d, f"SVM_P7_{name}_barlow_{subject_id}.pkl")
    scaler_path = os.path.join(d, f"scaler_P7_{name}_barlow_{subject_id}.pkl")
    return model_path, scaler_path


def load_fine_bundles(fullscale_root, subject_id):
    """Read-only load of the EXISTING fine_A/fine_I/fine_E sub-models
    (never retrained anywhere in this ablation pipeline)."""
    bundles = {}
    for letter, name in (("A", "fine_A"), ("I", "fine_I"), ("E", "fine_E")):
        model_path, scaler_path = _fine_bundle_paths(fullscale_root, name, subject_id)
        bundles[letter] = combined.SoftPredictBundle(
            model_path, scaler_path, "barlow", fs=p7base.E0_PROCESSOR_PARAMS["target_fs"]
        )
    return bundles


def load_combined_coarse_bundle(fullscale_root, feat_group, subject_id):
    d = os.path.join(fullscale_root, combined.COMBINED_DIR_NAME)
    model_path = os.path.join(d, f"CalibratedSVM_P7_coarse_final_combined_{feat_group}_{subject_id}.pkl")
    scaler_path = os.path.join(d, f"scaler_P7_coarse_final_combined_{feat_group}_{subject_id}.pkl")
    return combined.SoftPredictBundle(model_path, scaler_path, feat_group, fs=p7base.E0_PROCESSOR_PARAMS["target_fs"])


def load_variant_bundle(fullscale_root, variant_dir_name, tag, feat_group, subject_id):
    d = os.path.join(fullscale_root, variant_dir_name)
    model_path = os.path.join(d, f"SVM_P7_coarse_{tag}_{feat_group}_{subject_id}.pkl")
    scaler_path = os.path.join(d, f"scaler_P7_coarse_{tag}_{feat_group}_{subject_id}.pkl")
    return combined.SoftPredictBundle(model_path, scaler_path, feat_group, fs=p7base.E0_PROCESSOR_PARAMS["target_fs"])


def best_syllable_and_prob_for_group(epoch_2d, group_id, fine_bundles):
    """(syllable_label, P(top syllable | group)) for one vowel group. O is
    deterministic -- single member, no fine-stage model by design."""
    group_letter = ID_TO_VOWEL_GROUP[group_id]
    if group_letter == "O":
        return BO_LABEL, 1.0
    proba, classes = fine_bundles[group_letter].predict_proba_single(epoch_2d)
    top_idx = int(np.argmax(proba))
    return int(classes[top_idx]), float(proba[top_idx])


def resolve_syllable_for_group(epoch_2d, group_id, fine_bundles):
    """Hard first-syllable prediction for one vowel group (top fine
    prediction, or BO for group O)."""
    group_letter = ID_TO_VOWEL_GROUP[group_id]
    if group_letter == "O":
        return BO_LABEL
    return fine_bundles[group_letter].predict_single(epoch_2d)


def hierarchical_predict_single(epoch_2d, coarse_bundle, fine_bundles):
    """Plain (non-gated) coarse->fine prediction: always commit to the
    coarse model's top-1 group. This is Strategi (1)'s "combined model
    alone" reference point."""
    group_id = coarse_bundle.predict_single(epoch_2d)
    return resolve_syllable_for_group(epoch_2d, group_id, fine_bundles)


def confidence_gated_predict_single(epoch_2d, coarse_bundle, fine_bundles, threshold):
    proba, classes = coarse_bundle.predict_proba_single(epoch_2d)
    order = np.argsort(proba)[::-1]
    top_group_id = int(classes[order[0]])
    top_prob = float(proba[order[0]])

    if top_prob >= threshold or len(order) < 2:
        return resolve_syllable_for_group(epoch_2d, top_group_id, fine_bundles)

    candidates = []
    for rank in (0, 1):
        group_id = int(classes[order[rank]])
        group_prob = float(proba[order[rank]])
        syl_label, fine_prob = best_syllable_and_prob_for_group(epoch_2d, group_id, fine_bundles)
        candidates.append((syl_label, group_prob * fine_prob))

    return max(candidates, key=lambda c: c[1])[0]


def ensemble_vote_predict_single(epoch_2d, bundle_a, bundle_c, bundle_combined, fine_bundles):
    """Majority vote among 3 independently-trained coarse models' own
    hierarchical predictions; ties broken by mean voter probability for
    the tied syllable candidates."""
    votes, probs = [], []
    for bundle in (bundle_a, bundle_c, bundle_combined):
        group_id = bundle.predict_single(epoch_2d)
        syl_label = resolve_syllable_for_group(epoch_2d, group_id, fine_bundles)
        proba, _ = bundle.predict_proba_single(epoch_2d)
        votes.append(syl_label)
        probs.append(float(np.max(proba)))  # predict() is the argmax, so max(proba) is that vote's own confidence

    counts = {}
    for v, p in zip(votes, probs):
        counts.setdefault(v, []).append(p)
    max_count = max(len(v) for v in counts.values())
    tied = [label for label, plist in counts.items() if len(plist) == max_count]
    if len(tied) == 1:
        return tied[0]
    return max(tied, key=lambda label: np.mean(counts[label]))


def first_syllable_accuracy(X_3d, y_syllable, predict_fn):
    preds = [predict_fn(X_3d[i]) for i in range(len(X_3d))]
    n = len(y_syllable)
    correct = sum(1 for p, t in zip(preds, y_syllable) if p == int(t))
    return (correct / n if n else 0.0), preds


def tune_confidence_threshold(X_val_3d, y_val_syllable, coarse_bundle, fine_bundles):
    val_acc_per_threshold = {}
    for t in THRESHOLD_GRID:
        acc, _ = first_syllable_accuracy(
            X_val_3d, y_val_syllable,
            lambda epoch, t=t: confidence_gated_predict_single(epoch, coarse_bundle, fine_bundles, t),
        )
        val_acc_per_threshold[str(t)] = acc
    best_t = max(THRESHOLD_GRID, key=lambda t: val_acc_per_threshold[str(t)])
    return best_t, val_acc_per_threshold


def process_one_subject(fullscale_root, feat_group, subject_id, cached):
    coarse_bundle = load_combined_coarse_bundle(fullscale_root, feat_group, subject_id)
    fine_bundles = load_fine_bundles(fullscale_root, subject_id)

    X_val_3d, y_val_syl = cached["X_val_3d_coarse"], cached["y_val_coarse_syllable"]
    X_test_3d, y_test_syl = cached["X_test_3d_coarse"], cached["y_test_coarse_syllable"]

    # Strategi 1: confidence gating, threshold tuned on VAL, evaluated once on test.
    best_t, val_acc_per_t = tune_confidence_threshold(X_val_3d, y_val_syl, coarse_bundle, fine_bundles)
    test_acc_gated, _ = first_syllable_accuracy(
        X_test_3d, y_test_syl,
        lambda epoch: confidence_gated_predict_single(epoch, coarse_bundle, fine_bundles, best_t),
    )

    # Reference point: combined model alone, plain hierarchical, no gating.
    test_acc_combined_only, _ = first_syllable_accuracy(
        X_test_3d, y_test_syl,
        lambda epoch: hierarchical_predict_single(epoch, coarse_bundle, fine_bundles),
    )

    # Strategi 2: ensemble voting among Varian A, Varian C, and the combined model.
    bundle_a = load_variant_bundle(fullscale_root, ablation.VARIANT_DIRS["A_balanced"], "variant_a", "barlow", subject_id)
    bundle_c = load_variant_bundle(fullscale_root, ablation.VARIANT_DIRS["C_tuned"], "variant_c", "barlow", subject_id)
    test_acc_ensemble, _ = first_syllable_accuracy(
        X_test_3d, y_test_syl,
        lambda epoch: ensemble_vote_predict_single(epoch, bundle_a, bundle_c, coarse_bundle, fine_bundles),
    )

    return {
        "subject_id": subject_id,
        "confidence_gating": {
            "threshold_val_accuracy": val_acc_per_t,
            "chosen_threshold": best_t,
            "test_first_syllable_accuracy": test_acc_gated,
        },
        "strategy_comparison": {
            "combined_only": test_acc_combined_only,
            "confidence_gated": test_acc_gated,
            "ensemble_voting": test_acc_ensemble,
        },
    }


def summarize_phase3(per_subject_results):
    subjects = sorted(per_subject_results.keys())
    means, per_candidate = {}, {}
    for name in STRATEGY_NAMES:
        accs = [per_subject_results[s]["strategy_comparison"][name] * 100.0 for s in subjects]
        means[name] = float(np.mean(accs))
        per_candidate[name] = accs
    winner = max(STRATEGY_NAMES, key=lambda n: means[n])

    return {
        "n_subjects": len(subjects), "subjects": subjects,
        "means_pct": means, "per_candidate_pct": per_candidate,
        "winning_strategy": winner,
        "chosen_threshold_per_subject": {
            s: per_subject_results[s]["confidence_gating"]["chosen_threshold"] for s in subjects
        },
    }


def append_phase3_report(per_subject_results, phase3_summary):
    lines = []
    lines.append("\n---\n")
    lines.append("## Fase 3 -- Post-processing (Tanpa Training Ulang)")
    lines.append("")
    lines.append(
        "Seluruh strategi di bawah memakai model kombinasi final (Fase 2, dibungkus kalibrasi) dan "
        "model `fine_A`/`fine_I`/`fine_E` yang SUDAH ADA, tanpa melatih ulang apa pun."
    )
    lines.append("")
    lines.append("### Strategi 1 -- Confidence-gated Coarse->Fine")
    lines.append("")
    lines.append(
        "Threshold diuji pada VALIDATION set per subjek, dipilih otomatis berdasarkan akurasi "
        "validasi tertinggi, baru dievaluasi SEKALI ke test set (test set tetap murni held-out)."
    )
    lines.append("")
    lines.append("| Subject | Threshold Terpilih | " + " | ".join(f"val@{t}" for t in THRESHOLD_GRID) +
                 " | Test Acc (gated, %) |")
    lines.append("|---|---|" + "---|" * len(THRESHOLD_GRID) + "---|")
    for subj in phase3_summary["subjects"]:
        cg = per_subject_results[subj]["confidence_gating"]
        val_cells = " | ".join(f"{cg['threshold_val_accuracy'][str(t)]*100:.2f}" for t in THRESHOLD_GRID)
        lines.append(f"| {subj} | {cg['chosen_threshold']} | {val_cells} | "
                     f"{cg['test_first_syllable_accuracy']*100:.2f} |")
    lines.append("")

    lines.append("### Perbandingan 3 Strategi (akurasi end-to-end suku kata pertama, test set)")
    lines.append("")
    lines.append("| Subject | (1) Kombinasi Final Saja (%) | (2) Confidence-gated (%) | (3) Ensemble Voting (%) |")
    lines.append("|---|---|---|---|")
    for subj in phase3_summary["subjects"]:
        sc = per_subject_results[subj]["strategy_comparison"]
        lines.append(f"| {subj} | {sc['combined_only']*100:.2f} | {sc['confidence_gated']*100:.2f} | "
                     f"{sc['ensemble_voting']*100:.2f} |")
    lines.append("")
    m = phase3_summary["means_pct"]
    lines.append(f"**Mean (n={phase3_summary['n_subjects']}):** Kombinasi Final Saja {m['combined_only']:.2f}% | "
                 f"Confidence-gated {m['confidence_gated']:.2f}% | Ensemble Voting {m['ensemble_voting']:.2f}%")
    lines.append("")
    lines.append(f"**Strategi terpilih otomatis (Fase 4 memakai ini):** `{phase3_summary['winning_strategy']}` "
                 f"-- akurasi end-to-end suku kata pertama rerata tertinggi di antara ketiganya.")

    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO][P7-Postproc] Fase 3 section appended to {REPORT_PATH}")


def run_postprocessing_phase3(subject_ids=None):
    print(f"\n{'=' * 70}\n P7 Coarse Sub-model Ablation -- Fase 3 (Post-processing)\n{'=' * 70}")

    fullscale_root = p7base.setup_experiment(FULLSCALE_STAGE_DIR, pilar=PILAR)["weights"]
    phase2_summary_path = os.path.join(fullscale_root, "phase2_summary.json")
    if not os.path.exists(phase2_summary_path):
        raise RuntimeError(
            f"[P7-Postproc] Fase 2 summary not found at {phase2_summary_path}. Run "
            f"`run_p7_coarse_combined.py` first to build the combined coarse model."
        )
    with open(phase2_summary_path) as f:
        phase2_data = json.load(f)
    feat_group = "all" if phase2_data["included_factors"]["D_feat_all"] else "barlow"

    if subject_ids is None:
        subject_ids = phase2_data["subjects"]

    per_subject_results = {}
    for subject_id in subject_ids:
        result_json = os.path.join(fullscale_root, f"phase3_postprocessing_{subject_id}.json")
        if os.path.exists(result_json):
            print(f"[SKIP][P7-Postproc] {subject_id} Fase 3 result already exists.")
            with open(result_json) as f:
                per_subject_results[subject_id] = json.load(f)
            continue

        print(f"[INFO][P7-Postproc] {subject_id}: loading cache + models...")
        cached = get_or_build_cached_coarse(subject_id)
        if cached is None:
            print(f"[WARNING][P7-Postproc] No cached epochs available for {subject_id}; skipping.")
            continue

        result = process_one_subject(fullscale_root, feat_group, subject_id, cached)
        per_subject_results[subject_id] = result
        with open(result_json, "w") as f:
            json.dump(result, f, indent=2)

        sc = result["strategy_comparison"]
        print(f"[INFO][P7-Postproc] {subject_id}: combined-only {sc['combined_only']*100:.2f}% | "
              f"gated(t={result['confidence_gating']['chosen_threshold']}) {sc['confidence_gated']*100:.2f}% | "
              f"ensemble {sc['ensemble_voting']*100:.2f}%")

    if not per_subject_results:
        raise RuntimeError("[P7-Postproc] No subjects processed -- nothing to summarize/report.")

    phase3_summary = summarize_phase3(per_subject_results)
    with open(os.path.join(fullscale_root, "phase3_summary.json"), "w") as f:
        json.dump(phase3_summary, f, indent=2)

    append_phase3_report(per_subject_results, phase3_summary)
    return per_subject_results, phase3_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P7 coarse sub-model ablation Fase 3: confidence-gated coarse->fine, ensemble "
                     "voting, and automatic final-strategy selection -- no retraining."
    )
    parser.add_argument("--subjects", nargs="+", default=None,
                         help="Restrict to specific subject IDs (e.g. --subjects S1 S2). "
                              "Default: all subjects present in the Fase 2 summary.")
    args = parser.parse_args()
    run_postprocessing_phase3(subject_ids=args.subjects)

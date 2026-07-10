"""
backend/src/experiments_p4_p7/run_p7_final_integration.py

P7 coarse sub-model ablation, Fase 4 (final integration): recomputes BOTH
end-to-end metrics (first-syllable, full-word) using Fase 3's automatically
selected winning post-processing strategy (run_p7_postprocessing.py),
combined with the EXISTING fine_A/fine_I/fine_E/sa_branch sub-models --
never retrained anywhere in this ablation pipeline -- and appends a
consolidated "Ablation Study & Final Combined Model" section to
P7_CoarseToFine_report.md (the ORIGINAL Stage A/B report), covering Fase
1-3 plus this Fase 4, WITHOUT deleting the original E0-baseline content
already there.

Writes NEW per-subject files only (final_e2e_{subject_id}.json) alongside
-- never overwriting -- the original results_{subject_id}.json, so both
the original baseline e2e numbers and the new final numbers stay available
side by side for a notebook before/after comparison.

run_p7_coarse_to_fine.py's own compute_full_word_e2e_accuracy() hardcodes
plain top-1 coarse->fine prediction (predict_word_for_trial()), so it
cannot express confidence-gated or ensemble-voting predictions. Since that
file is never modified, this module duplicates its OfflineTrialReader-based
trial reconstruction (list_valid_trials / read_trial / 80-20 holdout,
identical methodology and random_state=42) but dispatches slot-1
prediction through predict_first_syllable_with_strategy() instead, so the
final full-word e2e figure reflects whichever Fase 3 strategy actually won.

Usage:
    cd backend/src/experiments_p4_p7
    python run_p7_final_integration.py                 # all subjects with Fase 3 results
    python run_p7_final_integration.py --subjects S1 S2
"""
import os
import sys
import json
import argparse
import numpy as np

from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments_p4_p7 import run_p7_coarse_to_fine as p7base
from experiments_p4_p7 import run_p7_coarse_ablation as ablation
from experiments_p4_p7 import run_p7_coarse_combined as combined
from experiments_p4_p7 import run_p7_postprocessing as postproc
from experiments_p4_p7.dataset_builders_ext import (
    LABEL_TO_SYLLABLE, DETERMINISTIC_FIRST_SYLLABLE_TO_WORD, SA_BRANCH_SECOND_SYLLABLE_TO_WORD,
)
from experiments_p4_p7.p7_coarse_cache import get_or_build_cached_coarse

PILAR = ablation.PILAR
FULLSCALE_STAGE_DIR = ablation.FULLSCALE_STAGE_DIR
ABLATION_REPORT_PATH = ablation.PHASE1_REPORT_PATH
MAIN_REPORT_PATH = p7base.REPORT_PATH  # P7_CoarseToFine_report.md -- the ORIGINAL Stage A/B report
MIN_PAIRS_FOR_WILCOXON = ablation.MIN_PAIRS_FOR_WILCOXON

# P3/E5_Data_Augmentation/S3/barlow champion, flat 19-class classification --
# read-only reference figure for the honesty-caveat comparison below, not
# recomputed here (already established/reported in Bab 6's own materials).
P3_CHAMPION_ACCURACY_PCT = 18.10


def predict_first_syllable_with_strategy(epoch_2d, strategy, coarse_bundle, fine_bundles,
                                          bundle_a, bundle_c, threshold):
    """Dispatches to the correct Fase 3 prediction function for the
    winning strategy name (as recorded in phase3_summary.json). Returns a
    raw first-syllable label int (e.g. 0=MA, 10=BO, 14=SA)."""
    if strategy == "combined_only":
        return postproc.hierarchical_predict_single(epoch_2d, coarse_bundle, fine_bundles)
    if strategy == "confidence_gated":
        return postproc.confidence_gated_predict_single(epoch_2d, coarse_bundle, fine_bundles, threshold)
    if strategy == "ensemble_voting":
        return postproc.ensemble_vote_predict_single(epoch_2d, bundle_a, bundle_c, coarse_bundle, fine_bundles)
    raise ValueError(f"Unknown winning strategy: {strategy!r}")


def compute_final_first_syllable_e2e(cached, strategy, coarse_bundle, fine_bundles,
                                      bundle_a, bundle_c, threshold):
    X_test_3d, y_test_syl = cached["X_test_3d_coarse"], cached["y_test_coarse_syllable"]
    acc, _ = postproc.first_syllable_accuracy(
        X_test_3d, y_test_syl,
        lambda epoch: predict_first_syllable_with_strategy(
            epoch, strategy, coarse_bundle, fine_bundles, bundle_a, bundle_c, threshold
        ),
    )
    return {"n_test_samples": int(len(y_test_syl)), "accuracy": float(acc), "strategy_used": strategy}


def predict_word_for_trial_with_strategy(epoch_slot1_2d, epoch_slot2_2d, strategy, coarse_bundle,
                                          fine_bundles, sa_branch_bundle, bundle_a, bundle_c, threshold):
    """Mirrors run_p7_coarse_to_fine.py's predict_word_for_trial() exactly,
    except slot-1 prediction goes through predict_first_syllable_with_
    strategy() so it reflects the winning Fase 3 strategy."""
    first_syl_label = predict_first_syllable_with_strategy(
        epoch_slot1_2d, strategy, coarse_bundle, fine_bundles, bundle_a, bundle_c, threshold
    )
    first_syl = LABEL_TO_SYLLABLE[first_syl_label]
    if first_syl in DETERMINISTIC_FIRST_SYLLABLE_TO_WORD:
        return DETERMINISTIC_FIRST_SYLLABLE_TO_WORD[first_syl]
    if first_syl == "SA":
        sa_label = sa_branch_bundle.predict_single(epoch_slot2_2d)
        second_syl = LABEL_TO_SYLLABLE[sa_label]
        return SA_BRANCH_SECOND_SYLLABLE_TO_WORD.get(second_syl, "UNKNOWN")
    return "UNKNOWN"  # not reachable given full 9-class coarse->fine coverage


def load_sa_bundle(fullscale_root, subject_id):
    d = os.path.join(fullscale_root, "sa_branch")
    model_path = os.path.join(d, f"SVM_P7_sa_branch_barlow_{subject_id}.pkl")
    scaler_path = os.path.join(d, f"scaler_P7_sa_branch_barlow_{subject_id}.pkl")
    return combined.SoftPredictBundle(model_path, scaler_path, "barlow",
                                       fs=p7base.E0_PROCESSOR_PARAMS["target_fs"])


def compute_final_full_word_e2e_accuracy(subject_id, strategy, coarse_bundle, fine_bundles,
                                          sa_branch_bundle, bundle_a, bundle_c, threshold):
    """Duplicates run_p7_coarse_to_fine.py's compute_full_word_e2e_accuracy()
    trial-reconstruction logic (same OfflineTrialReader usage, same 80/20
    trial-level holdout with random_state=42, same caveat about not being
    strictly leakage-free -- see that function's docstring for the full
    rationale) because that function hardcodes plain top-1 prediction and
    run_p7_coarse_to_fine.py is never modified. Only the prediction
    dispatch (predict_word_for_trial_with_strategy) differs."""
    reader = p7base.OfflineTrialReader(p7base.RAW_DATA_DIR, p7base.E0_PROCESSOR_PARAMS)
    try:
        trials_meta = reader.list_valid_trials(subject_id)
    except Exception as e:
        return {"available": False, "reason": f"could not list trials: {e}"}

    valid_trials = []
    n_skipped_word, n_skipped_artifact = 0, 0
    for idx, meta in enumerate(trials_meta):
        word = meta["word"].strip().upper()
        if word not in p7base.WORD_TO_SYLLABLES:
            n_skipped_word += 1
            continue
        try:
            trial = reader.read_trial(subject_id, trial_index=idx)
        except ValueError:
            n_skipped_artifact += 1
            continue
        valid_trials.append(trial)

    if len(valid_trials) < 10:
        return {"available": False, "reason": f"only {len(valid_trials)} valid trials reconstructed (need >= 10)"}

    words = [t["word"].strip().upper() for t in valid_trials]
    _, counts = np.unique(words, return_counts=True)
    can_stratify = len(set(words)) > 1 and min(counts) >= 2
    _, test_trials = train_test_split(
        valid_trials, test_size=0.2, random_state=p7base.SPLIT_RANDOM_STATE,
        stratify=words if can_stratify else None,
    )

    y_true_words, y_pred_words = [], []
    for trial in test_trials:
        y_true_words.append(trial["word"].strip().upper())
        y_pred_words.append(predict_word_for_trial_with_strategy(
            trial["epoch_slot1"], trial["epoch_slot2"], strategy, coarse_bundle, fine_bundles,
            sa_branch_bundle, bundle_a, bundle_c, threshold
        ))

    accuracy = accuracy_score(y_true_words, y_pred_words)
    return {
        "available": True, "strategy_used": strategy,
        "n_valid_trials_total": len(valid_trials),
        "n_skipped_unknown_word": n_skipped_word,
        "n_skipped_artifact_rejected": n_skipped_artifact,
        "n_test_trials": len(test_trials),
        "accuracy": float(accuracy),
    }


def summarize_final(final_results, fullscale_root):
    subjects = sorted(final_results.keys())
    before_first, after_first = [], []
    before_word, after_word = [], []

    for s in subjects:
        with open(os.path.join(fullscale_root, f"results_{s}.json")) as f:
            original = json.load(f)
        before_first.append(original["first_syllable_e2e"]["accuracy"] * 100.0)
        after_first.append(final_results[s]["first_syllable_e2e"]["accuracy"] * 100.0)
        if original["full_word_e2e"].get("available"):
            before_word.append(original["full_word_e2e"]["accuracy"] * 100.0)
        if final_results[s]["full_word_e2e"].get("available"):
            after_word.append(final_results[s]["full_word_e2e"]["accuracy"] * 100.0)

    if len(subjects) >= MIN_PAIRS_FOR_WILCOXON:
        diffs = np.array(after_first) - np.array(before_first)
        p_first = None if np.all(diffs == 0) else float(wilcoxon(after_first, before_first)[1])
    else:
        p_first = None

    return {
        "n_subjects": len(subjects), "subjects": subjects,
        "mean_first_syllable_before_pct": float(np.mean(before_first)),
        "mean_first_syllable_after_pct": float(np.mean(after_first)),
        "delta_first_syllable_pp": float(np.mean(after_first) - np.mean(before_first)),
        "wilcoxon_p_first_syllable": p_first,
        "mean_full_word_before_pct": float(np.mean(before_word)) if before_word else None,
        "mean_full_word_after_pct": float(np.mean(after_word)) if after_word else None,
        "n_full_word_before": len(before_word), "n_full_word_after": len(after_word),
        "per_subject_first_syllable_before_pct": dict(zip(subjects, before_first)),
        "per_subject_first_syllable_after_pct": dict(zip(subjects, after_first)),
    }


def write_final_integration_sections(final_summary, feat_group, phase3_summary):
    lines = []
    lines.append("\n---\n")
    lines.append("## Ablation Study & Final Combined Model")
    lines.append("")
    lines.append(
        "Full ablation study detail (Fase 1 individual factors, Fase 2 automatic combination, Fase 3 "
        "post-processing) is reproduced below verbatim from `P7_CoarseAblation_Phase1_report.md` -- "
        "see that file directly for the same content in isolation. The ORIGINAL E0-baseline Stage A/B "
        "results ABOVE this section are UNCHANGED -- everything from here down is purely additive."
    )
    lines.append("")

    if os.path.exists(ABLATION_REPORT_PATH):
        with open(ABLATION_REPORT_PATH, encoding="utf-8") as f:
            ablation_doc = f.read()
        # Demote the ablation doc's own top-level title so it nests as a
        # subsection here instead of competing as a second document title.
        ablation_doc = ablation_doc.replace(
            "# P7 Coarse Sub-model Ablation Study", "### Ablation Study Detail (Fase 1-3)", 1
        )
        lines.append(ablation_doc.rstrip("\n"))
    else:
        lines.append(f"_Ablation report not found at `{ABLATION_REPORT_PATH}` -- run the Fase 1-3 "
                     f"scripts (run_p7_coarse_ablation.py, run_p7_coarse_combined.py, "
                     f"run_p7_postprocessing.py) first._")
    lines.append("")

    lines.append("### Fase 4 -- Final Integration: End-to-end Metrics (Before vs. After)")
    lines.append("")
    lines.append(f"Strategi post-processing terpilih (Fase 3): **`{phase3_summary['winning_strategy']}`**. "
                 f"Feature group coarse: **`{feat_group}`**. Digabungkan dengan `fine_A`/`fine_I`/`fine_E`/"
                 f"`sa_branch` yang TIDAK dilatih ulang.")
    lines.append("")
    lines.append("| Subject | First-syllable e2e -- before (%) | First-syllable e2e -- after (%) | Delta (pp) |")
    lines.append("|---|---|---|---|")
    for s in final_summary["subjects"]:
        before = final_summary["per_subject_first_syllable_before_pct"][s]
        after = final_summary["per_subject_first_syllable_after_pct"][s]
        lines.append(f"| {s} | {before:.2f} | {after:.2f} | {after - before:+.2f} |")
    lines.append("")
    p_str = (f"{final_summary['wilcoxon_p_first_syllable']:.4f}"
             if final_summary["wilcoxon_p_first_syllable"] is not None else "n/a")
    lines.append(f"**Mean first-syllable e2e (n={final_summary['n_subjects']}):** "
                 f"before {final_summary['mean_first_syllable_before_pct']:.4f}% -> "
                 f"after {final_summary['mean_first_syllable_after_pct']:.4f}% "
                 f"(delta {final_summary['delta_first_syllable_pp']:+.4f}pp, Wilcoxon p={p_str})")
    lines.append("")
    if final_summary["mean_full_word_before_pct"] is not None and final_summary["mean_full_word_after_pct"] is not None:
        lines.append(f"**Mean full-word e2e:** before {final_summary['mean_full_word_before_pct']:.4f}% "
                     f"(n={final_summary['n_full_word_before']}) -> "
                     f"after {final_summary['mean_full_word_after_pct']:.4f}% "
                     f"(n={final_summary['n_full_word_after']})")
    else:
        lines.append("_Full-word e2e before/after tidak tersedia untuk seluruh subjek -- lihat file "
                     "`final_e2e_{subject}.json` per subjek._")
    lines.append("")

    lines.append("### Perbandingan terhadap Champion P3")
    lines.append("")
    lines.append(f"Champion P3 (`P3_SVM/E5_Data_Augmentation/S3/barlow`, klasifikasi flat 19-kelas): "
                 f"**{P3_CHAMPION_ACCURACY_PCT:.2f}%**.")
    lines.append(f"P7 final (setelah ablation + kombinasi + post-processing), first-syllable e2e: "
                 f"**{final_summary['mean_first_syllable_after_pct']:.2f}%** "
                 f"(rerata {final_summary['n_subjects']} subjek).")
    lines.append("")
    lines.append(
        "**Catatan kejujuran:** kedua angka di atas TIDAK sepenuhnya apel-ke-apel. Champion P3 mengukur "
        "klasifikasi flat 19-kelas langsung (satu model, satu keputusan per window). Metrik end-to-end P7 "
        "mengukur pipeline decoding HIERARKIS dua tahap (coarse -> fine, plus post-processing Fase 3) yang "
        "secara struktural berbeda -- termasuk potensi akumulasi error lintas tahap dan definisi 'benar' "
        "yang berbeda (identitas suku kata pertama dari 9 kelas, bukan salah satu dari 19 kelas flat). "
        "Perbandingan ini bersifat INDIKATIF (menunjukkan skala relatif), bukan perbandingan langsung yang "
        "valid secara statistik antara dua metode pada soal yang identik."
    )

    with open(MAIN_REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO][P7-FinalIntegration] Final section appended to {MAIN_REPORT_PATH}")


def run_final_integration(subject_ids=None):
    print(f"\n{'=' * 70}\n P7 Coarse Sub-model Ablation -- Fase 4 (Final Integration)\n{'=' * 70}")

    fullscale_root = p7base.setup_experiment(FULLSCALE_STAGE_DIR, pilar=PILAR)["weights"]

    phase3_summary_path = os.path.join(fullscale_root, "phase3_summary.json")
    if not os.path.exists(phase3_summary_path):
        raise RuntimeError(
            f"[P7-FinalIntegration] Fase 3 summary not found at {phase3_summary_path}. Run "
            f"`run_p7_postprocessing.py` first."
        )
    with open(phase3_summary_path) as f:
        phase3_summary = json.load(f)

    phase2_summary_path = os.path.join(fullscale_root, "phase2_summary.json")
    with open(phase2_summary_path) as f:
        phase2_data = json.load(f)
    feat_group = "all" if phase2_data["included_factors"]["D_feat_all"] else "barlow"

    winning_strategy = phase3_summary["winning_strategy"]
    chosen_threshold_per_subject = phase3_summary["chosen_threshold_per_subject"]
    print(f"[INFO][P7-FinalIntegration] Winning Fase 3 strategy: {winning_strategy}")

    if subject_ids is None:
        subject_ids = phase3_summary["subjects"]

    final_results = {}
    for subject_id in subject_ids:
        final_json = os.path.join(fullscale_root, f"final_e2e_{subject_id}.json")
        if os.path.exists(final_json):
            print(f"[SKIP][P7-FinalIntegration] {subject_id} final e2e already computed.")
            with open(final_json) as f:
                final_results[subject_id] = json.load(f)
            continue

        print(f"[INFO][P7-FinalIntegration] {subject_id}: loading cache + models...")
        cached = get_or_build_cached_coarse(subject_id)
        if cached is None:
            print(f"[WARNING][P7-FinalIntegration] No cached epochs for {subject_id}; skipping.")
            continue

        coarse_bundle = postproc.load_combined_coarse_bundle(fullscale_root, feat_group, subject_id)
        fine_bundles = postproc.load_fine_bundles(fullscale_root, subject_id)
        sa_bundle = load_sa_bundle(fullscale_root, subject_id)

        bundle_a = bundle_c = threshold = None
        if winning_strategy == "ensemble_voting":
            bundle_a = postproc.load_variant_bundle(
                fullscale_root, ablation.VARIANT_DIRS["A_balanced"], "variant_a", "barlow", subject_id
            )
            bundle_c = postproc.load_variant_bundle(
                fullscale_root, ablation.VARIANT_DIRS["C_tuned"], "variant_c", "barlow", subject_id
            )
        elif winning_strategy == "confidence_gated":
            threshold = chosen_threshold_per_subject.get(subject_id)

        first_syl_e2e = compute_final_first_syllable_e2e(
            cached, winning_strategy, coarse_bundle, fine_bundles, bundle_a, bundle_c, threshold
        )
        full_word_e2e = compute_final_full_word_e2e_accuracy(
            subject_id, winning_strategy, coarse_bundle, fine_bundles, sa_bundle, bundle_a, bundle_c, threshold
        )

        result = {
            "subject_id": subject_id, "winning_strategy": winning_strategy,
            "first_syllable_e2e": first_syl_e2e, "full_word_e2e": full_word_e2e,
        }
        final_results[subject_id] = result
        with open(final_json, "w") as f:
            json.dump(result, f, indent=2)

        fw = full_word_e2e
        fw_str = f"{fw['accuracy']*100:.2f}%" if fw.get("available") else "n/a"
        print(f"[INFO][P7-FinalIntegration] {subject_id}: first-syllable e2e "
              f"{first_syl_e2e['accuracy']*100:.2f}% | full-word e2e {fw_str}")

    if not final_results:
        raise RuntimeError("[P7-FinalIntegration] No subjects processed -- nothing to summarize/report.")

    final_summary = summarize_final(final_results, fullscale_root)
    with open(os.path.join(fullscale_root, "phase4_final_summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2)

    write_final_integration_sections(final_summary, feat_group, phase3_summary)
    return final_results, final_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P7 coarse sub-model ablation Fase 4: recompute end-to-end metrics using the "
                     "Fase 3 winning strategy + existing fine models, update P7_CoarseToFine_report.md."
    )
    parser.add_argument("--subjects", nargs="+", default=None,
                         help="Restrict to specific subject IDs (e.g. --subjects S1 S2). "
                              "Default: all subjects present in the Fase 3 summary.")
    args = parser.parse_args()
    run_final_integration(subject_ids=args.subjects)

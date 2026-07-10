"""
backend/src/experiments_p4_p7/run_followup_orchestrator.py

Single entry point for the PC-Lab follow-up run: P6 (exit-code-fixed
re-run) -> P7 coarse sub-model ablation pipeline (cache warm-up -> Fase 1
individual-factor ablation -> Fase 2 automatic combination -> Fase 3
post-processing -> Fase 4 final integration) -> P4_P7_Analysis.ipynb
(Restart & Run All), executed in that order, unattended.

Distinct from run_orchestrator_p4_p7.py (the original P4-P7 baseline
pipeline that produced the existing P4/P5/P6/P7 artifacts -- untouched,
not re-run here). This script is new, isolated code scoped only to the
follow-up work described in the P7 ablation study: re-running P6 now that
its baseline artifacts are in place, running the full coarse sub-model
ablation/combination/post-processing/final-integration chain, and
refreshing the analysis notebook. It does NOT re-run P4, P5, or P7's own
Stage A/B baseline (already complete -- P7's baseline artifacts are a
precondition this pipeline reads, not something it produces).

Each stage runs as an isolated subprocess (same memory-hygiene pattern
already used by run_orchestrator_p4_p7.py) and is checkpointed to
backend/reports/P4_P7_Experiments/followup_orchestrator_run_log.md before
the next stage starts. Every non-skipped stage runs regardless of earlier
failures (P6 is fully independent of the P7 chain; within the P7 chain,
each stage already fails fast with a clear error if its own prerequisite
is missing -- see each script's own RuntimeError message -- so letting the
orchestrator attempt every stage and log SUCCESS/FAILED per stage is
simpler and still accurate, rather than re-implementing dependency
tracking here). Failures are summarized at the end and the orchestrator
exits non-zero if any stage failed -- mirroring the same "never silently
report success" principle behind the P6 exit-code fix itself
(run_p6_transfer_overt_imagined.py).

Environment note: every stage up to and including Fase 4 needs whatever
environment already runs the other `run_p*.py` scripts (sklearn/numpy/
pandas/scipy -- no jupyter needed). The notebook stage needs
jupyter+nbconvert, which may live in a DIFFERENT environment (on the dev
machine this code was written on, `backend/venv` has sklearn but not
jupyter, while a separate root `.venv` has both) -- pass --notebook-python
if that's the case here too.

Usage:
    cd backend/src/experiments_p4_p7
    python run_followup_orchestrator.py
    python run_followup_orchestrator.py --subjects S1 S2        # restrict every stage to a subset
    python run_followup_orchestrator.py --skip-p6                # only the P7 chain + notebook
    python run_followup_orchestrator.py --skip-p7                # only P6 + notebook
    python run_followup_orchestrator.py --skip-cache-warm         # let each P7 stage build cache lazily instead
    python run_followup_orchestrator.py --skip-notebook           # skip the notebook re-run
    python run_followup_orchestrator.py --notebook-python "C:/path/to/python.exe"
    python run_followup_orchestrator.py --list-stages
"""
import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
REPORTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'reports', 'P4_P7_Experiments'))
LOG_PATH = os.path.join(REPORTS_DIR, "followup_orchestrator_run_log.md")

NOTEBOOK_DIR = os.path.join(REPO_ROOT, 'notebooks')
NOTEBOOK_FILENAME = 'P4_P7_Analysis.ipynb'

# Notebook stage only re-reads existing JSON results and does read-only
# inference (.predict()/.evaluate()) on already-trained models -- not real
# training -- so this is generous headroom, not an estimate of typical
# runtime.
DEFAULT_NOTEBOOK_TIMEOUT = 1800


def _timestamp():
    return datetime.now().isoformat(timespec="seconds")


def log(line):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def run_stage(name, cmd, cwd):
    log(f"- **[{_timestamp()}]** Stage `{name}`: STARTED -- `{' '.join(cmd)}` (cwd={cwd})")
    start = time.time()

    try:
        result = subprocess.run(cmd, cwd=cwd)
    except Exception as e:
        elapsed = time.time() - start
        log(f"- **[{_timestamp()}]** Stage `{name}`: FAILED ({elapsed:.1f}s) -- exception launching command: {e}")
        return False

    elapsed = time.time() - start
    if result.returncode != 0:
        log(f"- **[{_timestamp()}]** Stage `{name}`: FAILED ({elapsed:.1f}s) -- exited with code {result.returncode}")
        return False

    log(f"- **[{_timestamp()}]** Stage `{name}`: SUCCESS ({elapsed:.1f}s)")
    return True


def build_stages(args):
    py = sys.executable
    nb_py = args.notebook_python or sys.executable
    subject_flags = (["--subjects"] + args.subjects) if args.subjects else []

    stages = []
    if not args.skip_p6:
        stages.append(("p6", [py, "run_p6_transfer_overt_imagined.py"] + subject_flags, SCRIPT_DIR))

    if not args.skip_p7:
        if not args.skip_cache_warm:
            stages.append(("p7-cache-warm", [py, "p7_coarse_cache.py"] + subject_flags, SCRIPT_DIR))
        if not args.skip_ablation:
            stages.append(("p7-coarse-ablation", [py, "run_p7_coarse_ablation.py"] + subject_flags, SCRIPT_DIR))
        if not args.skip_combined:
            stages.append(("p7-coarse-combined", [py, "run_p7_coarse_combined.py"] + subject_flags, SCRIPT_DIR))
        if not args.skip_postprocessing:
            stages.append(("p7-postprocessing", [py, "run_p7_postprocessing.py"] + subject_flags, SCRIPT_DIR))
        if not args.skip_final_integration:
            stages.append(("p7-final-integration", [py, "run_p7_final_integration.py"] + subject_flags, SCRIPT_DIR))

    if not args.skip_notebook:
        stages.append((
            "notebook",
            [nb_py, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             NOTEBOOK_FILENAME, f"--ExecutePreprocessor.timeout={args.notebook_timeout}"],
            NOTEBOOK_DIR,
        ))
    return stages


def main():
    parser = argparse.ArgumentParser(
        description="PC-Lab follow-up orchestrator: P6 -> P7 coarse sub-model ablation pipeline "
                     "(cache warm-up -> Fase 1-4) -> P4_P7_Analysis.ipynb (Restart & Run All), run "
                     "unattended in that order."
    )
    parser.add_argument("--subjects", nargs="+", default=None,
                         help="Restrict every stage to specific subject IDs (e.g. --subjects S1 S2). "
                              "Default: all subjects, auto-discovered/inherited by each stage. Does "
                              "not affect the notebook stage.")
    parser.add_argument("--skip-p6", action="store_true", help="Skip the P6 re-run stage.")
    parser.add_argument("--skip-p7", action="store_true",
                         help="Skip the entire P7 ablation chain (cache warm-up through Fase 4). "
                              "Overrides the individual --skip-cache-warm/--skip-ablation/etc flags.")
    parser.add_argument("--skip-cache-warm", action="store_true",
                         help="Skip pre-warming the coarse cache -- Fase 1 (run_p7_coarse_ablation.py) "
                              "will still build it lazily per subject on first use, just without a "
                              "dedicated up-front stage.")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip Fase 1 (individual factor ablation).")
    parser.add_argument("--skip-combined", action="store_true", help="Skip Fase 2 (automatic combination).")
    parser.add_argument("--skip-postprocessing", action="store_true", help="Skip Fase 3 (post-processing).")
    parser.add_argument("--skip-final-integration", action="store_true", help="Skip Fase 4 (final integration).")
    parser.add_argument("--skip-notebook", action="store_true",
                         help="Skip the notebook Restart & Run All stage.")
    parser.add_argument("--notebook-python", default=None,
                         help="Path to a python executable that has jupyter/nbconvert installed, "
                              "used ONLY for the notebook stage (defaults to the interpreter running "
                              "this script -- override this if that environment lacks jupyter/"
                              "nbconvert, e.g. if you normally launch this with backend/venv's python "
                              "but jupyter lives in a separate environment).")
    parser.add_argument("--notebook-timeout", type=int, default=DEFAULT_NOTEBOOK_TIMEOUT,
                         help=f"Per-cell execution timeout in seconds for the notebook stage "
                              f"(default: {DEFAULT_NOTEBOOK_TIMEOUT}).")
    parser.add_argument("--list-stages", action="store_true",
                         help="Print the resolved stage commands (given the other flags) and exit, "
                              "without running anything.")
    args = parser.parse_args()

    stages = build_stages(args)

    if args.list_stages:
        for name, cmd, cwd in stages:
            print(f"{name}: {' '.join(cmd)}  (cwd={cwd})")
        return

    if not stages:
        print("[INFO] All stages skipped -- nothing to do.")
        return

    print(f"[INFO] Running P6/P7 stages with: {sys.executable}")
    print(f"[INFO] Running notebook stage with: {args.notebook_python or sys.executable}")

    log(f"\n---\n\n## Follow-up orchestrator run started: {_timestamp()}")
    log(f"Stages queued: {[name for name, _, _ in stages]}\n")

    summary = []
    for name, cmd, cwd in stages:
        ok = run_stage(name, cmd, cwd)
        summary.append((name, ok))

    log("\n### Run summary\n")
    log("| Stage | Result |")
    log("|---|---|")
    any_failed = False
    for name, ok in summary:
        log(f"| {name} | {'SUCCESS' if ok else 'FAILED'} |")
        any_failed = any_failed or (not ok)

    if any_failed:
        failed_names = [name for name, ok in summary if not ok]
        log(f"\n**[WARNING] {len(failed_names)} stage(s) failed: {failed_names}.** "
            f"Re-run the failed stage directly to see its full error output (e.g. "
            f"`python run_p6_transfer_overt_imagined.py` for P6, `python run_p7_coarse_ablation.py` "
            f"for Fase 1, etc). Stages later in the P7 chain will also fail fast (with a clear "
            f"RuntimeError naming the missing prerequisite file) if an earlier P7 stage failed -- "
            f"fix the earliest failure first.")
    else:
        log(f"\n**All {len(summary)} stage(s) completed successfully.**")

    log(f"\nFollow-up orchestrator run finished: {_timestamp()}\n")
    print(f"\n[INFO] Full run log: {LOG_PATH}")

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

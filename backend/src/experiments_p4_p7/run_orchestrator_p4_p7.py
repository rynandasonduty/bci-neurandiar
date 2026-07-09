"""
backend/src/experiments_p4_p7/run_orchestrator_p4_p7.py

Single entry point that runs the entire P4-P7 pipeline automatically and in
sequence, with no manual pauses: pre-flight verification (Langkah 0.3/0.4)
-> P4 Stage A -> P4 Stage B -> P5 Stage A -> P5 Stage B -> P6 -> P7 Stage A
-> P7 Stage B. The researcher runs one command on the lab machine and can
leave it unattended until it finishes.

Every stage runs as an isolated subprocess (matching the memory-hygiene
pattern already used by models/train_word_assembler.py for repeated
large-raw-CSV processing on this memory-constrained machine) and is
checkpointed to backend/reports/P4_P7_Experiments/orchestrator_run_log.md
before the next stage starts. A failed stage is logged and the pipeline
CONTINUES to the next stage rather than aborting -- one experiment's failure
must never cost the researcher the rest of an unattended overnight run.
Failures are summarized again at the very end.

All per-stage decisions (including feature-group selection) are already
fully automatic inside each stage script -- --resume-from exists purely as
a safety net for resuming after an interruption (crash, restart, power
loss), not as an intentional pause for manual input.

Usage:
    cd backend/src/experiments_p4_p7
    python run_orchestrator_p4_p7.py
    python run_orchestrator_p4_p7.py --resume-from p6
    python run_orchestrator_p4_p7.py --list-stages
"""
import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'reports', 'P4_P7_Experiments'))
LOG_PATH = os.path.join(REPORTS_DIR, "orchestrator_run_log.md")

STAGES = [
    ("verify", [
        [sys.executable, "verify_p6_phase_labels.py"],
        [sys.executable, "verify_p7_label_scheme.py"],
    ]),
    ("p4-stage-a", [[sys.executable, "run_p4_nowindowing.py", "--stage", "a"]]),
    ("p4-stage-b", [[sys.executable, "run_p4_nowindowing.py", "--stage", "b"]]),
    ("p5-stage-a", [[sys.executable, "run_p5_shifted_bandpass.py", "--stage", "a"]]),
    ("p5-stage-b", [[sys.executable, "run_p5_shifted_bandpass.py", "--stage", "b"]]),
    ("p6", [[sys.executable, "run_p6_transfer_overt_imagined.py"]]),
    ("p7-stage-a", [[sys.executable, "run_p7_coarse_to_fine.py", "--stage", "a"]]),
    ("p7-stage-b", [[sys.executable, "run_p7_coarse_to_fine.py", "--stage", "b"]]),
]
STAGE_NAMES = [name for name, _ in STAGES]


def _timestamp():
    return datetime.now().isoformat(timespec="seconds")


def log(line):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def run_stage(name, cmds):
    log(f"- **[{_timestamp()}]** Stage `{name}`: STARTED")
    start = time.time()

    for cmd in cmds:
        cmd_str = " ".join(cmd)
        try:
            result = subprocess.run(cmd, cwd=SCRIPT_DIR)
        except Exception as e:
            elapsed = time.time() - start
            log(f"- **[{_timestamp()}]** Stage `{name}`: FAILED ({elapsed:.1f}s) -- "
                f"exception launching `{cmd_str}`: {e}")
            return False

        if result.returncode != 0:
            elapsed = time.time() - start
            log(f"- **[{_timestamp()}]** Stage `{name}`: FAILED ({elapsed:.1f}s) -- "
                f"`{cmd_str}` exited with code {result.returncode}")
            return False

    elapsed = time.time() - start
    log(f"- **[{_timestamp()}]** Stage `{name}`: SUCCESS ({elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="P4-P7 unattended orchestrator.")
    parser.add_argument("--resume-from", choices=STAGE_NAMES, default=None,
                         help="Skip stages before this one (safety net for resuming after an "
                              "interruption -- e.g. lab machine restart or power loss -- not a "
                              "manual pause; every stage's own decisions are already automatic).")
    parser.add_argument("--list-stages", action="store_true",
                         help="Print the stage sequence and exit, without running anything.")
    args = parser.parse_args()

    if args.list_stages:
        for name in STAGE_NAMES:
            print(name)
        return

    start_idx = STAGE_NAMES.index(args.resume_from) if args.resume_from else 0
    stages_to_run = STAGES[start_idx:]

    log(f"\n---\n\n## Orchestrator run started: {_timestamp()}")
    if args.resume_from:
        log(f"Resuming from stage `{args.resume_from}` (stages before it are skipped, not re-run).")
    log(f"Stages queued: {[name for name, _ in stages_to_run]}\n")

    summary = []
    for name, cmds in stages_to_run:
        ok = run_stage(name, cmds)
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
            f"Successful stages' results are preserved. Re-run with "
            f"`--resume-from <stage>` after investigating, or re-run this stage's script "
            f"directly to see its full error output.")
    else:
        log(f"\n**All {len(summary)} stage(s) completed successfully.**")

    log(f"\nOrchestrator run finished: {_timestamp()}\n")

    print(f"\n[INFO] Full run log: {LOG_PATH}")
    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

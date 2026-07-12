# P4 -- No-Windowing: Experiment Report

Variable tested: epoch length (one full 5-second epoch vs. the standard five 1-second windows). Locked: 0.5-50Hz broadband filter, SVM, subject-dependent architecture, E0 Baseline (no augmentation), phase_filter='all'.

> **Status: PARTIAL (11/12 subjects).** This report reflects a smoke-test or in-progress run, not the full 12-subject grid. Re-run `run_p4_nowindowing.py --stage b` on the lab machine to complete the grid; already-completed subjects are skipped automatically (auto-resume).

## Stage A -- Feature Spot-check (S3, E0)

| Feature Group | Test Accuracy (%) | Class Coverage |
|---|---|---|
| time | 4.3478 | 1/19 |
| hjorth | 0.0000 | 0/19 |
| barlow | 0.0000 | 0/19 |
| band_ratio | 0.0000 | 0/19 |
| all | 0.0000 | 0/19 |

**Automatic selection:** `time` -- highest test accuracy, no tie within 1pp

**[PERINGATAN]** Akurasi spot-check pemenang tidak melampaui chance level (5.26% untuk 19 kelas). Hasil skala penuh tetap dijalankan otomatis; perlu ditinjau kritis sebelum dimasukkan ke Bab 6.

## Stage B -- Full-scale Results

Feature group used: `time` | Subjects completed: 11/12
Subjects skipped (no raw data found): ['S6']

| Subject | Test Accuracy (%) | Val Accuracy (%) | Class Coverage |
|---|---|---|---|
| S10 | 12.9032 | 19.3548 | 3/19 |
| S11 | 5.8824 | 0.0000 | 1/19 |
| S12 | 10.0000 | 0.0000 | 2/19 |
| S1 | 0.0000 | 0.0000 | 0/19 |
| S2 | 0.0000 | 25.0000 | 0/19 |
| S3 | 4.3478 | 13.0435 | 1/19 |
| S4 | 7.1429 | 14.2857 | 1/19 |
| S5 | 0.0000 | 16.6667 | 0/19 |
| S6 | -- | -- | not yet run |
| S7 | 2.5641 | 7.6923 | 1/19 |
| S8 | 8.6957 | 4.3478 | 2/19 |
| S9 | 14.2857 | 0.0000 | 1/19 |

Mean test accuracy so far: 5.9838% (std 5.1325 pp, n=11)

## Prior pilot context

A same-day prior pilot (S3 only, Barlow only) is preserved at `backend/models/weights/P4_NoWindowing/E0_Baseline/` -- see `backend/reports/p4_no_windowing_pilot_report.md` (0% test accuracy at n=106) and `p4_control_subsampled_report.md` (matched-size subsampling control, mean ~6.09% test accuracy, attributing the pilot's 0% mainly to small sample size, z~-1.57, not significant). This report's Stage A `barlow`/S3 result is a useful consistency check against that pilot, since both use the same data, split seed, and model config.

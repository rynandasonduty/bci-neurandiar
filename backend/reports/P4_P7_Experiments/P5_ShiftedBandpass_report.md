# P5 -- Shifted Bandpass Filter: Experiment Report

Variable tested: bandpass range (15-65Hz vs. the standard 0.5-50Hz broadband). Locked: standard 5x1s windowing, SVM, E0 Baseline (no augmentation), phase_filter='all'.

## Stage A -- Feature Spot-check (S3, E0)

| Feature Group | Test Accuracy (%) | Class Coverage |
|---|---|---|
| time | 8.2781 | 12/19 |
| hjorth | 12.2517 | 14/19 |
| barlow | 15.5629 | 19/19 |
| band_ratio | 6.2914 | 12/19 |
| all | 9.2715 | 15/19 |

**Automatic selection:** `barlow` -- highest test accuracy, no tie within 1pp

## Stage B -- Full-scale Results

Feature group used: `barlow` | Subjects completed: 12/12

| Subject | Test Accuracy (%) | Val Accuracy (%) | Class Coverage |
|---|---|---|---|
| S10 | 11.8812 | 9.9010 | 15/19 |
| S11 | 9.9010 | 8.2781 | 13/19 |
| S12 | 6.6225 | 8.2781 | 12/19 |
| S1 | 8.0537 | 10.7383 | 14/19 |
| S2 | 11.4478 | 13.4680 | 15/19 |
| S3 | 15.5629 | 15.8940 | 19/19 |
| S4 | 15.1815 | 9.2715 | 17/19 |
| S5 | 8.2781 | 9.2715 | 11/19 |
| S6 | 10.9272 | 8.6093 | 14/19 |
| S7 | 10.2649 | 7.9470 | 14/19 |
| S8 | 9.4276 | 8.4175 | 12/19 |
| S9 | 7.3333 | 9.3333 | 12/19 |

Mean test accuracy so far: 10.4068% (std 2.8281 pp, n=12)

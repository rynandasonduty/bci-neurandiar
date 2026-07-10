# P7 -- Coarse-to-Fine Hierarchical Decoding: Experiment Report

Variable tested: decision structure (hierarchical vowel-group -> syllable vs. flat 19-way). Locked: standard windowing/filter, SVM, E0 Baseline, phase_filter='all'. One three_way_split(seed=42) per subject; all 5 sub-models derived by filtering that same split by label.

## Stage A -- Coarse Sub-model Feature Spot-check (S3, E0)

| Feature Group | Test Accuracy (%) | Class Coverage (/4) |
|---|---|---|
| time | 35.2381 | 3 |
| hjorth | 39.0476 | 3 |
| barlow | 41.9048 | 4 |
| band_ratio | 36.1905 | 3 |
| all | 41.9048 | 3 |

**Automatic selection (coarse sub-model):** `barlow` -- tie-break within 1pp among ['all', 'barlow']; barlow preferred (robust/consistent across P1-P3)

`fine_A`/`fine_I`/`fine_E`/`sa_branch` always use Barlow directly (no spot-check), per the agreed design -- their classification granularity is equivalent to P1-P3.

## Stage B -- Full-scale Sub-model Results

Coarse feature group used: `barlow` | Subjects completed: 12/12

| Subject | coarse (%) | fine_A (%) | fine_I (%) | fine_E (%) | sa_branch (%) |
|---|---|---|---|---|---|
| S10 | 38.89 | 62.75 | 44.74 | 75.00 | 70.37 |
| S11 | 42.86 | 51.16 | 54.55 | 70.00 | 54.55 |
| S12 | 33.33 | 36.96 | 51.52 | 54.55 | 78.26 |
| S1 | 42.86 | 40.00 | 34.78 | 73.33 | 81.25 |
| S2 | 40.00 | 47.83 | 77.78 | 50.00 | 69.23 |
| S3 | 41.90 | 65.12 | 65.62 | 76.19 | 54.17 |
| S4 | 41.84 | 52.38 | 46.67 | 55.56 | 80.00 |
| S5 | 38.60 | 44.00 | 31.25 | 63.64 | 80.00 |
| S6 | 26.32 | 28.57 | 57.14 | 50.00 | 50.00 |
| S7 | 41.09 | 52.08 | 40.48 | 50.00 | 33.33 |
| S8 | 36.36 | 48.84 | 32.35 | 38.10 | 54.55 |
| S9 | 43.82 | 37.14 | 53.33 | 82.35 | 66.67 |

## End-to-end Metrics

| Subject | First-syllable e2e acc (%) | Full-word e2e acc (%) | Full-word n test trials |
|---|---|---|---|
| S10 | 22.22 | 12.50 | 40 |
| S11 | 22.86 | 16.22 | 37 |
| S12 | 11.71 | 7.69 | 39 |
| S1 | 17.14 | 14.29 | 28 |
| S2 | 25.00 | 7.14 | 28 |
| S3 | 28.57 | 13.89 | 36 |
| S4 | 22.45 | 2.70 | 37 |
| S5 | 14.04 | 18.52 | 27 |
| S6 | 15.79 | 10.00 | 10 |
| S7 | 19.38 | 12.50 | 40 |
| S8 | 14.55 | 5.26 | 38 |
| S9 | 17.98 | 18.92 | 37 |

Mean first-syllable e2e accuracy so far: 19.3068% (n=12)
Mean full-word e2e accuracy so far: 11.6358% (n=12)

### Full-word e2e methodology caveat

Trial-level 80/20 holdout (test_size=0.2, random_state=42), mirroring `train_word_assembler_s3.py`'s own methodology exactly, so the two are comparable 'word accuracy' figures. This trial-level split is independent of the window-level three_way_split used to train the coarse/fine/sa_branch sub-models -- some evaluated trials may have contributed windows to those sub-models' own training data, so this is not a strictly leakage-free estimate. The same caveat already applies to the existing word assembler's reported accuracy (it pools ALL trials, including ones whose windows trained the champion SVM, and only holds out its own 20% for the assembler's fit) -- so this is at least as rigorous as existing precedent. Treat 'first-syllable e2e accuracy' above (computed from the proper held-out window-level test split) as the leakage-free estimate for this paradigm.

### Baseline reference: P3 per-syllable recall, 9 first-syllable classes (T18)

| Syllable | Mean Recall (P3, %) | Std Recall (pp) | N Subjects |
|---|---|---|---|
| MA | 13.41 | 19.89 | 12 |
| MI | 17.91 | 12.05 | 12 |
| BE | 25.48 | 13.06 | 12 |
| PI | 15.67 | 15.85 | 12 |
| MAN | 9.79 | 18.89 | 12 |
| BO | 5.49 | 12.93 | 12 |
| LE | 7.18 | 10.17 | 12 |
| SA | 25.25 | 26.14 | 12 |
| TI | 12.07 | 18.17 | 12 |

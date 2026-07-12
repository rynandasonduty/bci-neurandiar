# P7 Coarse Sub-model Ablation Study

Scope: ONLY the `coarse` sub-model is varied below. `fine_A`/`fine_I`/`fine_E`/`sa_branch` are untouched throughout -- reused as-is from `P7_CoarseToFine/Fullscale_12Subj/`. Every variant shares the exact same coarse-filtered `three_way_split(seed=42)` as the existing coarse baseline (via `p7_coarse_cache.py`'s cached raw split + Barlow features).

## Fase 1 -- Individual Factor Ablation

- **Varian A (balanced):** `WeightedClassicalClassifier` -- `class_weight='balanced'` SVC, cached Barlow features.
- **Varian B (augmented):** plain SVM, training set enriched with E5-recipe augmented copies (`add_noise=True, noise_factor=0.05, apply_jitter=True, jitter_ms=10`) applied to training epochs only.
- **Varian C (tuned):** per-subject C in {0.1, 1, 10, 100, 1000}, selected by validation accuracy.
- **Varian D (feat=all):** only alternative feature group tested ('all' tied with 'barlow' at Stage A spot-check; the other three groups were clearly behind).
- **Varian E (calibrated):** baseline SVM wrapped in `CalibratedClassifierCV(cv=5, method='sigmoid')`; reports raw accuracy AND calibration quality (Brier score, ECE) since its purpose is enabling Fase 3's confidence gating, not raw accuracy alone.

### Per-subject coarse test accuracy (%)

| Subject | Baseline | A: balanced | B: augmented | C: tuned | D: feat=all | E: calibrated |
|---|---|---|---|---|---|---|
| S1 | 42.86 | 40.00 | 35.71 | 41.43 | 31.43 | 37.14 |
| S10 | 38.89 | 35.71 | 36.51 | 45.24 | 34.92 | 42.86 |
| S11 | 42.86 | 35.24 | 40.00 | 43.81 | 37.14 | 40.00 |
| S12 | 33.33 | 27.93 | 28.83 | 36.04 | 34.23 | 39.64 |
| S2 | 40.00 | 36.67 | 36.67 | 40.00 | 36.67 | 46.67 |
| S3 | 41.90 | 40.95 | 46.67 | 40.95 | 41.90 | 40.00 |
| S4 | 41.84 | 46.94 | 41.84 | 42.86 | 34.69 | 44.90 |
| S5 | 38.60 | 43.86 | 47.37 | 43.86 | 40.35 | 45.61 |
| S6 | 26.32 | 26.32 | 15.79 | 36.84 | 15.79 | 52.63 |
| S7 | 41.09 | 33.33 | 41.09 | 38.76 | 30.23 | 42.64 |
| S8 | 36.36 | 31.82 | 36.36 | 45.45 | 34.55 | 44.55 |
| S9 | 43.82 | 43.82 | 42.70 | 41.57 | 46.07 | 42.70 |

### Mean test accuracy, delta vs. baseline, Wilcoxon p-value (n=12)

| Candidate | Mean (%) | Std (pp) | Delta vs baseline (pp) | Wilcoxon p |
|---|---|---|---|---|
| baseline | 38.99 | 5.00 | +0.00 | n/a |
| A_balanced | 36.88 | 6.46 | -2.11 | 0.1602 |
| B_augmented | 37.46 | 8.52 | -1.53 | 0.3594 |
| C_tuned | 41.40 | 3.07 | +2.41 | 0.1543 |
| D_feat_all | 34.83 | 7.45 | -4.16 | 0.0186 |
| E_calibrated | 43.28 | 4.07 | +4.29 | 0.0640 |

### Chosen C per subject (Varian C)

| Subject | Chosen C | val@C=0.1 | val@C=1 | val@C=10 | val@C=100 | val@C=1000 |
|---|---|---|---|---|---|---|
| S1 | 1 | 37.68 | 47.83 | 47.83 | 47.83 | 47.83 |
| S10 | 1 | 40.48 | 42.06 | 37.30 | 38.10 | 38.10 |
| S11 | 1 | 40.00 | 48.57 | 44.76 | 39.05 | 39.05 |
| S12 | 1 | 42.34 | 46.85 | 45.05 | 39.64 | 39.64 |
| S2 | 1 | 38.33 | 45.00 | 30.00 | 38.33 | 38.33 |
| S3 | 1 | 40.95 | 48.57 | 41.90 | 41.90 | 41.90 |
| S4 | 0.1 | 43.75 | 38.54 | 38.54 | 39.58 | 39.58 |
| S5 | 0.1 | 44.64 | 42.86 | 35.71 | 37.50 | 37.50 |
| S6 | 0.1 | 40.00 | 25.00 | 30.00 | 30.00 | 30.00 |
| S7 | 0.1 | 41.09 | 41.09 | 36.43 | 37.98 | 34.11 |
| S8 | 1 | 39.09 | 40.00 | 34.55 | 30.00 | 30.00 |
| S9 | 1 | 39.33 | 49.44 | 40.45 | 40.45 | 40.45 |

### Varian E calibration quality

| Subject | Test Brier score (lower better) | Test ECE (lower better) |
|---|---|---|
| S1 | 0.7089 | 0.0444 |
| S10 | 0.6849 | 0.0353 |
| S11 | 0.6858 | 0.0160 |
| S12 | 0.7031 | 0.0800 |
| S2 | 0.6818 | 0.0780 |
| S3 | 0.6720 | 0.0844 |
| S4 | 0.6706 | 0.0175 |
| S5 | 0.6706 | 0.1732 |
| S6 | 0.6564 | 0.1466 |
| S7 | 0.6946 | 0.0421 |
| S8 | 0.6994 | 0.0551 |
| S9 | 0.6782 | 0.0525 |

### Fase 2 inclusion preview (automatic rule: delta > 1.0 pp)

This is a PREVIEW only -- the actual inclusion decision (and Varian E's exception, see below) is made by `run_p7_coarse_combined.py` when it runs, using these same numbers.

| Variant | Delta vs baseline (pp) | Included by threshold? |
|---|---|---|
| A_balanced | -2.11 | no |
| B_augmented | -1.53 | no |
| C_tuned | +2.41 | YES |
| D_feat_all | -4.16 | no |
| E_calibrated | +4.29 | YES |

**Note on Varian E:** regardless of whether it clears the 1pp threshold above, Varian E (calibration) is always included STRUCTURALLY in the Fase 2 combined model (wrapped at the outer layer) -- its purpose is supporting Fase 3's confidence gating, not raising raw accuracy, so the inclusion threshold does not apply to it the same way. This is intentional, not a rule violation -- see run_p7_coarse_combined.py's docstring.

---

## Fase 2 -- Kombinasi Otomatis

Aturan inklusi: sebuah Varian Fase 1 diikutsertakan jika (rerata akurasi 12 subjek) - (rerata baseline) > 1.0 pp.

| Varian | Delta vs baseline (pp) | Diikutsertakan? | Alasan |
|---|---|---|---|
| A_balanced | -2.11 | tidak | delta -2.11pp <= 1.0pp |
| B_augmented | -1.53 | tidak | delta -1.53pp <= 1.0pp |
| C_tuned | +2.41 | YA | delta +2.41pp > 1.0pp |
| D_feat_all | -4.16 | tidak | delta -4.16pp <= 1.0pp |
| E_calibrated | +4.29 | YA | SELALU disertakan secara struktural (kalibrasi, bukan untuk akurasi mentah) -- lihat catatan di bawah |

**Catatan Varian E:** disertakan di lapisan terluar model final terlepas dari lolos/tidaknya ambang 1pp di atas, karena fungsinya mendukung Fase 3 (confidence gating), bukan menaikkan akurasi mentah -- ini BUKAN pelanggaran aturan inklusi, melainkan aturan yang berbeda untuk tujuan yang berbeda, dijelaskan eksplisit di sini supaya tidak disalahpahami.

**Komposisi model kombinasi final:** per-subject tuned C + dibungkus CalibratedClassifierCV (selalu).

### Tabel akhir: baseline -> tiap varian individual -> kombinasi final

| Kandidat | Mean akurasi (%) | Delta vs baseline (pp) | Wilcoxon p |
|---|---|---|---|
| baseline | 38.99 | +0.00 | -- |
| A_balanced | 36.88 | -2.11 | 0.1602 |
| B_augmented | 37.46 | -1.53 | 0.3594 |
| C_tuned | 41.40 | +2.41 | 0.1543 |
| D_feat_all | 34.83 | -4.16 | 0.0186 |
| E_calibrated | 43.28 | +4.29 | 0.0640 |
| **kombinasi final** | **42.44** | **+3.45** | **0.1763** |

### Per-subject: kombinasi final

| Subject | Baseline (%) | Kombinasi Final (%) | Delta (pp) | Feature | C | class_weight | Augmented |
|---|---|---|---|---|---|---|---|
| S1 | 42.86 | 41.43 | -1.43 | barlow | 1 | - | no |
| S10 | 38.89 | 43.65 | +4.76 | barlow | 1 | - | no |
| S11 | 42.86 | 44.76 | +1.90 | barlow | 1 | - | no |
| S12 | 33.33 | 37.84 | +4.50 | barlow | 1 | - | no |
| S2 | 40.00 | 38.33 | -1.67 | barlow | 1 | - | no |
| S3 | 41.90 | 40.95 | -0.95 | barlow | 1 | - | no |
| S4 | 41.84 | 43.88 | +2.04 | barlow | 0.1 | - | no |
| S5 | 38.60 | 45.61 | +7.02 | barlow | 0.1 | - | no |
| S6 | 26.32 | 47.37 | +21.05 | barlow | 0.1 | - | no |
| S7 | 41.09 | 35.66 | -5.43 | barlow | 0.1 | - | no |
| S8 | 36.36 | 48.18 | +11.82 | barlow | 1 | - | no |
| S9 | 43.82 | 41.57 | -2.25 | barlow | 1 | - | no |


---

## Fase 3 -- Post-processing (Tanpa Training Ulang)

Seluruh strategi di bawah memakai model kombinasi final (Fase 2, dibungkus kalibrasi) dan model `fine_A`/`fine_I`/`fine_E` yang SUDAH ADA, tanpa melatih ulang apa pun.

### Strategi 1 -- Confidence-gated Coarse->Fine

Threshold diuji pada VALIDATION set per subjek, dipilih otomatis berdasarkan akurasi validasi tertinggi, baru dievaluasi SEKALI ke test set (test set tetap murni held-out).

| Subject | Threshold Terpilih | val@0.4 | val@0.5 | val@0.6 | val@0.7 | val@0.8 | Test Acc (gated, %) |
|---|---|---|---|---|---|---|---|
| S1 | 0.4 | 26.09 | 17.39 | 17.39 | 17.39 | 17.39 | 15.71 |
| S10 | 0.5 | 23.81 | 24.60 | 24.60 | 24.60 | 24.60 | 24.60 |
| S11 | 0.5 | 17.14 | 19.05 | 19.05 | 19.05 | 19.05 | 21.90 |
| S12 | 0.5 | 17.12 | 21.62 | 21.62 | 21.62 | 21.62 | 19.82 |
| S2 | 0.6 | 18.33 | 23.33 | 25.00 | 25.00 | 25.00 | 15.00 |
| S3 | 0.5 | 32.38 | 33.33 | 31.43 | 31.43 | 31.43 | 24.76 |
| S4 | 0.5 | 21.88 | 22.92 | 22.92 | 22.92 | 22.92 | 20.41 |
| S5 | 0.5 | 23.21 | 26.79 | 26.79 | 26.79 | 26.79 | 26.32 |
| S6 | 0.5 | 20.00 | 25.00 | 25.00 | 25.00 | 25.00 | 21.05 |
| S7 | 0.4 | 23.26 | 21.71 | 21.71 | 21.71 | 21.71 | 17.83 |
| S8 | 0.4 | 20.91 | 20.00 | 20.00 | 20.00 | 20.00 | 20.91 |
| S9 | 0.5 | 19.10 | 20.22 | 20.22 | 20.22 | 20.22 | 21.35 |

### Perbandingan 3 Strategi (akurasi end-to-end suku kata pertama, test set)

| Subject | (1) Kombinasi Final Saja (%) | (2) Confidence-gated (%) | (3) Ensemble Voting (%) |
|---|---|---|---|
| S1 | 11.43 | 15.71 | 11.43 |
| S10 | 26.19 | 24.60 | 26.98 |
| S11 | 22.86 | 21.90 | 21.90 |
| S12 | 14.41 | 19.82 | 13.51 |
| S2 | 25.00 | 15.00 | 25.00 |
| S3 | 27.62 | 24.76 | 27.62 |
| S4 | 24.49 | 20.41 | 25.51 |
| S5 | 19.30 | 26.32 | 19.30 |
| S6 | 21.05 | 21.05 | 21.05 |
| S7 | 17.83 | 17.83 | 19.38 |
| S8 | 20.91 | 20.91 | 20.91 |
| S9 | 17.98 | 21.35 | 16.85 |

**Mean (n=12):** Kombinasi Final Saja 20.76% | Confidence-gated 20.81% | Ensemble Voting 20.79%

**Strategi terpilih otomatis (Fase 4 memakai ini):** `confidence_gated` -- akurasi end-to-end suku kata pertama rerata tertinggi di antara ketiganya.

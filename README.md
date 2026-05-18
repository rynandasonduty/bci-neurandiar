# NEURANDIAR-BCI
**Real-Time EEG-Based Imagined Speech Decoding via EEGNet and Classical ML**

![Status](https://img.shields.io/badge/Status-Active_Development-emerald)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![React](https://img.shields.io/badge/Frontend-Next.js_16-black)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-00a3ff)

Neurandiar is a Brain-Computer Interface (BCI) research prototype developed as an undergraduate thesis project at Institut Teknologi Sepuluh Nopember (ITS). The system decodes imagined and overt speech from 14-channel EEG signals recorded with the Emotiv EPOC X headset, classifying 19 Indonesian syllables and assembling them into 10 target words. Three independent classification paradigms are investigated across eight ablation experiments, evaluated with rigorous non-parametric statistical testing.

---

## Overview

The research objective is to establish whether a consumer-grade EEG headset can reliably decode imagined speech at the syllable level for clinical communication assistance. The system addresses this through a three-paradigm architecture:

| Paradigm | Description | Model |
|----------|-------------|-------|
| **P1 — Global** | Single EEGNet trained on pooled data from all subjects | EEGNet-8,2 |
| **P2 — Subject-Dependent** | Per-subject EEGNet trained and evaluated individually | EEGNet-8,2 |
| **P3 — Classical ML** | Per-subject SVM trained on handcrafted EEG features | SVM (RBF) |

---

## Research Paradigms and Experiment Grid

Eight preprocessing and augmentation conditions (E0–E7) are applied identically across all three paradigms, yielding 584+ trained model artefacts:

| ID | Experiment | Key Variable |
|----|-----------|--------------|
| E0 | Baseline | Broadband filter, 256 Hz, all channels |
| E1 | ICA Artifact Removal | FastICA with kurtosis-based component rejection |
| E2 | Resampling | Upsample to 512 Hz |
| E3 | ERP N400 Window | Crop to 200–600 ms post-stimulus |
| E4 | Channel Ablation | Language cortex channels only (F7, F3, FC5, T7, P7) |
| E5 | Data Augmentation | Gaussian noise injection + temporal jittering |
| E6 | Cross-Modality | Imagined speech phase only |
| E7 | Frequency Band | Alpha band (8–13 Hz) isolation |

P3 additionally tests five feature groups (time-domain, Hjorth, Barlow, band-power ratios, all) across all eight experiments, yielding 480 SVM models (5 groups × 8 experiments × 12 subjects).

---

## System Architecture

```
Emotiv EPOC X (14ch EEG, 256 Hz)
        |
        | LSL / Cortex API
        v
 backend/src/acquisition/
   experiment_runner.py        -- pygame acquisition protocol (overt/imagined phases)
   experiment_runner_cortex.py -- Cortex API variant
   cortex_client.py            -- Emotiv WebSocket API wrapper
        |
        v
 backend/src/preprocessing/
   signal_processor.py   -- Butterworth bandpass, ICA, resampling, windowing
   build_dataset.py       -- epoch extraction, phase filtering, channel selection
   build_logreg_dataset.py -- slot probability feature construction for word assembler
        |
        v
 backend/src/models/
   eegnet_model.py         -- EEGNet-8,2 Keras architecture
   classical_models.py     -- SVM / Random Forest wrapper
   logreg_model.py         -- Logistic Regression word assembler (19-dim x2 -> 10 words)
   train_pipeline.py       -- Optuna HPO + MLflow logging
   run_master_experiments.py   -- end-to-end P1 orchestrator (E0-E7)
   run_subject_dependent.py    -- P2 EEGNet subject-dependent grid
   run_e8_classical.py         -- P3 SVM feature ablation grid
   evaluate_model.py       -- syllable and word accuracy evaluation
   explain_model.py        -- SHAP GradientExplainer interpretability
        |
        v
 backend/src/api/
   main.py                -- FastAPI REST + WebSocket inference server
        |
        v
 frontend/
   app/                   -- Next.js 16 / React 19 clinical dashboard
   components/pages/      -- LiveSession, EEGMonitor, Evaluation, History
```

### MLOps Directory Structure (Golden Standard)

```
backend/models/weights/
  P1_Global/
    E0_Baseline/
      eegnet_trained_E0_Baseline.h5
      scaler_E0_Baseline.pkl
    E1_ICA_Filtering/ ...
    ...
  P2_EEGNet/
    E0_Baseline/
      E0_Baseline_SUBJ01.h5
      scaler_E0_Baseline_SUBJ01.pkl
    ...
  P3_SVM/
    E0_Baseline/
      SVM_all_E0_Baseline_SUBJ01.pkl
      scaler_SVM_all_E0_Baseline_SUBJ01.pkl
    ...

backend/logs/                  -- experiment acquisition logs
backend/mlflow.db              -- MLflow SQLite tracking database
```

---

## Repository Structure

```
neurandiar-bci/
  backend/
    src/
      acquisition/    -- EEG recording scripts (pygame + Cortex API)
      api/            -- FastAPI inference server
      config.py       -- Golden Standard path engine (setup_experiment)
      features/       -- Handcrafted EEG feature extractor (E8/P3)
      models/         -- EEGNet, SVM, LogReg, training scripts
      pipeline/       -- Real-time stream processor (stub)
      preprocessing/  -- Signal processing, dataset builders, QC tools
      utils/          -- data_utils.py (anti-leakage split + scaler)
    requirements.txt
  frontend/
    app/              -- Next.js pages
    components/       -- React UI components
  notebooks/
    BCI_Master_Journal_Q1_Final.ipynb  -- full analysis notebook
    outputs/          -- generated figures
    reports/data_export_claude/        -- CSV exports for analysis
  README.md
```

---

## Installation and Usage

### Backend (FastAPI + Training Engine)

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

**Run the inference API:**
```bash
python -m src.api.main
# Server available at http://127.0.0.1:8000
```

**Run the full experiment pipeline (P1):**
```bash
cd backend/src/models
python run_master_experiments.py
```

**Run the subject-dependent EEGNet grid (P2):**
```bash
python run_subject_dependent.py
```

**Run the SVM feature ablation grid (P3):**
```bash
python run_e8_classical.py
```

**Validate pipeline anti-leakage integrity:**
```bash
python smoke_test.py
```

### Frontend (Clinical Dashboard)

```bash
cd frontend
npm install --legacy-peer-deps
npm run dev
# Dashboard available at http://localhost:3000
```

### Notebooks (Statistical Analysis)

Open `notebooks/BCI_Master_Journal_Q1_Final.ipynb` in Jupyter. All generated figures are saved to `notebooks/outputs/`.

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| EEG acquisition protocol (overt + imagined) | Complete | pygame + LSL or Cortex API |
| Signal preprocessing pipeline (E0–E7) | Complete | Butterworth, ICA, resampling, ERP windowing |
| EEGNet-8,2 training (P1 + P2) | Complete | Optuna HPO, MLflow tracking, early stopping |
| SVM feature ablation (P3 / E8) | Complete | 480 models across 5 feature groups |
| Word assembler (Logistic Regression) | Complete | 38-dim slot probability input |
| Statistical evaluation | Complete | Friedman + Nemenyi + Holm-Bonferroni, bootstrap CI |
| SHAP explainability | Partial | Computation complete; sample size limited to 3 test samples |
| Transfer learning (E10) | Partial | Fine-tuning layer selection has a known bug (all layers frozen) |
| Real-time inference (WebSocket) | Not implemented | Currently simulated with random outputs |
| Emotiv Cortex live data ingestion | Not implemented | CortexClient stub only |
| LLM sentence refinement | Not implemented | llm_agent.py is an empty placeholder |
| `/api/metrics` endpoint | Not implemented | Evaluation dashboard returns 404 |

---

## Data Anti-Leakage Design

All experiments follow a strict anti-leakage protocol:

- `three_way_split()` applies stratified 70/15/15 partitioning before any normalisation.
- `fit_and_apply_scaler()` fits `StandardScaler` exclusively on the training split and transforms validation and test sets independently.
- Data augmentation (E5) is applied post-split to training data only.
- Test sets are serialised to disk at training time and never revisited during hyperparameter search.

---

## Statistical Evaluation Pipeline

Performance is compared across experiments using the following non-parametric pipeline:

1. **Friedman test** — omnibus test across all experiment conditions per paradigm.
2. **Nemenyi post-hoc test** — pairwise condition comparisons.
3. **Holm-Bonferroni correction** — family-wise error rate control.
4. **Effect sizes** — rank-biserial correlation and Cohen's d.
5. **Bootstrap 95% confidence intervals** — on mean accuracy per condition.

---

## Research Acknowledgements

Developed as an undergraduate thesis (Skripsi) at Institut Teknologi Sepuluh Nopember (ITS), Department of Informatics.

**Author:** Andiar Rinanda

**Supervisor:** [Supervisor name]

**Device:** Emotiv EPOC X, 14-channel EEG, 256 Hz native sampling rate

**Dataset:** 12 subjects, 200 trials per subject (100 overt + 100 imagined), 19-class syllable taxonomy, 10 target words

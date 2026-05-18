# NEURANDIAR-BCI

**Real-Time EEG-Based Imagined Speech Decoding via Decoupled Acoustic-Language Architecture**

![Status](https://img.shields.io/badge/Status-Thesis_Defense_Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI_0.135-00a3ff)
![React](https://img.shields.io/badge/Frontend-Next.js_16_·_React_19-black)
![MLflow](https://img.shields.io/badge/MLOps-MLflow_3.10-0194E2)

---

## Abstract

NEURANDIAR is a Brain-Computer Interface (BCI) research prototype developed as an undergraduate thesis at Institut Teknologi Sepuluh Nopember (ITS). The system decodes imagined and overt speech from 14-channel EEG signals recorded using the Emotiv EPOC X consumer-grade headset, classifying 19 Indonesian syllables and assembling them into 10 clinically relevant target words. The overarching research question is whether consumer-grade EEG can sustain reliable syllable-level imagined speech decoding for clinical communication assistance applications.

The investigation employs a three-paradigm, eight-experiment ablation framework yielding 584+ trained model artefacts, evaluated with a rigorous non-parametric statistical pipeline (Friedman omnibus test, Nemenyi post-hoc, Holm-Bonferroni correction, rank-biserial effect sizes, bootstrap 95% confidence intervals). A fourth paradigm, transfer learning from the global champion model to individual subjects (P4), is architecturally complete and prepared for future evaluation.

**Author:** Andiar Rinanda Agastya
**Institution:** Institut Teknologi Sepuluh Nopember (ITS), Department of Informatics
**Device:** Emotiv EPOC X, 14-channel EEG, 256 Hz native sampling rate
**Dataset:** 12 subjects · 200 trials/subject · 19-class syllable taxonomy · 10 target words

---

## Table of Contents

- [NEURANDIAR-BCI](#neurandiar-bci)
  - [Abstract](#abstract)
  - [Table of Contents](#table-of-contents)
  - [1. System Architecture](#1-system-architecture)
  - [2. Three-Paradigm MLOps Training Pipeline](#2-three-paradigm-mlops-training-pipeline)
  - [3. Inference Pipeline: Acoustic-Language Decoupling](#3-inference-pipeline-acoustic-language-decoupling)
  - [4. Hardware and Dataset Specifications](#4-hardware-and-dataset-specifications)
    - [EEG Hardware](#eeg-hardware)
    - [Dataset](#dataset)
    - [19-Class Syllable Taxonomy](#19-class-syllable-taxonomy)
  - [5. Technology Stack](#5-technology-stack)
    - [Backend](#backend)
    - [Frontend](#frontend)
    - [Infrastructure](#infrastructure)
  - [6. Repository Structure](#6-repository-structure)
  - [7. MLOps Directory Structure (Golden Standard)](#7-mlops-directory-structure-golden-standard)
  - [8. Experiment Grid: E0–E7](#8-experiment-grid-e0e7)
  - [9. Deep Technical Definitions](#9-deep-technical-definitions)
    - [9.1 EEGNet-8,2 Architecture](#91-eegnet-82-architecture)
    - [9.2 Anti-Leakage Data Protocol](#92-anti-leakage-data-protocol)
    - [9.3 Acoustic-Language Decoupling](#93-acoustic-language-decoupling)
    - [9.4 Statistical Evaluation Pipeline](#94-statistical-evaluation-pipeline)
    - [9.5 SHAP GradientExplainer](#95-shap-gradientexplainer)
  - [10. Installation and Environment Setup](#10-installation-and-environment-setup)
    - [Prerequisites](#prerequisites)
    - [Step 1 — Clone the Repository](#step-1--clone-the-repository)
    - [Step 2 — Backend Python Environment](#step-2--backend-python-environment)
    - [Step 3 — Environment Variables](#step-3--environment-variables)
    - [Step 4 — Frontend Node Environment](#step-4--frontend-node-environment)
  - [11. Usage Guide for Future Researchers](#11-usage-guide-for-future-researchers)
    - [11.1 System Integrity Check (Always Run First)](#111-system-integrity-check-always-run-first)
    - [11.2 Starting the Backend API](#112-starting-the-backend-api)
    - [11.3 Starting the Frontend Dashboard](#113-starting-the-frontend-dashboard)
    - [11.4 Running the P1 Global Training Pipeline](#114-running-the-p1-global-training-pipeline)
    - [11.5 Running the P2 Subject-Dependent EEGNet Grid](#115-running-the-p2-subject-dependent-eegnet-grid)
    - [11.6 Running the P3 Classical SVM Feature Ablation Grid](#116-running-the-p3-classical-svm-feature-ablation-grid)
    - [11.7 Running P4 Transfer Learning](#117-running-p4-transfer-learning)
    - [11.8 Replicating the Q1 Journal Analysis (Jupyter Notebook)](#118-replicating-the-q1-journal-analysis-jupyter-notebook)
  - [12. Implementation Status](#12-implementation-status)
  - [13. Future Work and Research Roadmap](#13-future-work-and-research-roadmap)
  - [14. Research Acknowledgements](#14-research-acknowledgements)

---

## 1. System Architecture

The end-to-end system spans four layers: physical hardware, acquisition, a multi-paradigm training and inference backend, and a clinical-grade Next.js dashboard. All layers communicate via standardized interfaces (WebSocket, REST, LSL).

```mermaid
graph TB
    subgraph HW["Hardware Layer"]
        EEG["Emotiv EPOC X\n14-channel EEG · 256 Hz\nConsumer-grade BCI headset"]
        CORTEX["Emotiv Cortex API\nwss://localhost:6868\nProprietary WebSocket SDK"]
        EEG -->|"Raw EEG stream"| CORTEX
    end

    subgraph ACQ["Acquisition Layer  ·  backend/src/acquisition/"]
        RUNNER["experiment_runner_cortex.py\nPygame stimulus protocol\n5 s overt + 2 s pause + 5 s imagined\n200 trials · 10 words · 12 subjects"]
        CLIENT["cortex_client.py\nEmotiv WebSocket API wrapper\nEEG + motion data subscription"]
        CORTEX --> CLIENT
        RUNNER <-->|"LSL / WebSocket"| CLIENT
    end

    subgraph PREP["Preprocessing Layer  ·  backend/src/preprocessing/"]
        SP["signal_processor.py\nButterworth bandpass, FastICA,\nresampling, ERP windowing"]
        BD["build_dataset.py\nEpoch extraction, phase filtering,\nchannel selection, QC"]
        LRBD["build_logreg_dataset.py\nSlot probability feature builder\n19-dim × 2 slots → 38-dim word features"]
        CLIENT -->|"Raw EEG CSV"| SP
        SP --> BD
        BD --> LRBD
    end

    subgraph MODEL["Model Layer  ·  backend/src/models/"]
        EEGNET["EEGNet-8,2\nTemporal + Depthwise + Separable Conv\nSoftmax(19 syllables)"]
        ASSEMBLER["WordAssembler\nLogisticRegression\nSoftmax(10 words)"]
        SVM["SVM-RBF\nHandcrafted features\n5 feature groups"]
        BD --> EEGNET
        LRBD --> ASSEMBLER
        BD --> SVM
    end

    subgraph MLOPS["MLOps Layer"]
        MLFLOW["MLflow 3.10\nSQLite tracking backend\nParameters · Metrics · Artefact URIs"]
        OPTUNA["Optuna 4.7\nHyperparameter optimization\nDropout · LR · Kernel size"]
        EEGNET <--> OPTUNA
        EEGNET --> MLFLOW
        SVM --> MLFLOW
    end

    subgraph API["Inference API  ·  backend/src/api/main.py  ·  FastAPI 0.135"]
        WS_T["/ws/telemetry\nReal-time EEG oscilloscope stream"]
        WS_I["/ws/inference\nEnd-to-end decoding pipeline"]
        REST["REST Endpoints\nGET  /api/metrics\nPOST /api/v1/calibrate\nGET  /api/logs\nPOST /api/logs"]
        EEGNET --> WS_I
        ASSEMBLER --> WS_I
    end

    subgraph FE["Clinical Dashboard  ·  Next.js 16 · React 19  ·  localhost:3000"]
        LS["LiveSession\nInference control panel\nPipeline stepper · Result display"]
        MC["MonitorControl\nEEG oscilloscope\nHardware telemetry"]
        EVAL["Evaluation\nMLflow metrics · Training curves\nOptuna trials · Dataset meta"]
        HIST["History\nInference session log\nCSV export"]
        WS_T --> MC
        WS_I --> LS
        REST --> EVAL
        REST --> HIST
    end
```

---

## 2. Three-Paradigm MLOps Training Pipeline

Three independent paradigms are trained across eight identical ablation conditions, enabling direct cross-paradigm and within-paradigm comparisons under controlled preprocessing variables.

```mermaid
flowchart TD
    RAW["Raw EEG Dataset\n12 Subjects × 200 Trials/Subject\n19-class Syllable Taxonomy · 10 Target Words"]

    RAW --> ABLATION

    subgraph ABLATION["8 Ablation Experiments Applied Identically Across All Paradigms"]
        direction LR
        E0["E0\nBaseline"]
        E1["E1\nICA Filter"]
        E2["E2\n512 Hz"]
        E3["E3\nN400 ERP"]
        E4["E4\nChannel"]
        E5["E5\nAugment"]
        E6["E6\nImagined"]
        E7["E7\nAlpha"]
    end

    ABLATION --> SPLIT

    subgraph ANTILEAK["Anti-Leakage Data Protocol"]
        SPLIT["three_way_split()\nStratified 70 / 15 / 15\nSplit before any normalization"]
        SCALER["fit_and_apply_scaler()\nStandardScaler fitted on train split only\nVal + test transformed independently\nScaler persisted to .pkl alongside model"]
        SPLIT --> SCALER
    end

    SCALER --> P1 & P2 & P3

    subgraph P1["P1 — Global EEGNet  ·  8 artefact sets"]
        P1A["Pool all 12 subjects' epochs\ninto a single training set"]
        P1B["Optuna HPO · 5 trials\ndropout_rate · kern_length"]
        P1C["EEGNet-8,2 production run\nMax 300 epochs · Early stopping\nReduceLROnPlateau"]
        P1D["P1 Champion Model\n→ LogReg WordAssembler training\n→ SHAP explainability analysis"]
        P1A --> P1B --> P1C --> P1D
    end

    subgraph P2["P2 — Subject-Dependent EEGNet  ·  96 artefact sets"]
        P2A["Per-subject train / val / test splits\n(no cross-subject data leakage)"]
        P2B["Optuna HPO per subject × experiment\n5 trials per configuration"]
        P2C["EEGNet-8,2 × 12 subjects × 8 experiments\n= 96 trained models"]
        P2A --> P2B --> P2C
    end

    subgraph P3["P3 — Classical SVM  ·  480 artefact sets"]
        P3A["Handcrafted feature extraction\nTime-domain · Hjorth · Barlow\nBand-power ratio · All-combined"]
        P3B["NaN / Inf sanitization via np.nan_to_num\n(ICA numerical stability guard)"]
        P3C["SVM-RBF with Optuna C / γ search\n5 groups × 12 subjects × 8 experiments\n= 480 trained models"]
        P3A --> P3B --> P3C
    end

    subgraph P4["P4 — Transfer Learning  ·  Prepared · Not yet evaluated"]
        P4A["Load P1 Champion model weights\n(freeze all layers below classification head)"]
        P4B["isinstance() layer type check\n(Dense · Softmax · Activation remain trainable)"]
        P4C["Fine-tune per subject · LR = 1e-4\nMax 50 epochs · EarlyStopping patience=10"]
        P4A --> P4B --> P4C
    end

    P1 & P2 & P3 --> GOLDEN

    subgraph GOLDEN["Golden Standard MLOps Artefact Paths"]
        direction LR
        G1["models/weights/P1_Global/{exp_id}/\neegnet_trained_{exp}.h5\nscaler_{exp}.pkl\nX_test.npy · y_test.npy\nlogreg_assembler_{exp}.pkl"]
        G2["models/weights/P2_EEGNet/{exp_id}/\n{exp}_{subj}.h5\nscaler_{exp}_{subj}.pkl\nXtest_{exp}_{subj}.npy\nytest_{exp}_{subj}.npy"]
        G3["models/weights/P3_SVM/{exp_id}/\nSVM_{feat}_{exp}_{subj}.pkl\nscaler_SVM_{feat}_{exp}_{subj}.pkl"]
    end

    GOLDEN --> MLFLOW_TRACK["MLflow 3.10 Experiment Tracking\nSQLite · backend/logs/mlflow/mlruns.db\nAll parameters · metrics · artefact URIs logged"]
    GOLDEN --> STAT["Statistical Evaluation\nFriedman omnibus · Nemenyi post-hoc\nHolm-Bonferroni · Bootstrap 95% CI\nRank-biserial r · Cohen's d"]
```

---

## 3. Inference Pipeline: Acoustic-Language Decoupling

The inference architecture deliberately separates two cognitive levels of the decoding problem. The **Acoustic Decoder** (EEGNet) operates at the neurophysiological signal level, classifying individual syllables from raw EEG epochs. The **Language Assembler** (LogisticRegression) operates at the lexical level, mapping the probability distributions emitted by two sequential syllable slots into a word-class decision.

This decoupling is architecturally motivated: the 10 target words share syllabic constituents (e.g., the syllable `SA` appears in both *Sakit* and *Sayang*), making a direct EEG-to-word mapping ill-posed. By exposing the full 19-dimensional probability vector from each slot — rather than a hard argmax — the word assembler can resolve lexical ambiguity using the joint conditional distribution over both slots simultaneously.

```mermaid
sequenceDiagram
    participant HW as Emotiv EPOC X<br/>14ch · 256 Hz
    participant BUF as EEG Buffer<br/>(WebSocket /ws/inference)
    participant SP as SignalProcessor
    participant NORM as StandardScaler<br/>(fitted on train split)
    participant EEG as EEGNet-8,2<br/>Acoustic Decoder
    participant ASSEM as WordAssembler<br/>LogisticRegression
    participant LLM as LLM Agent<br/>(Future — not implemented)
    participant FE as Clinical Dashboard

    HW->>BUF: Raw EEG buffer (14 × T samples)
    BUF->>SP: process(buffer)
    SP->>SP: 1. Butterworth bandpass filter (0.5–40 Hz)
    SP->>SP: 2. FastICA artifact removal (optional, E1)
    SP->>SP: 3. Resampling (256 → 512 Hz, E2 only)
    SP->>SP: 4. ERP windowing (200–600 ms, E3 only)
    SP->>NORM: Epoch tensor (14 × samples)
    NORM->>NORM: Apply pre-fitted StandardScaler<br/>(zero-copy transform — scaler NOT re-fitted)
    NORM->>EEG: X_epoch reshaped to (1, 14, samples, 1)

    Note over EEG: Block 1: Temporal Conv2D (F1=8, kern=samples/2)<br/>→ DepthwiseConv2D (D=2, MaxNorm=1)<br/>→ BatchNorm → ELU → AvgPool(1,4) → Dropout

    Note over EEG: Block 2: SeparableConv2D (F2=16, kern=16)<br/>→ BatchNorm → ELU → AvgPool(1,8) → Dropout

    Note over EEG: Classification: Flatten → Dense(19, MaxNorm=0.25)<br/>→ Softmax(19 syllable classes)

    EEG-->>ASSEM: P(syllable | slot_1)  [19-dim probability vector]
    EEG-->>ASSEM: P(syllable | slot_2)  [19-dim probability vector]

    Note over ASSEM: Concatenate → 38-dim feature vector<br/>LogisticRegression.predict_proba() → Softmax(10 words)<br/>Resolves shared-syllable ambiguity<br/>(e.g., SA in Sakit vs. Sayang)

    ASSEM->>LLM: decoded_word_sequence (future)
    Note over LLM: Planned: LLM refines raw word sequence<br/>into grammatical Indonesian sentence
    LLM--xFE: refined_sentence (not yet implemented)

    ASSEM->>FE: decoded_word + confidence + syllable_pair
    FE->>FE: Display result panel
    FE->>FE: POST /api/logs → append to inference_history.csv
```

---

## 4. Hardware and Dataset Specifications

### EEG Hardware

| Parameter | Specification |
|-----------|--------------|
| Device | Emotiv EPOC X |
| Electrode count | 14 active channels + 2 reference (CMS/DRL) |
| Channel layout | AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4 (10–20 system) |
| Native sampling rate | 256 Hz |
| Resolution | 14-bit ADC |
| Connectivity | Bluetooth 5.0 (2.4 GHz) |
| SDK | Emotiv Cortex API v2 (WebSocket, `wss://localhost:6868`) |
| Acquisition protocol | overt speech (5 s) → silence pause (2 s) → imagined speech (5 s) |

### Dataset

| Parameter | Value |
|-----------|-------|
| Subjects | 12 (S1–S12) |
| Trials per subject | 200 (20 per word × 10 words) |
| Block size | 20 trials per recording block |
| Recording phases | 2 per trial: overt + imagined |
| Syllable classes | 19 (see taxonomy below) |
| Word classes | 10 |
| Chance level | 5.26% (1/19 for syllable classification) |

### 19-Class Syllable Taxonomy

The 10 target words decompose into 19 unique syllabic units. The syllable `SA` is shared between *Sakit* and *Sayang*, creating the lexical ambiguity that motivates the two-stage Acoustic-Language decoupling architecture.

| Word (Indonesian) | Translation | Slot 1 Syllable | Slot 2 Syllable |
|-------------------|-------------|-----------------|-----------------|
| Makan | eat | MA | KAN |
| Minum | drink | MI | NUM |
| Berak | defecate | BE | RAK |
| Pipis | urinate | PI | PIS |
| Mandi | bathe | MAN | DI |
| Bosan | bored | BO | SAN |
| Lelah | tired | LE | LAH |
| Sakit | pain / sick | SA *(shared)* | KIT |
| Tidur | sleep | TI | DUR |
| Sayang | love / dear | SA *(shared)* | YANG |

---

## 5. Technology Stack

### Backend

| Component | Library / Version | Role |
|-----------|------------------|------|
| Deep learning | TensorFlow 2.21.0 · Keras 3.13.2 | EEGNet-8,2 training and inference |
| Classical ML | scikit-learn 1.8.0 | SVM-RBF, LogisticRegression, StandardScaler |
| HPO | Optuna 4.7.0 | Hyperparameter search (dropout, kernel, LR) |
| Experiment tracking | MLflow 3.10.1 | Parameter logging, metric tracking, artefact registry |
| Interpretability | SHAP 0.51.0 | GradientExplainer channel importance |
| Signal processing | NumPy 2.4.3 · SciPy 1.17.1 | Filtering, resampling, feature extraction |
| Numerical analysis | pandas 2.3.3 · seaborn 0.13.2 | Statistical aggregation and visualization |
| API framework | FastAPI 0.135.1 · uvicorn 0.41.0 | REST and WebSocket inference server |
| Data acquisition | pylsl 1.18.1 | Lab Streaming Layer EEG synchronization |
| Serialization | h5py 3.14.0 · joblib 1.5.3 | Model weights (.h5) and scaler (.pkl) persistence |

### Frontend

| Component | Library / Version | Role |
|-----------|------------------|------|
| Framework | Next.js 16.2.4 · React 19.2.4 | Clinical dashboard SPA |
| Language | TypeScript 5.7.3 | Type-safe component development |
| Styling | Tailwind CSS 4.2 | Utility-first layout and theming |
| UI primitives | Radix UI (full suite) · shadcn/ui | Accessible component library |
| Charts | Recharts 2.15 | EEG oscilloscope and accuracy trend charts |
| Forms | React Hook Form 7.54 · Zod 3.24 | Validated input components |

### Infrastructure

| Component | Detail |
|-----------|--------|
| Python version | 3.10 or later (3.11 recommended) |
| Node.js version | 18 LTS or later |
| OS | Windows 11 (primary), Ubuntu 22.04+ (compatible) |
| GPU (optional) | CUDA 12.x + cuDNN 8.x (accelerates EEGNet training; inference runs on CPU) |
| Storage | Minimum 4 GB for full model artefact set (584+ files) |

---

## 6. Repository Structure

```
neurandiar-bci/
├── backend/
│   ├── dataset/
│   │   └── raw/                        # Raw EEG CSV files and experiment logs
│   │       └── logs/                   # Per-subject _experiment_log.txt files
│   ├── logs/
│   │   ├── mlflow/
│   │   │   └── mlruns.db               # MLflow SQLite tracking database
│   │   └── inference_history.csv       # Runtime inference session log
│   ├── models/
│   │   └── weights/                    # Golden Standard artefact root
│   │       ├── P1_Global/              # Global EEGNet (8 experiment dirs)
│   │       ├── P2_EEGNet/              # Subject-dependent EEGNet (8 × 12)
│   │       ├── P3_SVM/                 # Classical SVM (8 × 12 × 5 feature groups)
│   │       └── P4_TransferLearning/    # Transfer learning fine-tuned models
│   ├── src/
│   │   ├── acquisition/
│   │   │   ├── cortex_client.py        # Emotiv Cortex API WebSocket wrapper
│   │   │   ├── experiment_runner.py    # Pygame acquisition (LSL variant)
│   │   │   └── experiment_runner_cortex.py  # Pygame acquisition (Cortex API variant)
│   │   ├── api/
│   │   │   └── main.py                 # FastAPI server (REST + WebSocket)
│   │   ├── config.py                   # Golden Standard path engine (setup_experiment)
│   │   ├── features/
│   │   │   └── feature_extractor.py    # Handcrafted EEG features for P3/SVM
│   │   ├── models/
│   │   │   ├── classical_models.py     # SVM and Random Forest wrappers
│   │   │   ├── eegnet_model.py         # EEGNet-8,2 Keras architecture
│   │   │   ├── evaluate_model.py       # Syllable and word accuracy evaluation
│   │   │   ├── explain_model.py        # SHAP GradientExplainer analysis
│   │   │   ├── logreg_model.py         # Word assembler (LogisticRegression)
│   │   │   ├── run_e8_classical.py     # P3 SVM feature ablation runner
│   │   │   ├── run_master_experiments.py  # P1 end-to-end orchestrator (E0–E7)
│   │   │   ├── run_p4_transfer_learning.py  # P4 transfer learning runner
│   │   │   ├── run_subject_dependent.py    # P2 subject-dependent EEGNet runner
│   │   │   └── train_pipeline.py       # Optuna HPO + MLflow training core
│   │   ├── pipeline/
│   │   │   └── llm_agent.py            # LLM sentence refinement (stub — future work)
│   │   ├── preprocessing/
│   │   │   ├── build_dataset.py        # Primary epoch dataset builder
│   │   │   ├── build_logreg_dataset.py # Word-level feature dataset builder
│   │   │   └── signal_processor.py     # Bandpass, ICA, resample, ERP windowing
│   │   └── utils/
│   │       └── data_utils.py           # three_way_split() + fit_and_apply_scaler()
│   └── requirements.txt
├── frontend/
│   ├── app/                            # Next.js App Router pages
│   ├── components/
│   │   └── pages/
│   │       ├── LiveSession.tsx         # Inference control panel (WebSocket client)
│   │       ├── MonitorControl.tsx      # EEG oscilloscope (WebSocket client)
│   │       ├── Evaluation.tsx          # Metrics dashboard (REST client)
│   │       └── History.tsx             # Inference session log viewer
│   ├── lib/
│   │   └── api.ts                      # API_URL / WS_URL from NEXT_PUBLIC_* env vars
│   └── package.json
├── notebooks/
│   ├── BCI_Master_Journal_Q1_Final.ipynb  # Primary Q1 analysis notebook
│   ├── outputs/                        # Generated figures (PNG)
│   └── reports/
│       └── data_export_claude/         # CSV exports for LLM-assisted analysis
│           └── INDEX.json              # Export manifest
├── run_system_diagnostics.py           # Repository integrity checker (8 checks)
└── README.md
```

---

## 7. MLOps Directory Structure (Golden Standard)

All training artefacts follow a strict co-location convention enforced by `config.setup_experiment()`. Every model weight file, scaler, and held-out test array for a given experiment-subject pair reside in the same directory, eliminating path-chasing and guaranteeing reproducibility.

```
backend/models/weights/
│
├── P1_Global/
│   └── {exp_id}/                               # One dir per experiment (E0–E7)
│       ├── eegnet_trained_{exp_id}.h5           # Trained EEGNet weights
│       ├── scaler_{exp_id}.pkl                  # StandardScaler (train-fit)
│       ├── X_test.npy                           # Held-out test features (sealed)
│       ├── y_test.npy                           # Held-out test labels  (sealed)
│       └── logreg_assembler_{exp_id}.pkl        # Trained WordAssembler
│
├── P2_EEGNet/
│   └── {exp_id}/
│       ├── {exp_id}_{subj_id}.h5               # Per-subject EEGNet weights
│       ├── scaler_{exp_id}_{subj_id}.pkl        # Per-subject scaler
│       ├── Xtest_{exp_id}_{subj_id}.npy         # Held-out test features
│       └── ytest_{exp_id}_{subj_id}.npy         # Held-out test labels
│
├── P3_SVM/
│   └── {exp_id}/
│       ├── SVM_{feat_group}_{exp_id}_{subj_id}.pkl        # Trained SVM
│       └── scaler_SVM_{feat_group}_{exp_id}_{subj_id}.pkl # Feature scaler
│
└── P4_TransferLearning/
    └── {exp_id}/
        ├── TL_{exp_id}_{subj_id}.h5            # Fine-tuned model
        ├── scaler_TL_{exp_id}_{subj_id}.pkl     # Per-subject scaler
        ├── Xtest_TL_{exp_id}_{subj_id}.npy      # Held-out test features
        └── ytest_TL_{exp_id}_{subj_id}.npy      # Held-out test labels
```

The function `setup_experiment(exp_id, pilar="P1_Global")` in `backend/src/config.py` constructs and guarantees the existence of this directory tree for any given experiment-paradigm combination, returning a dictionary of guaranteed-existent absolute paths.

---

## 8. Experiment Grid: E0–E7

Eight preprocessing and augmentation conditions are applied identically across all three paradigms, enabling controlled ablation analysis.

| ID | Name | Key Preprocessing Variable | Channels | Sampling Rate | Notes |
|----|------|---------------------------|----------|--------------|-------|
| E0 | Baseline | Broadband Butterworth (0.5–40 Hz), no ICA | All 14 | 256 Hz | Reference condition |
| E1 | ICA Artifact Removal | FastICA · Kurtosis rejection threshold > 3.0 | All 14 | 256 Hz | Removes ocular/muscle artifacts |
| E2 | Resampling | Bilinear upsampling to 512 Hz | All 14 | 512 Hz | Increased temporal resolution |
| E3 | ERP N400 Window | Crop to 200–600 ms post-stimulus | All 14 | 256 Hz | Targets N400 cognitive response window |
| E4 | Channel Ablation | Language cortex subset only | F7, F3, FC5, T7, P7 | 256 Hz | Broca + Wernicke region hypothesis |
| E5 | Data Augmentation | Gaussian noise (σ=5%) + temporal jitter (±10 ms) | All 14 | 256 Hz | Applied to train split only |
| E6 | Cross-Modality | Imagined speech epochs only | All 14 | 256 Hz | Eliminates overt speech contribution |
| E7 | Frequency Band | Alpha band isolation (8–13 Hz) | All 14 | 256 Hz | Tests spectral specificity of decoding |

**P3 Feature Groups** (applied additionally across all E0–E7 for the SVM paradigm):

| Feature Group | Description | Dimensionality |
|---------------|-------------|----------------|
| `all` | Union of all features below | Full vector |
| `time` | Statistical moments: mean, variance, skewness, kurtosis | 4 × 14 channels |
| `hjorth` | Activity, Mobility, Complexity | 3 × 14 channels |
| `barlow` | Barlow's complexity features | Variable |
| `band_ratio` | Power ratios across delta/theta/alpha/beta/gamma | 5 × 14 channels |

---

## 9. Deep Technical Definitions

### 9.1 EEGNet-8,2 Architecture

EEGNet-8,2 is a compact depthwise convolutional neural network optimized for small EEG datasets. The `8,2` suffix denotes F1=8 temporal filters and D=2 depthwise spatial multiplier (yielding F2=16 separable filters).

```
Input:  (batch, channels=14, samples=256, 1)
         ↓
Block 1: Conv2D(F1=8, kernel=(1, kern_length))    # Temporal: learn frequency content
         BatchNorm(axis=1)
         DepthwiseConv2D(depth_multiplier=D=2,     # Spatial: learn channel weighting
                         depthwise_constraint=MaxNorm(1.))
         BatchNorm(axis=1) → ELU
         AveragePooling2D((1, 4))
         Dropout(rate)
         ↓
Block 2: SeparableConv2D(F2=16, kernel=(1, 16))   # Separable: efficient feature mixing
         BatchNorm(axis=1) → ELU
         AveragePooling2D((1, 8))
         Dropout(rate)
         ↓
Classification: Flatten
                Dense(nb_classes=19, MaxNorm(0.25))
                Softmax → P(syllable | EEG epoch)
```

**Key design decisions:**
- `kern_length` defaults to `samples // 2` (128 at 256 Hz), capturing roughly 2 Hz temporal resolution.
- `AveragePooling2D` size in Block 2 is computed dynamically (`remaining_samples = samples // 4`) to prevent shape crashes for short ERP windows (E3: 400 ms → 103 samples).
- `MaxNorm` constraints on DepthwiseConv and Dense layers act as implicit regularizers without an explicit L2 penalty.
- Early stopping monitors `val_loss` with patience scaled to the epoch budget: 50 epochs for production runs (≥200 epochs), or 30% of the budget for Optuna trial runs.

### 9.2 Anti-Leakage Data Protocol

All experiments enforce a strict three-phase anti-leakage protocol implemented in `backend/src/utils/data_utils.py`:

1. **`three_way_split(X, y, test_size=0.15, val_size=0.15)`** — Applies stratified splitting via `sklearn.model_selection.StratifiedShuffleSplit`, preserving class proportions in all three splits. The split is performed before any normalization.

2. **`fit_and_apply_scaler(X_train, X_val, X_test, save_path)`** — Fits `StandardScaler` exclusively on `X_train`, transforms `X_val` and `X_test` using the fitted parameters, and serializes the scaler to disk. The scaler is co-located with the model weights so the identical transform can be replicated at inference time.

3. **Test set sealing** — `X_test.npy` and `y_test.npy` (or `Xtest_*.npy` for P2/P4) are persisted to disk immediately after the split. They are never accessed during hyperparameter optimization; the Optuna study uses `X_val` exclusively.

4. **Augmentation gating** — Data augmentation (E5) is applied to `X_train` only, after the split, so augmented samples never appear in validation or test partitions.

### 9.3 Acoustic-Language Decoupling

The decoupled two-stage inference architecture is motivated by the structure of the target vocabulary:

- **Stage 1 — Acoustic Decoder (EEGNet):** Maps a single EEG epoch to a 19-dimensional probability distribution over syllable classes. This stage is paradigm-specific (P1, P2, or P4 model may be loaded).
- **Stage 2 — Language Assembler (WordAssembler / LogisticRegression):** Receives the full 19-dim probability vector from Slot 1 and the full 19-dim probability vector from Slot 2, concatenates them into a 38-dimensional feature vector, and applies a trained LogisticRegression classifier to produce a 10-dimensional word-class probability distribution.

The critical advantage of using the full probability vector (rather than a hard argmax) is that the assembler can leverage uncertainty information. For the ambiguous syllable `SA` (shared by *Sakit* and *Sayang*), the assembler can inspect the relative probabilities of `KIT` vs. `YANG` in Slot 2 to resolve the ambiguity, even when neither prediction is maximally confident.

The `WordAssembler` class in `logreg_model.py` carries an `_is_loaded` flag that is set to `True` only after a successful `load_model()` call. The inference gate in `main.py` checks this flag before invoking `predict()`, preventing silent random-output fallback when the assembler is instantiated but no trained model file exists on disk.

### 9.4 Statistical Evaluation Pipeline

Performance comparisons follow the standard non-parametric pipeline for multi-classifier, multi-dataset experimental benchmarks:

1. **Friedman omnibus test** — Tests whether at least one experiment configuration differs significantly from the others across subjects (paired non-parametric ANOVA equivalent).
2. **Wilcoxon signed-rank test** — Pairwise comparison of E0 baseline against the best-performing alternative configuration.
3. **Nemenyi post-hoc test** — All pairwise comparisons across configurations with familywise error rate control.
4. **Holm-Bonferroni correction** — Applied to the Wilcoxon p-values to control the familywise error rate across all comparisons.
5. **Effect sizes** — Rank-biserial correlation *r* (non-parametric) and Cohen's *d* (parametric approximation).
6. **Bootstrap 95% confidence intervals** — 10,000-iteration bootstrap on mean accuracy per condition.

### 9.5 SHAP GradientExplainer

`explain_model.py` uses `shap.GradientExplainer` to compute channel-level feature attributions. The background distribution uses `min(50, len(X) // 2)` randomly sampled training examples, which represents a statistically robust sample for gradient-based attribution (minimum 50 or half the available training data, whichever is smaller). SHAP values are aggregated across the time axis to produce a per-channel importance ranking, enabling the Frontal Dominance Ratio (FDR) analysis in the Q1 notebook.

---

## 10. Installation and Environment Setup

### Prerequisites

- Python 3.10 or later (3.11 recommended)
- Node.js 18 LTS or later (for the frontend)
- Git
- (Optional) NVIDIA GPU with CUDA 12.x and cuDNN 8.x for accelerated EEGNet training

### Step 1 — Clone the Repository

```bash
git clone <repository-url> neurandiar-bci
cd neurandiar-bci
```

### Step 2 — Backend Python Environment

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install all dependencies (pinned versions)
pip install -r requirements.txt
```

> **Note on TensorFlow and GPU:** `requirements.txt` pins `tensorflow==2.21.0`. For GPU acceleration, ensure your CUDA and cuDNN versions are compatible. CPU-only inference is fully supported without any additional configuration.

### Step 3 — Environment Variables

Create a `.env` file in `backend/` with your Emotiv Cortex API credentials. These are only required for live EEG acquisition; the inference API and training scripts run without them.

```bash
# backend/.env
EMOTIV_CLIENT_ID=your_client_id_here
EMOTIV_CLIENT_SECRET=your_client_secret_here
```

### Step 4 — Frontend Node Environment

```bash
cd frontend
npm install --legacy-peer-deps
```

> The `--legacy-peer-deps` flag is required due to React 19's peer dependency resolution with certain Radix UI packages.

Create a `frontend/.env.local` file to configure the API connection:

```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_WS_URL=ws://127.0.0.1:8000
```

---

## 11. Usage Guide for Future Researchers

### 11.1 System Integrity Check (Always Run First)

Before executing any training or inference workflow, run the diagnostics script from the repository root to verify the integrity of all model artefacts, test sets, raw data, and the MLflow database:

```bash
# From repository root
python run_system_diagnostics.py
```

The script performs 8 sequential checks:

| Check | Scope | What is Verified |
|-------|-------|-----------------|
| 1 | P1_Global | `.h5` model, `.pkl` scaler, `X_test.npy`, `y_test.npy` for all 8 experiments |
| 2 | P2_EEGNet | `.h5`, `.pkl`, `Xtest_{exp}_{subj}.npy`, `ytest_{exp}_{subj}.npy` for all 96 configurations |
| 3 | P3_SVM | `.pkl` model and scaler for all 480 SVM configurations (5 feature groups × 8 exp × 12 subj) |
| 4 | Test sets | Shape integrity of P1 `X_test / y_test` pairs (sample count match) |
| 5 | Raw data | CSV recordings and experiment logs for all 12 subjects |
| 6 | MLflow DB | SQLite database existence, readability, run count |
| 7 | Syntax | AST dry-run across all `backend/src/**/*.py` files |
| 8 | Inference log | `backend/logs/inference_history.csv` presence |

Exit code `0` = all checks passed. Exit code `1` = one or more FAIL conditions.

### 11.2 Starting the Backend API

```bash
cd backend
venv\Scripts\activate  # (if not already activated)

# Start the FastAPI inference server
python -m src.api.main

# The server will initialize at http://127.0.0.1:8000
# WebSocket telemetry : ws://127.0.0.1:8000/ws/telemetry
# WebSocket inference : ws://127.0.0.1:8000/ws/inference
# API documentation   : http://127.0.0.1:8000/docs
```

### 11.3 Starting the Frontend Dashboard

```bash
cd frontend
npm run dev

# Dashboard available at http://localhost:3000
```

The dashboard connects automatically to the backend API. Ensure the backend server is running before opening the dashboard. WebSocket connections in `LiveSession` and `MonitorControl` include a 3-second auto-reconnect so temporary disconnects are recovered without a page reload.

### 11.4 Running the P1 Global Training Pipeline

> **Warning:** This retrains all 8 P1 experiments from scratch. Training time is approximately 2–6 hours per experiment depending on hardware. Existing model artefacts will be overwritten.

```bash
cd backend/src/models

# Train all 8 experiments sequentially (E0 → E7)
python run_master_experiments.py

# Train a single experiment
python -c "
from run_master_experiments import execute_experiment
execute_experiment(
    exp_id='E0_Baseline',
    processor_params={'band': 'broadband', 'apply_ica': False, 'target_fs': 256},
    n_trials_optuna=5,
    max_epochs=300
)
"
```

### 11.5 Running the P2 Subject-Dependent EEGNet Grid

```bash
cd backend/src/models

# Train all 8 experiments × 12 subjects (96 models)
python run_subject_dependent.py

# Train a single experiment
python run_subject_dependent.py --exp E0_Baseline

# Train a single subject
python run_subject_dependent.py --subj S1
```

### 11.6 Running the P3 Classical SVM Feature Ablation Grid

```bash
cd backend/src/models

# Train all 5 feature groups × 8 experiments × 12 subjects (480 models)
python run_e8_classical.py
```

### 11.7 Running P4 Transfer Learning

P4 fine-tunes the P1 champion model on each subject's data. The base model must exist at `models/weights/P1_Global/{champion_exp}/eegnet_trained_{champion_exp}.h5` before P4 can run.

```bash
cd backend/src/models

# Fine-tune all subjects on all experiments using the default E0_Baseline champion
python run_p4_transfer_learning.py

# Specify a different champion model
python run_p4_transfer_learning.py --champion E2_Resampling_512Hz

# Fine-tune on a specific experiment (with validation)
python run_p4_transfer_learning.py --exp E0_Baseline E1_ICA_Filtering

# Fine-tune specific subjects only
python run_p4_transfer_learning.py --subj S1 S2 S3

# Reduce epoch budget for a quick test
python run_p4_transfer_learning.py --epochs 10
```

### 11.8 Replicating the Q1 Journal Analysis (Jupyter Notebook)

The master analysis notebook `notebooks/BCI_Master_Journal_Q1_Final.ipynb` reproduces all statistical figures, tables, and interpretability analyses from the Q1 journal submission.

**Setup:**

```bash
# Install Jupyter in the backend virtual environment (already present via requirements.txt)
cd backend
venv\Scripts\activate

# Launch Jupyter from the notebooks directory (IMPORTANT: CWD must be notebooks/)
cd ../notebooks
jupyter notebook BCI_Master_Journal_Q1_Final.ipynb
```

> **Critical:** The notebook must be launched with `notebooks/` as the working directory. All relative paths (figure output to `outputs/`, CSV export to `reports/data_export_claude/`) are anchored to this directory. The `EXPORT_DIR` cell uses `os.getcwd()` as an explicit anchor, but the Jupyter server CWD must still match.

**Notebook execution order:**

The notebook is divided into 15 chapters with a strict dependency order. Chapters 1–3 load and validate all artefacts; Chapters 4–13 compute statistics and generate figures; Chapters 14–15 produce the final export CSVs. Run cells sequentially from top to bottom without skipping.

| Chapter | Topic |
|---------|-------|
| 1 | Environment validation and artefact inventory |
| 2 | P1 Global accuracy per experiment (E0–E7) |
| 3 | P2 Subject-Dependent accuracy per experiment |
| 4 | P3 SVM feature group ablation |
| 5 | Cross-paradigm comparison (P1 vs. P2 vs. P3) |
| 6 | Best experiment identification per paradigm |
| 7 | Generalization analysis: subject-dependent vs. global |
| 8 | Subject variability analysis |
| 9–10 | Confusion matrix analysis (syllable and word level) |
| 11 | Friedman + Nemenyi + Holm-Bonferroni statistical tests |
| 12 | Effect sizes and bootstrap confidence intervals |
| 13 | SHAP channel importance + Frontal Dominance Ratio |
| 14 | Critical Difference diagram |
| 15 | CSV export for archival and reproducibility |

**Expected outputs:**

- PNG figures → `notebooks/outputs/` (approximately 20 figures)
- CSV exports → `notebooks/reports/data_export_claude/` (artefact manifest in `INDEX.json`)
- Cells referencing `backend/reports/` (deprecated path) will print a `[WARNING]` and skip gracefully without aborting execution.

---

## 12. Implementation Status

| Component | Status | Detail |
|-----------|--------|--------|
| EEG acquisition protocol (overt + imagined) | Complete | Pygame + Cortex API; LSL synchronization |
| Signal preprocessing (E0–E7, all conditions) | Complete | Butterworth, FastICA, resampling, ERP windowing, channel selection, augmentation |
| EEGNet-8,2 training — P1 Global | Complete | Optuna HPO · MLflow tracking · 8 experiments · 300 epochs max |
| EEGNet-8,2 training — P2 Subject-Dependent | Complete | 96 models · per-subject split · anti-leakage protocol |
| SVM feature ablation — P3 | Complete | 480 models · 5 feature groups · Optuna C/γ search |
| Word assembler (LogisticRegression) | Complete | 38-dim input · `_is_loaded` inference guard |
| P4 Transfer Learning (architecture) | Complete | Champion model loading · layer freeze via `isinstance()` · fine-tune loop |
| P4 Transfer Learning (evaluation) | Not yet run | Artefacts absent; future research item |
| Statistical evaluation notebook | Complete | Friedman · Nemenyi · Holm-Bonferroni · bootstrap CI · effect sizes |
| SHAP interpretability | Complete | GradientExplainer · min(50, n//2) background samples |
| System diagnostics script | Complete | 8 checks · P1/P2/P3/P4 artefact verification · AST syntax dry-run |
| FastAPI inference server | Simulated | WebSocket streams use mock data; real EEG pipeline is architecturally stubbed |
| Clinical dashboard (Next.js) | Complete (UI) | Full UI with WebSocket auto-reconnect; inference result is mocked |
| Emotiv Cortex live data ingestion | Not implemented | `cortex_client.py` is a stub requiring active Cortex SDK session |
| LLM sentence refinement | Not implemented | `llm_agent.py` is an empty placeholder — see Section 13 |
| `/api/metrics` endpoint | Implemented (partial) | Reads MLflow SQLite and inference CSV; latency metrics are proxy estimates |
| Real-time EEGNet inference | Not implemented | Inference pipeline uses random word selection for UI demonstration |

---

## 13. Future Work and Research Roadmap

The following items are explicitly deferred from the current thesis scope and are documented here for future researchers who extend this system.

| Priority | Item | Motivation |
|----------|------|-----------|
| P0 | **P4 Transfer Learning evaluation** | Architecture is complete; a full grid run and statistical comparison against P1/P2 is needed |
| P0 | **Replace mock inference with real EEGNet pipeline** | Wire `SignalProcessor → EEGNet → WordAssembler` into `/ws/inference` to enable genuine live decoding |
| P1 | **Leave-One-Subject-Out (LOSO) cross-validation for P1** | Current P1 uses a single 70/15/15 random split; LOSO over 12 subjects provides a community-standard generalization estimate for small-N BCI studies |
| P1 | **Per-subject z-scoring before pooling in P1** | Current P1 fits a single global StandardScaler; inter-subject amplitude variability creates implicit domain shift not accounted for in the current design |
| P1 | **Emotiv Cortex live data ingestion** | `cortex_client.py` requires a working Emotiv Cortex API subscription and device; implement the EEG data subscription loop and feed into the inference WebSocket |
| P2 | **LLM sentence refinement agent** | Implement `llm_agent.py` using the Anthropic Claude API (or a local model) to convert the decoded word sequence into grammatical Indonesian for clinical output |
| P2 | **Expand SHAP analysis** | Increase background and test sample count to ≥ 50 per class for statistically reliable channel importance rankings |
| P3 | **Temporal block-based splitting** | Replace random trial splitting with block-based splitting to prevent temporal autocorrelation leakage within EEG recording sessions |
| P3 | **Within-paradigm statistical testing for P1** | The Wilcoxon signed-rank test requires per-subject samples; P1 trains a single pooled model per experiment, so standard pairwise non-parametric tests are not directly applicable — a permutation-based approach is needed |
| P3 | **WebSocket authentication** | Add token-based auth to `/ws/inference` and `/ws/telemetry` before any networked deployment |

---

## 14. Research Acknowledgements

| Field | Detail |
|-------|--------|
| **Author** | Andiar Rinanda |
| **Institution** | Institut Teknologi Sepuluh Nopember (ITS), Department of Informatics |
| **Research type** | Undergraduate thesis (Skripsi) |
| **Hardware** | Emotiv EPOC X (14-channel EEG, 256 Hz, Bluetooth 5.0) |
| **Dataset** | 12 subjects · 200 trials/subject · 19-class syllable taxonomy · 10 target words |
| **Clinical motivation** | Communication assistance for individuals with severe motor disabilities (ALS, locked-in syndrome) |
| **Reference architecture** | Lawhern et al. (2018). *EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces.* Journal of Neural Engineering, 15(5). |
| **Statistical methodology** | Demšar (2006). *Statistical comparisons of classifiers over multiple data sets.* JMLR, 7, 1–30. |

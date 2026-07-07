import os
import sys
import time
import asyncio
import sqlite3
import csv
import glob
import random
from datetime import datetime

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Internal module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline.offline_trial_reader import OfflineTrialReader
from pipeline.svm_champion import SVMChampion
from pipeline.sentence_refiner import refine_sentence_rule_based
from models.transfer_learning import calibrate_new_user
from models.logreg_model import WordAssembler
from config import TRIALS_PER_SUBJECT, RAW_DATA_DIR

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEIGHTS_DIR   = os.path.join(BASE_DIR, "models", "weights")
LOGS_DIR      = os.path.join(BASE_DIR, "logs")
MLFLOW_DB     = os.path.join(LOGS_DIR, "mlflow", "mlruns.db")
HISTORY_FILE  = os.path.join(LOGS_DIR, "inference_history.csv")
LATENCY_FILE  = os.path.join(LOGS_DIR, "latency_history.csv")
RAW_LOGS_DIR  = os.path.join(BASE_DIR, "dataset", "raw", "logs")

os.makedirs(LOGS_DIR, exist_ok=True)

# Initialise inference history CSV header if the file does not yet exist
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, mode="w", newline="") as f:
        csv.writer(f).writerow(
            ["timestamp", "subject_id", "raw_word", "final_sentence", "confidence"]
        )

# Initialise latency history CSV header if the file does not yet exist
LATENCY_FIELDS = [
    "timestamp", "subject_id", "trial_index",
    "signal_read_filter_ms", "svm_inference_slot1_ms", "svm_inference_slot2_ms",
    "word_assembly_ms", "refinement_ms", "total_ms",
]
if not os.path.exists(LATENCY_FILE):
    with open(LATENCY_FILE, mode="w", newline="") as f:
        csv.writer(f).writerow(LATENCY_FIELDS)

# ---------------------------------------------------------------------------
# CHAMPION MODEL CONFIGURATION
# ---------------------------------------------------------------------------
# Champion resmi: SVM, Paradigma 3 (feature ablation), subjek S3,
# konfigurasi E5 (Data Augmentation), grup fitur Barlow.
# Akurasi uji 18,10%, cakupan 18/19 kelas suku kata (lihat notebook analisis).
# Model ini bersifat subject-dependent (dilatih hanya dari data S3), sehingga
# demo inferensi online dikunci hanya untuk subjek S3.
CHAMPION_PARADIGM   = "P3_SVM"
CHAMPION_EXP        = "E5_Data_Augmentation"
CHAMPION_SUBJECT    = "S3"
CHAMPION_FEAT_GROUP = "barlow"
CHAMPION_DIR        = os.path.join(WEIGHTS_DIR, CHAMPION_PARADIGM, CHAMPION_EXP)

CHAMPION_MODEL_PATH = os.path.join(
    CHAMPION_DIR, f"SVM_{CHAMPION_FEAT_GROUP}_{CHAMPION_EXP}_{CHAMPION_SUBJECT}.pkl"
)
CHAMPION_SCALER_PATH = os.path.join(
    CHAMPION_DIR, f"scaler_SVM_{CHAMPION_FEAT_GROUP}_{CHAMPION_EXP}_{CHAMPION_SUBJECT}.pkl"
)

# Parameter preprocessing SIS persis sama dengan processor_params resep E5
# di EXPERIMENT_RECIPES (models/run_subject_dependent.py). Nilainya dituliskan
# ulang di sini (bukan import langsung) agar proses API produksi tidak perlu
# memuat dependensi berat khusus training (mlflow, TensorFlow/EEGNet).
E5_PROCESSOR_PARAMS = {"band": "broadband", "apply_ica": False, "target_fs": 256}

# Word assembler untuk demo dilatih KHUSUS dari trial subjek S3 saja
# (train_word_assembler_s3.py), bukan varian pooled 12-subjek
# (train_word_assembler.py) — champion SVM subject-dependent, sehingga
# assembler yang representatif untuk demo juga harus konsisten dilatih
# hanya dari sinyal S3. Model pooled tetap ada di disk terpisah sebagai
# bukti pendukung diskusi keterbatasan generalisasi lintas-subjek, dan
# TIDAK dimuat di sini.
WORD_ASSEMBLER_FILENAME = (
    f"logreg_assembler_svm_S3only_{CHAMPION_FEAT_GROUP}_{CHAMPION_EXP}.pkl"
)

# ---------------------------------------------------------------------------
# APPLICATION
# ---------------------------------------------------------------------------
app = FastAPI(title="Neurandiar BCI Production Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# REQUEST SCHEMAS
# ---------------------------------------------------------------------------
class LogPayload(BaseModel):
    subject: str
    raw_word: str
    final_sentence: str
    confidence: float

class CalibrationPayload(BaseModel):
    subject_id: str
    eeg_data: list
    labels: list
    base_model_id: str

# ---------------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------------
# Populated at startup
svm_champion = None   # SVMChampion instance (feature extraction + SVM predict_proba)
assembler    = None   # WordAssembler instance (LogReg: 2x19-dim probs -> word)
trial_reader = None   # OfflineTrialReader instance (real raw-data epoch extraction)

# ---------------------------------------------------------------------------
# STARTUP — Load champion model
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global assembler, svm_champion, trial_reader

    print("[INFO] Neurandiar BCI backend initialising...")

    # ---- SVM champion model (feature extraction is owned by SVMChampion) ----
    if os.path.exists(CHAMPION_MODEL_PATH) and os.path.exists(CHAMPION_SCALER_PATH):
        try:
            svm_champion = SVMChampion(
                model_path=CHAMPION_MODEL_PATH,
                scaler_path=CHAMPION_SCALER_PATH,
                feat_group=CHAMPION_FEAT_GROUP,
                fs=E5_PROCESSOR_PARAMS["target_fs"],
            )
            print(
                f"[INFO] Model champion berhasil dimuat: "
                f"{CHAMPION_PARADIGM}/{CHAMPION_EXP}/{CHAMPION_SUBJECT}/{CHAMPION_FEAT_GROUP}."
            )
        except Exception as e:
            print(f"[WARNING] Gagal memuat model champion SVM: {e}")
    else:
        print(f"[WARNING] Artefak model champion tidak ditemukan di: {CHAMPION_MODEL_PATH}")

    # ---- LogReg word assembler (trained on real SVM champion outputs) ----
    assembler = WordAssembler(
        exp_id=CHAMPION_EXP, pilar=CHAMPION_PARADIGM, filename=WORD_ASSEMBLER_FILENAME
    )
    if os.path.exists(assembler.model_path):
        try:
            assembler.load_model()
            print(f"[INFO] Word assembler berhasil dimuat: {os.path.basename(assembler.model_path)}")
        except Exception as e:
            print(f"[WARNING] Gagal memuat word assembler: {e}")
    else:
        print(
            f"[WARNING] Word assembler tidak ditemukan di: {assembler.model_path}. "
            f"Jalankan models/train_word_assembler.py terlebih dahulu."
        )

    # ---- Offline trial reader (real raw-data epoch extraction) ----
    trial_reader = OfflineTrialReader(RAW_DATA_DIR, E5_PROCESSOR_PARAMS)
    try:
        trial_reader._load_subject(CHAMPION_SUBJECT)
        print(f"[INFO] Data mentah subjek {CHAMPION_SUBJECT} berhasil dimuat dan difilter.")
    except Exception as e:
        print(f"[WARNING] Gagal memuat data mentah subjek {CHAMPION_SUBJECT}: {e}")

    print("[INFO] Server ready. Awaiting frontend connection.")

# ---------------------------------------------------------------------------
# LATENCY LOGGING HELPER
# ---------------------------------------------------------------------------
def _log_latency(subject_id, trial_index, timings):
    """Persist one real per-inference latency measurement to disk (Masalah 5)."""
    try:
        with open(LATENCY_FILE, mode="a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                subject_id,
                trial_index,
                timings.get("signal_read_filter_ms", ""),
                timings.get("svm_inference_slot1_ms", ""),
                timings.get("svm_inference_slot2_ms", ""),
                timings.get("word_assembly_ms", ""),
                timings.get("refinement_ms", ""),
                timings.get("total_ms", ""),
            ])
    except Exception as e:
        print(f"[WARNING] Gagal menyimpan log latensi: {e}")

# ---------------------------------------------------------------------------
# REST ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/api/logs")
async def get_inference_logs():
    """Return the full inference history log as a JSON list."""
    if not os.path.exists(HISTORY_FILE):
        return {"status": "success", "data": []}
    try:
        with open(HISTORY_FILE, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return {"status": "success", "data": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/logs")
async def save_inference_log(payload: LogPayload):
    """Persist an inference result from the frontend to the history log."""
    try:
        with open(HISTORY_FILE, mode="a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                payload.subject,
                payload.raw_word,
                payload.final_sentence,
                payload.confidence,
            ])
        print(f"[INFO] Log saved: {payload.subject} -> {payload.final_sentence}")
        return {"status": "success", "message": "Log saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics")
async def get_metrics():
    """
    Return a consolidated metrics payload for the Evaluation dashboard.

    Reads from:
      - MLflow SQLite database (model registry, Optuna trials)
      - latency_history.csv (real per-inference latency measurements)
      - backend/logs/logs_backup/ (raw experiment log preview)
      - P1/P2/P3 model weight directories (dataset metadata inferred from
        the number of .npy test files)
    """
    # ---- 1. Overview latency from REAL per-inference measurements --------
    median_latency  = 0.0
    p95_latency     = 0.0
    latency_is_proxy = True
    latency_note = "Belum ada siklus inferensi tercatat — jalankan Live Session untuk mengukur latensi nyata."

    if os.path.exists(LATENCY_FILE):
        try:
            with open(LATENCY_FILE, mode="r") as f:
                rows = list(csv.DictReader(f))
            totals = [float(r["total_ms"]) for r in rows if r.get("total_ms")]
            if totals:
                totals_sorted = sorted(totals)
                mid = len(totals_sorted) // 2
                median_latency = round(totals_sorted[mid], 1)
                p95_idx = int(len(totals_sorted) * 0.95)
                p95_latency = round(totals_sorted[min(p95_idx, len(totals_sorted) - 1)], 1)
                latency_is_proxy = False
                latency_note = f"Diukur nyata (time.perf_counter()) dari {len(totals)} siklus inferensi."
        except Exception:
            pass

    overview = {
        "median_latency":   median_latency,
        "p95_latency":      p95_latency,
        "active_model":     f"{CHAMPION_PARADIGM}/{CHAMPION_EXP}/{CHAMPION_SUBJECT}/{CHAMPION_FEAT_GROUP}",
        "latency_is_proxy": latency_is_proxy,
        "latency_note":     latency_note,
    }

    # ---- 2–3–6. MLflow queries — single shared connection ---------------
    mlflow_registry = []
    optuna_trials   = []
    training_curves = []

    if os.path.exists(MLFLOW_DB):
        conn = None
        try:
            conn = sqlite3.connect(MLFLOW_DB)
            cur  = conn.cursor()

            # 2. Model registry (most recent 20 runs)
            cur.execute("""
                SELECT r.run_uuid, r.name, r.status,
                       MAX(CASE WHEN m.key='best_val_accuracy' THEN m.value END) AS f1,
                       MAX(CASE WHEN m.key='test_loss' OR m.key='val_loss' THEN m.value END) AS loss
                FROM runs r
                LEFT JOIN latest_metrics m ON r.run_uuid = m.run_uuid
                GROUP BY r.run_uuid
                ORDER BY r.start_time DESC
                LIMIT 20
            """)
            for uuid_, name_, status_, f1_, loss_ in cur.fetchall():
                mlflow_registry.append({
                    "version":  name_ or uuid_[:8],
                    "status":   status_ or "FINISHED",
                    "f1_score": round(float(f1_)   if f1_   is not None else 0.0, 4),
                    "loss":     round(float(loss_)  if loss_ is not None else 0.0, 4),
                })

            # 3. Optuna hyperparameter trials (most recent 10)
            cur.execute("""
                SELECT r.run_uuid,
                       MAX(CASE WHEN p.key='dropout_rate' THEN p.value END) AS dropout,
                       MAX(CASE WHEN m.key='val_accuracy'  THEN m.value END) AS val_acc
                FROM runs r
                LEFT JOIN params         p ON r.run_uuid = p.run_uuid
                LEFT JOIN latest_metrics m ON r.run_uuid = m.run_uuid
                WHERE r.name LIKE '%trial%' OR r.tags LIKE '%optuna%'
                GROUP BY r.run_uuid
                ORDER BY r.start_time DESC
                LIMIT 10
            """)
            for i, (_, dropout_, val_acc_) in enumerate(cur.fetchall()):
                optuna_trials.append({
                    "trial":   i + 1,
                    "dropout": round(float(dropout_), 3) if dropout_ else "N/A",
                    "valAcc":  f"{float(val_acc_)*100:.2f}%" if val_acc_ else "N/A",
                })

            # 6. Training curves from the most recent completed run
            cur.execute("""
                SELECT run_uuid FROM runs
                WHERE status='FINISHED'
                ORDER BY start_time DESC LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                run_id = row[0]
                cur.execute(
                    "SELECT step, value FROM metrics WHERE run_uuid=? AND key='val_accuracy' ORDER BY step",
                    (run_id,),
                )
                acc_map = {r[0]: r[1] for r in cur.fetchall()}
                cur.execute(
                    "SELECT step, value FROM metrics WHERE run_uuid=? AND key='val_loss' ORDER BY step",
                    (run_id,),
                )
                loss_map = {r[0]: r[1] for r in cur.fetchall()}
                for step in sorted(set(acc_map) | set(loss_map)):
                    training_curves.append({
                        "epoch": step,
                        "acc":   round(float(acc_map.get(step, 0)), 4),
                        "loss":  round(float(loss_map.get(step, 0)), 4),
                    })

        except Exception as e:
            print(f"[WARNING] MLflow query error: {e}")
        finally:
            if conn:
                conn.close()

    # Fall-back registry entry so the dashboard is never blank
    if not mlflow_registry:
        mlflow_registry = [{
            "version":  f"{CHAMPION_PARADIGM}/{CHAMPION_EXP}/{CHAMPION_SUBJECT}",
            "status":   "FINISHED",
            "f1_score": 0.0,
            "loss":     0.0,
        }]

    # ---- 4. Dataset metadata (from P2 test .npy files as proxy) --------
    dataset_meta = []
    subjects = [f"S{i}" for i in range(1, 13)]
    p2_e0_dir = os.path.join(WEIGHTS_DIR, "P2_EEGNet", "E0_Baseline")
    for subj in subjects:
        x_path = os.path.join(p2_e0_dir, f"Xtest_E0_Baseline_{subj}.npy")
        clean_epochs = 0
        if os.path.exists(x_path):
            try:
                arr = np.load(x_path)
                clean_epochs = len(arr)
            except Exception:
                pass
        dataset_meta.append({
            "subject":     subj,
            "trials":      TRIALS_PER_SUBJECT,
            "rejected":    max(0, TRIALS_PER_SUBJECT - clean_epochs * 5) if clean_epochs else "N/A",
            "cleanEpochs": clean_epochs,
        })

    # ---- 5. Raw experiment log preview ----------------------------------
    raw_logs_preview = "No experiment logs found."
    logs_dir = os.path.join(LOGS_DIR, "logs_backup")
    if not os.path.isdir(logs_dir):
        logs_dir = RAW_LOGS_DIR
    log_files = sorted(glob.glob(os.path.join(logs_dir, "S*_experiment_log.txt")))
    if log_files:
        try:
            with open(log_files[0], "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            raw_logs_preview = "".join(lines[:30])
        except Exception:
            pass

    return {
        "status":          "success",
        "overview":        overview,
        "mlflow_registry": mlflow_registry,
        "dataset_meta":    dataset_meta,
        "raw_logs_preview": raw_logs_preview,
        "optuna_trials":   optuna_trials,
        "training_curves": training_curves,
    }


@app.post("/api/v1/calibrate")
async def calibrate(payload: CalibrationPayload):
    """Calibrate the champion SVM model to a new user via fast retraining."""
    try:
        save_dir = os.path.join(WEIGHTS_DIR, "P4_TransferLearning", "Calibrated")
        os.makedirs(save_dir, exist_ok=True)

        X_new = np.array(payload.eeg_data, dtype=np.float32)
        y_new = np.array(payload.labels, dtype=np.int32)

        save_path, model_type = calibrate_new_user(
            base_model_path=CHAMPION_MODEL_PATH,
            X_new_3d=X_new,
            y_new=y_new,
            new_subject_id=payload.subject_id,
            save_dir=save_dir,
            champion_type="svm",
            feat_group=CHAMPION_FEAT_GROUP,
        )
        return {"status": "success", "message": "Kalibrasi selesai.", "model_path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# WEBSOCKET — EEG TELEMETRY
# ---------------------------------------------------------------------------
@app.websocket("/ws/telemetry")
async def telemetry_endpoint(websocket: WebSocket):
    """Stream EEG telemetry data to the monitor dashboard."""
    await websocket.accept()
    print("[INFO] Telemetry dashboard connected.")
    channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    try:
        while True:
            eeg_data = {ch: round(random.uniform(-50, 50), 2) for ch in channels}
            cq_data  = {
                ch: random.choices(["good", "fair", "poor"], weights=[80, 15, 5])[0]
                for ch in channels
            }
            await websocket.send_json({
                "eeg": eeg_data,
                "cq":  cq_data,
                "bandpower": {
                    "Theta": round(random.uniform(10, 30), 1),
                    "Alpha": round(random.uniform(20, 60), 1),
                    "Beta":  round(random.uniform(15, 40), 1),
                    "Gamma": round(random.uniform(5,  20), 1),
                },
                "mental_state": {
                    "Stress":     round(random.uniform(20, 35), 1),
                    "Fatigue":    round(random.uniform(10, 25), 1),
                    "Focus":      round(random.uniform(70, 95), 1),
                    "Relaxation": round(random.uniform(60, 85), 1),
                },
            })
            await asyncio.sleep(0.1)  # 10 FPS
    except WebSocketDisconnect:
        print("[INFO] Telemetry dashboard disconnected.")


# ---------------------------------------------------------------------------
# WEBSOCKET — LIVE INFERENCE (Model Offline: real raw-data replay)
# ---------------------------------------------------------------------------
@app.websocket("/ws/inference")
async def inference_endpoint(websocket: WebSocket):
    """
    Step-by-step BCI inference pipeline over WebSocket, running on real EEG
    epochs replayed from stored raw recordings ("Model Offline" — no live
    LSL/Cortex streaming is involved).

    Message protocol (client → server):
        START_DECODE|{subject_id}   — trigger one inference cycle
        EMERGENCY_STOP              — abort the current sequence

    Message protocol (server → client):
        {"status": "processing", "step": N, "message": "..."}   — pipeline step N
        {"status": "error", "message": "..."}                   — request could not be served
        {"status": "success", "step": 5, "decoded_word": "MAKAN",
         "refined_sentence": "Saya ingin makan.", "confidence": 91.3,
         "ground_truth_word": "MAKAN", "trial_index": 42,
         "latency_ms": {...}}
    """
    await websocket.accept()
    print("[INFO] Live session connected.")
    try:
        while True:
            data = await websocket.receive_text()

            if data.startswith("START_DECODE"):
                parts      = data.split("|")
                subject_id = parts[1] if len(parts) > 1 else CHAMPION_SUBJECT
                print(f"[INFO] Inference requested by: {subject_id}")

                if svm_champion is None or assembler is None or not assembler._is_loaded:
                    await websocket.send_json({
                        "status": "error",
                        "message": "Model champion atau word assembler belum termuat di server.",
                    })
                    continue

                # Champion SVM is subject-dependent (dilatih hanya dari data S3).
                if subject_id != CHAMPION_SUBJECT:
                    await websocket.send_json({
                        "status": "error",
                        "message": (
                            f"Model champion bersifat subject-dependent dan hanya valid untuk "
                            f"subjek {CHAMPION_SUBJECT}. Subjek '{subject_id}' tidak didukung."
                        ),
                    })
                    continue

                timings = {}
                t_pipeline_start = time.perf_counter()

                # ---- Step 1: read real trial + bandpass filter + windowing ----
                await websocket.send_json({
                    "status": "processing", "step": 1,
                    "message": "Membaca epoch trial nyata dan menerapkan filter bandpass...",
                })
                t0 = time.perf_counter()
                try:
                    trial = trial_reader.read_trial(subject_id, trial_index=None)
                except Exception as e:
                    await websocket.send_json({
                        "status": "error",
                        "message": f"Gagal membaca data trial nyata: {e}",
                    })
                    continue
                timings["signal_read_filter_ms"] = round((time.perf_counter() - t0) * 1000, 2)

                # ---- Step 2: Barlow feature extraction + SVM inference (x2) ----
                await websocket.send_json({
                    "status": "processing", "step": 2,
                    "message": "Mengekstraksi fitur Barlow dan menjalankan inferensi SVM...",
                })
                t0 = time.perf_counter()
                prob_slot1 = svm_champion.predict_proba_full(trial["epoch_slot1"])
                t1 = time.perf_counter()
                prob_slot2 = svm_champion.predict_proba_full(trial["epoch_slot2"])
                t2 = time.perf_counter()
                timings["svm_inference_slot1_ms"] = round((t1 - t0) * 1000, 2)
                timings["svm_inference_slot2_ms"] = round((t2 - t1) * 1000, 2)

                # ---- Step 3: word assembly ----
                await websocket.send_json({
                    "status": "processing", "step": 3,
                    "message": "Merangkai kata dari probabilitas dua slot suku kata...",
                })
                t0 = time.perf_counter()
                decoded_word, assembler_confidence = assembler.assemble_word_with_confidence(
                    prob_slot1, prob_slot2
                )
                confidence = round(assembler_confidence * 100, 2)
                timings["word_assembly_ms"] = round((time.perf_counter() - t0) * 1000, 2)

                # ---- Step 4: rule-based sentence refinement ----
                await websocket.send_json({
                    "status": "processing", "step": 4,
                    "message": "Menyusun kalimat komunikatif (rule-based)...",
                })
                t0 = time.perf_counter()
                refined = refine_sentence_rule_based(decoded_word)
                timings["refinement_ms"] = round((time.perf_counter() - t0) * 1000, 2)

                timings["total_ms"] = round((time.perf_counter() - t_pipeline_start) * 1000, 2)
                _log_latency(subject_id, trial["trial_index"], timings)

                await websocket.send_json({
                    "status":            "success",
                    "step":              5,
                    "decoded_word":      decoded_word,
                    "refined_sentence":  refined,
                    "confidence":        confidence,
                    "ground_truth_word": trial["word"],
                    "trial_index":       trial["trial_index"],
                    "latency_ms":        timings,
                })
                print(
                    f"[INFO] Inference complete: {decoded_word} -> {refined} ({confidence}%) "
                    f"[ground truth: {trial['word']}, trial {trial['trial_index']}] "
                    f"total_latency={timings['total_ms']}ms"
                )

            elif data == "EMERGENCY_STOP":
                print("[INFO] Inference sequence terminated by user.")

    except WebSocketDisconnect:
        print("[INFO] Live session disconnected.")

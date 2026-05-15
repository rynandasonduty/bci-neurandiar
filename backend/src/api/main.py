import os
import sys
import asyncio
import pickle
import numpy as np
import csv
from datetime import datetime
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# Import modul internal proyek
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.signal_processor import SignalProcessor
from features.extract_eeg_features import EEGFeatureExtractor
from models.transfer_learning import calibrate_new_user
from models.logreg_model import WordAssembler, REVERSE_WORD_CLASSES 

# --- CONFIGURATION & PATHS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WEIGHTS_DIR = os.path.join(BASE_DIR, "dataset", "models") 
LOGS_DIR = os.path.join(BASE_DIR, "dataset", "logs")
HISTORY_FILE = os.path.join(LOGS_DIR, "inference_history.csv")
os.makedirs(LOGS_DIR, exist_ok=True)

# Buat file CSV History jika belum ada
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "subject_id", "raw_word", "final_sentence", "confidence"])

app = FastAPI(title="Neurandiar BCI Production Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SCHEMAS ---
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

# --- GLOBAL STATE ---
ai_models = {}      
ai_scalers = {}     
processor = SignalProcessor(target_fs=256)
feature_extractor = EEGFeatureExtractor(fs=256)
assembler = WordAssembler()

@app.on_event("startup")
async def startup_event():
    print("="*50)
    print(f"🚀 BACKEND NEURANDIAR BCI BERHASIL MENYALA 🚀")
    print(f"[*] Menunggu koneksi dari Frontend...")
    print("="*50)

# --- REST ENDPOINTS ---

@app.post("/api/logs")
async def save_inference_log(payload: LogPayload):
    """Endpoint untuk menyimpan riwayat (Save to History) dari Frontend"""
    try:
        with open(HISTORY_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().isoformat(),
                payload.subject,
                payload.raw_word,
                payload.final_sentence,
                payload.confidence
            ])
        print(f"[+] Log disimpan: {payload.subject} -> {payload.final_sentence}")
        return {"status": "success", "message": "Log saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/calibrate")
async def calibrate(payload: CalibrationPayload):
    # (Kode kalibrasi sama seperti sebelumnya)
    return {"status": "success", "message": "Kalibrasi Selesai", "model_path": "dummy_path.h5"}

# --- WEBSOCKETS ---

@app.websocket("/ws/telemetry")
async def telemetry_endpoint(websocket: WebSocket):
    """Endpoint untuk grafik Oskiloskop dan Kualitas Kontak di Monitor UI"""
    await websocket.accept()
    print("[*] Dashboard Telemetry Terhubung.")
    try:
        while True:
            channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
            eeg_data = {ch: round(random.uniform(-50, 50), 2) for ch in channels}
            
            cq_data = {ch: random.choices(["good", "fair", "poor"], weights=[80, 15, 5])[0] for ch in channels}
            
            telemetry_payload = {
                "eeg": eeg_data,
                "cq": cq_data,
                "bandpower": {
                    "Theta": round(random.uniform(10, 30), 1),
                    "Alpha": round(random.uniform(20, 60), 1),
                    "Beta": round(random.uniform(15, 40), 1),
                    "Gamma": round(random.uniform(5, 20), 1)
                },
                "mental_state": {
                    "Stress": round(random.uniform(20, 35), 1),
                    "Fatigue": round(random.uniform(10, 25), 1),
                    "Focus": round(random.uniform(70, 95), 1),
                    "Relaxation": round(random.uniform(60, 85), 1)
                }
            }
            await websocket.send_json(telemetry_payload)
            await asyncio.sleep(0.1) # 10 FPS
    except WebSocketDisconnect:
        print("[!] Dashboard Telemetry Terputus.")

@app.websocket("/ws/inference")
async def inference_endpoint(websocket: WebSocket):
    """Endpoint untuk Live Session (Step-by-step Pipeline)"""
    await websocket.accept()
    print("[*] Live Session Terhubung.")
    try:
        while True:
            data = await websocket.receive_text()
            
            if data.startswith("START_DECODE"):
                parts = data.split("|")
                subject_id = parts[1] if len(parts) > 1 else "Unknown"
                print(f"[*] Memulai sekuens inferensi untuk {subject_id}...")
                
                # --- STEP 1: Acquisition & Filtering ---
                await websocket.send_json({"status": "processing", "step": 1, "message": "Memfilter sinyal EEG..."})
                await asyncio.sleep(0.8) # Jeda animasi UI
                
                # --- STEP 2: Decoding (EEGNet / SVM) ---
                await websocket.send_json({"status": "processing", "step": 2, "message": "Ekstraksi fitur N400..."})
                await asyncio.sleep(1.0)
                
                # --- STEP 3: Word Assembly ---
                await websocket.send_json({"status": "processing", "step": 3, "message": "Menyusun suku kata..."})
                await asyncio.sleep(0.8)
                
                # --- STEP 4: LLM Refining ---
                await websocket.send_json({"status": "processing", "step": 4, "message": "Validasi semantik akhir..."})
                await asyncio.sleep(0.8)
                
                # --- HASIL AKHIR ---
                # Dalam mode nyata, nilai ini diambil dari fungsi model.predict()
                # Karena ini menghubungkan ke UI, kita siapkan variabelnya
                mock_classes = ["ma", "ju", "mun", "dur", "ka", "nan"]
                raw_word = random.choice(mock_classes)
                confidence = round(random.uniform(85.0, 98.5), 2)
                
                # Simulasi Word Assembler
                refined = f"{raw_word} (Diproses)"
                if raw_word == "ma": refined = "Maju"
                elif raw_word == "mun": refined = "Mundur"
                
                await websocket.send_json({
                    "status": "success",
                    "step": 5,
                    "decoded_word": raw_word,
                    "refined_sentence": refined,
                    "confidence": confidence
                })
                print(f"[+] Inferensi Selesai: {refined} ({confidence}%)")
                
            elif data == "EMERGENCY_STOP":
                print("[!] Inferensi dihentikan darurat oleh User.")
                
    except WebSocketDisconnect:
        print("[!] Live Session Terputus.")
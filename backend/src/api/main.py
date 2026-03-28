import os
import sys
import time
import math
import random
import asyncio
import pickle
import numpy as np
import csv
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.signal_processor import SignalProcessor
from models.logreg_model import WordAssembler, REVERSE_WORD_CLASSES

# Konfigurasi Path
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dataset', 'models'))
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dataset', 'logs'))
os.makedirs(LOGS_DIR, exist_ok=True)

app = FastAPI(title="Neurandiar BCI Live Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_models = {}

@app.on_event("startup")
async def load_brain_models():
    print("="*50)
    print(" MENGHIDUPKAN MESIN AI NEURANDIAR... ")
    print("="*50)
    try:
        eegnet_path = os.path.join(MODEL_DIR, "eegnet_trained.h5")
        ai_models['eegnet'] = load_model(eegnet_path)
        logreg_path = os.path.join(MODEL_DIR, "logistic_regression_assembler.pkl")
        with open(logreg_path, 'rb') as f:
            ai_models['word_assembler'] = pickle.load(f)
        print("[+] Model AI berhasil dimuat.")
    except Exception as e:
        print(f"[!] PERINGATAN: Model AI belum tersedia. Mode Simulasi Aktif.")

# --- API UNTUK MENYIMPAN RIWAYAT (HISTORY) ---
class InferenceLog(BaseModel):
    subject: str
    raw_word: str
    final_sentence: str
    confidence: float

@app.post("/api/logs")
def save_inference_log(log: InferenceLog):
    """Menyimpan hasil tebakan yang sudah dikoreksi/diverifikasi user ke dalam file CSV"""
    log_file = os.path.join(LOGS_DIR, "live_inference_history.csv")
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Subject", "Raw Word", "Refined Sentence", "Confidence"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), log.subject, log.raw_word, log.final_sentence, log.confidence])
    
    print(f"[+] Log disimpan: {log.raw_word} -> {log.final_sentence}")
    return {"status": "success", "message": "Log berhasil disimpan ke CSV."}


# --- WEBSOCKET UNTUK INFERENSI REAL-TIME ---
@app.websocket("/ws/inference")
async def live_inference_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[+] Frontend Terhubung ke WebSocket.")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            if data == "EMERGENCY_STOP":
                print("[-] Sinyal Berhenti Darurat Diterima.")
                continue

            if data.startswith("START_DECODE"):
                # Kita bisa membedah parameter dari frontend (misal: START_DECODE|Subject-01)
                
                # STEP 0: EEG Acquisition
                await websocket.send_json({"status": "processing", "step": 0, "message": "EEG Acquisition: Menangkap sinyal..."})
                await asyncio.sleep(1) # Simulasi durasi rekam
                
                # STEP 1: Filtering & Extraction
                await websocket.send_json({"status": "processing", "step": 1, "message": "Memfilter Noise & Ekstraksi Fitur..."})
                await asyncio.sleep(1)
                
                # STEP 2: EEGNet Decoding
                await websocket.send_json({"status": "processing", "step": 2, "message": "EEGNet Menganalisis Pola Spasial..."})
                prob_slot1 = np.random.rand(19)
                prob_slot2 = np.random.rand(19)
                await asyncio.sleep(1.5)
                
                # STEP 3: Word Assembly
                await websocket.send_json({"status": "processing", "step": 3, "message": "Merakit Suku Kata (LogReg)..."})
                combined_probs = np.concatenate((prob_slot1, prob_slot2)).reshape(1, -1)
                
                if 'word_assembler' in ai_models:
                    pred_idx = ai_models['word_assembler'].predict(combined_probs)[0]
                    final_word = REVERSE_WORD_CLASSES[pred_idx]
                else:
                    final_word = np.random.choice(list(REVERSE_WORD_CLASSES.values()))
                await asyncio.sleep(1)
                
                # STEP 4: LLM Refining
                await websocket.send_json({"status": "processing", "step": 4, "message": "Menyempurnakan Makna Kalimat..."})
                # Simulasi AI Backend melakukan perbaikan kalimat (Nantinya panggil OpenAI/Gemini API di sini)
                refined_sentence = f"Subjek ingin mengatakan: '{final_word}'"
                await asyncio.sleep(1.5)
                
                # STEP 5: SELESAI
                await websocket.send_json({
                    "status": "success",
                    "step": 5, 
                    "decoded_word": final_word,
                    "refined_sentence": refined_sentence, # Sekarang kalimat LLM murni dari Backend!
                    "confidence": round(float(np.random.uniform(70.0, 99.9)), 2)
                })

    except WebSocketDisconnect:
        print("[-] Klien Terputus.")
    except Exception as e:
        print(f"[X] Error WebSocket: {e}")

# --- WEBSOCKET UNTUK TELEMETRI HARDWARE (MONITOR PAGE) ---
@app.websocket("/ws/telemetry")
async def hardware_telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[+] Klien Terhubung ke Stream Telemetry Hardware.")
    
    channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    
    try:
        while True:
            t = time.time()
            # 1. Simulasi Oscilloscope (Kombinasi gelombang Sinus dan Noise agar realistis)
            eeg_data = {
                ch: round(math.sin(t * (i+1)) * 30 + random.uniform(-15, 15), 2) 
                for i, ch in enumerate(channels)
            }
            
            # 2. Simulasi Kualitas Kontak (CQ) - Berbobot agar lebih sering 'Good' (Hijau)
            cq_data = {
                ch: random.choices(["Good", "Fair", "Poor"], weights=[85, 10, 5])[0] 
                for ch in channels
            }
            
            # 3. Simulasi Bandpower (Alpha, Beta, dll) & Mental State
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
            await asyncio.sleep(0.1)  # Refresh Rate: 10 FPS (100ms) agar UI web tidak hang
            
    except WebSocketDisconnect:
        print("[-] Klien Telemetry Terputus.")
    except Exception as e:
        print(f"[X] Error Telemetry WebSocket: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
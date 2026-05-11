import os
import sys
import asyncio
import pickle
import numpy as np
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
# Sesuaikan import ini jika nama file/variabel Anda berbeda
from models.logreg_model import WordAssembler, REVERSE_WORD_CLASSES 

# --- CONFIGURATION & PATHS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WEIGHTS_DIR = os.path.join(BASE_DIR, "dataset", "models") # Disesuaikan dengan path asli Anda
LOGS_DIR = os.path.join(BASE_DIR, "dataset", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

app = FastAPI(title="Neurandiar BCI Production Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SCHEMAS ---
class PredictPayload(BaseModel):
    subject_id: str
    eeg_data: list  # Shape: [14 channels x Time Samples]
    champion_type: str = "eegnet" # "eegnet" atau "svm"
    feature_group: str = "all"    # Digunakan jika champion adalah SVM

class CalibrationPayload(BaseModel):
    subject_id: str
    eeg_data: list  # Shape: [Samples, 14 channels, Time]
    labels: list    # List of labels
    base_model_id: str # ID model P1/P2/P3 yang jadi basis

# --- GLOBAL STATE (MEMORY CACHING) ---
ai_models = {}      # Menyimpan bobot model .h5 atau .pkl
ai_scalers = {}     # Menyimpan scaler .pkl khusus untuk SVM
processor = SignalProcessor(target_fs=256)
feature_extractor = EEGFeatureExtractor(fs=256)
assembler = WordAssembler() # Inisialisasi Word Assembler

def load_user_model(subject_id, champion_type):
    """Fungsi cerdas untuk me-load model ke RAM jika belum ada."""
    cache_key = f"{champion_type}_{subject_id}"
    
    if cache_key not in ai_models:
        print(f"[*] Loading model {cache_key} ke Memory (RAM)...")
        if champion_type == "eegnet":
            model_path = os.path.join(WEIGHTS_DIR, f"calibrated_EEGNet_{subject_id}.h5")
            if not os.path.exists(model_path):
                print(f"[!] Model personal tidak ada. Fallback ke Global Champion EEGNet.")
                model_path = os.path.join(WEIGHTS_DIR, "ultimate_champion_eegnet.h5") # Sesuaikan nama
            ai_models[cache_key] = load_model(model_path, compile=False)
            
        elif champion_type == "svm":
            model_path = os.path.join(WEIGHTS_DIR, f"calibrated_SVM_{subject_id}.pkl")
            scaler_path = os.path.join(WEIGHTS_DIR, f"calibrated_scaler_{subject_id}.pkl")
            
            if not os.path.exists(model_path):
                print(f"[!] Model personal tidak ada. Fallback ke Global Champion SVM.")
                model_path = os.path.join(WEIGHTS_DIR, "ultimate_champion_svm.pkl")
                scaler_path = os.path.join(WEIGHTS_DIR, "ultimate_champion_scaler.pkl")
                
            with open(model_path, 'rb') as f:
                ai_models[cache_key] = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                ai_scalers[cache_key] = pickle.load(f)
                
    return cache_key

@app.on_event("startup")
async def startup_event():
    print("="*50)
    print(f"🚀 BACKEND NEURANDIAR BCI BERHASIL MENYALA 🚀")
    print(f"[*] Menunggu koneksi dari Frontend...")
    print("="*50)

# --- REST ENDPOINTS ---

@app.post("/api/v1/predict")
async def predict_speech(payload: PredictPayload):
    try:
        # 1. Pastikan model ada di RAM (Caching)
        cache_key = load_user_model(payload.subject_id, payload.champion_type)
        
        # 2. Preprocessing Sinyal (Raw -> Filtered)
        raw_signal = np.array(payload.eeg_data) # [14, Time]
        filtered_signal = processor.apply_filters(raw_signal.T).T 
        
        class_idx = 0
        confidence = 0.0

        # 3. Alur Prediksi Berdasarkan Tipe Champion
        if payload.champion_type == "eegnet":
            # Siapkan input 4D untuk CNN: (1, 14, T, 1)
            input_data = np.expand_dims(np.expand_dims(filtered_signal, axis=0), axis=-1)
            model = ai_models[cache_key]
            
            prediction_probs = model.predict(input_data, verbose=0)
            class_idx = int(np.argmax(prediction_probs))
            confidence = float(np.max(prediction_probs))
            
        elif payload.champion_type == "svm":
            # Siapkan input 3D untuk Extractor: (1, 14, T)
            input_data_3d = np.expand_dims(filtered_signal, axis=0)
            groups = None if payload.feature_group == 'all' else [payload.feature_group]
            
            # Ekstraksi Fitur -> 2D (1, Num_Features)
            features_2d = feature_extractor.transform(input_data_3d, groups=groups)
            features_2d = np.nan_to_num(features_2d, nan=0.0)
            
            # Scaling Fitur
            scaler = ai_scalers[cache_key]
            scaled_features = scaler.transform(features_2d)
            
            # Prediksi SVM
            svm_model = ai_models[cache_key]
            # Jika menggunakan sklearn SVM, panggil predict_proba jika di-enable, jika tidak fallback ke predict
            if hasattr(svm_model, "predict_proba"):
                probs = svm_model.predict_proba(scaled_features)
                class_idx = int(np.argmax(probs))
                confidence = float(np.max(probs))
            else:
                class_idx = int(svm_model.predict(scaled_features)[0])
                confidence = 1.0 # SVM standar tidak punya persentase probabilitas
        
        # 4. Dekode Index Kelas Menjadi Suku Kata (Word Assembler)
        # Gunakan dict REVERSE_WORD_CLASSES (misal {0: "ma", 1: "ju", ...})
        predicted_syllable = REVERSE_WORD_CLASSES.get(class_idx, "unknown")
        
        # Opsional: Memasukkan suku kata ke WordAssembler untuk menjadi kata utuh
        # current_word = assembler.add_syllable(predicted_syllable)
        
        return {
            "status": "success",
            "subject_id": payload.subject_id,
            "prediction_index": class_idx,
            "predicted_syllable": predicted_syllable,
            "confidence": round(confidence * 100, 2),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

@app.post("/api/v1/calibrate")
async def calibrate(payload: CalibrationPayload):
    try:
        X_new = np.array(payload.eeg_data)
        y_new = np.array(payload.labels)
        X_new = np.expand_dims(X_new, axis=-1)
        
        base_model_path = os.path.join(WEIGHTS_DIR, f"{payload.base_model_id}.h5")
        
        saved_path, m_type = calibrate_new_user(
            base_model_path=base_model_path,
            X_new_3d=X_new,
            y_new=y_new,
            new_subject_id=payload.subject_id,
            save_dir=WEIGHTS_DIR,
            champion_type="eegnet" 
        )
        
        # Hapus cache model lama agar sesi prediksi berikutnya memuat model yang baru dikalibrasi
        cache_key = f"eegnet_{payload.subject_id}"
        if cache_key in ai_models:
            del ai_models[cache_key]
            
        return {"status": "success", "message": "Kalibrasi Selesai", "model_path": saved_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- WEBSOCKET FOR TELEMETRY ---
@app.websocket("/ws/telemetry")
async def telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[*] Dashboard Telemetry Terhubung.")
    try:
        while True:
            # Simulasi data telemetry (Sesuai draf awal Anda)
            channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
            eeg_data = {ch: round(random.uniform(-50, 50), 2) for ch in channels}
            
            telemetry_payload = {
                "eeg": eeg_data,
                "mental_state": {
                    "Focus": round(random.uniform(70, 95), 1),
                    "Relaxation": round(random.uniform(60, 85), 1)
                }
            }
            await websocket.send_json(telemetry_payload)
            await asyncio.sleep(0.1) 
    except WebSocketDisconnect:
        print("[!] Dashboard Telemetry Terputus.")
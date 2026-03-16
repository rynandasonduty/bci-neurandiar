import os
import sys
import time
import asyncio
import pickle
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# Menambahkan root src ke path agar bisa mengimpor modul sebelumnya
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.signal_processor import SignalProcessor
from models.logreg_model import WordAssembler, REVERSE_WORD_CLASSES

# Konfigurasi Path
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dataset', 'models'))
SCALER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dataset', 'raw', 'scalers'))

# Inisialisasi Aplikasi Server
app = FastAPI(title="Neurandiar BCI Live Server", version="1.0")

# Mengizinkan Frontend React (biasanya di port 3000 atau 5173) untuk berkomunikasi dengan Server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Saat produksi, ganti dengan URL React Anda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variabel Global untuk menyimpan Model di memori (agar tidak me-load berulang kali)
ai_models = {}
processor = SignalProcessor()

@app.on_event("startup")
async def load_brain_models():
    """Fungsi ini berjalan sekali saat server dinyalakan untuk memuat AI ke RAM"""
    print("="*50)
    print(" MENGHIDUPKAN MESIN AI NEURANDIAR... ")
    print("="*50)
    try:
        # 1. Muat Arsitektur EEGNet
        eegnet_path = os.path.join(MODEL_DIR, "eegnet_trained.h5")
        ai_models['eegnet'] = load_model(eegnet_path)
        print("[+] Model EEGNet berhasil dimuat ke memori.")

        # 2. Muat Mesin Perakit Kata (LogReg)
        logreg_path = os.path.join(MODEL_DIR, "logistic_regression_assembler.pkl")
        with open(logreg_path, 'rb') as f:
            ai_models['word_assembler'] = pickle.load(f)
        print("[+] Model Word Assembler berhasil dimuat.")

        # 3. Muat Scaler (Opsional di fase simulasi ini)
        # scaler_path = os.path.join(SCALER_DIR, "SUBJ_TEST_scaler.pkl")
        # with open(scaler_path, 'rb') as f:
        #     ai_models['scaler'] = pickle.load(f)
        
    except Exception as e:
        print(f"[!] PERINGATAN: Gagal memuat model. Pastikan file model sudah ada. Error: {e}")

@app.get("/")
def read_root():
    return {"status": "Neurandiar BCI Server is Active", "websocket_endpoint": "/ws/inference"}

@app.websocket("/ws/inference")
async def live_inference_endpoint(websocket: WebSocket):
    """
    Jalur Komunikasi Real-Time. 
    Frontend akan terhubung ke sini untuk menerima teks hasil dekode gelombang otak.
    """
    await websocket.accept()
    print("[+] Klien Frontend/React terhubung ke WebSocket Inference.")
    
    try:
        while True:
            # Di masa depan, di sinilah aliran data LSL dari Emotiv ditangkap.
            # Untuk sekarang, kita MENSIMULASIKAN alur kerja real-time:
            
            # Frontend mengirim sinyal "START_DECODE"
            data = await websocket.receive_text()
            
            if data == "START_DECODE":
                await websocket.send_json({"status": "processing", "message": "Mendengarkan Sinyal Slot 1 (Suku Kata Pertama)..."})
                await asyncio.sleep(2) # Simulasi waktu rekam Slot 1
                
                # AI membuat tebakan Slot 1 (Simulasi Vektor Probabilitas 19 Kelas)
                prob_slot1 = np.random.rand(19)
                
                await websocket.send_json({"status": "processing", "message": "Mendengarkan Sinyal Slot 2 (Suku Kata Kedua)..."})
                await asyncio.sleep(2) # Simulasi waktu rekam Slot 2
                
                # AI membuat tebakan Slot 2
                prob_slot2 = np.random.rand(19)
                
                # MESIN MERAKIT KATA (Inferensi Real-Time)
                combined_probs = np.concatenate((prob_slot1, prob_slot2)).reshape(1, -1)
                
                if 'word_assembler' in ai_models:
                    # Jika model asli sudah ada, gunakan ini
                    pred_idx = ai_models['word_assembler'].predict(combined_probs)[0]
                    final_word = REVERSE_WORD_CLASSES[pred_idx]
                else:
                    # Fallback simulasi jika model belum dilatih
                    final_word = np.random.choice(list(REVERSE_WORD_CLASSES.values()))
                
                # Kirim Hasil Akhir ke Layar React
                await websocket.send_json({
                    "status": "success",
                    "decoded_word": final_word,
                    "confidence": round(float(np.random.uniform(70.0, 99.9)), 2) # Simulasi tingkat keyakinan AI
                })

    except WebSocketDisconnect:
        print("[-] Klien Frontend/React terputus.")
    except Exception as e:
        print(f"[X] Terjadi kesalahan WebSocket: {e}")

if __name__ == "__main__":
    import uvicorn
    # Menjalankan server lokal di port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
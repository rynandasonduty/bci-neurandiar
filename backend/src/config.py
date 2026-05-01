import os
from pathlib import Path
from dotenv import load_dotenv

# =========================================================
# 1. ACQUISITION & HARDWARE CONFIG (EMOTIV CORTEX)
# =========================================================
load_dotenv()
CLIENT_ID = os.getenv("EMOTIV_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("EMOTIV_CLIENT_SECRET", "")
CORTEX_URL = "wss://localhost:6868"

TARGET_WORDS = [
    "Makan", "Minum", "Berak", "Pipis", 
    "Mandi", "Bosan", "Lelah", "Sakit", "Tidur", "Sayang"
]

SLOT_1_DURATION = 5.0    # Durasi pengulangan suku kata pertama
PAUSE_DURATION = 2.0     # Durasi jeda untuk menetralkan keadaan mental
SLOT_2_DURATION = 5.0    # Durasi pengulangan suku kata kedua

TRIALS_PER_SUBJECT = 200 # Total uji coba (100 Terbuka, 100 Bayangan)
BLOCK_SIZE = 20          # Jumlah uji coba per 1 blok sebelum istirahat singkat

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BG_COLOR = (0, 0, 0)         
TEXT_COLOR = (255, 255, 255)

# =========================================================
# 2. ROOT DIRECTORIES (Statis)
# =========================================================
# Mendapatkan path absolut dari direktori 'backend'
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Folder Utama
DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
REPORTS_DIR = os.path.join(BACKEND_DIR, "reports")
LOGS_DIR = os.path.join(BACKEND_DIR, "logs")

# Folder Raw (Tidak disentuh oleh eksperimen, jadi statis)
RAW_DATA_DIR = os.path.join(DATASET_DIR, "raw")

# Path Statis untuk MLflow Database
MLFLOW_DB_PATH = f"sqlite:///{os.path.join(LOGS_DIR, 'mlflow', 'mlruns.db')}"

# =========================================================
# 3. EXPERIMENT ORCHESTRATION ENGINE (Dinamis)
# =========================================================
def setup_experiment(exp_id: str) -> dict:
    """
    Membangun arsitektur sub-folder dinamis untuk eksperimen tertentu.
    
    Args:
        exp_id (str): ID Eksperimen (Contoh: 'E0_Baseline', 'E2_ICA')
        
    Returns:
        dict: Kamus berisi path absolut yang sudah dijamin keberadaannya.
    """
    
    # Merakit Path Dinamis berdasarkan exp_id
    paths = {
        "raw_data": RAW_DATA_DIR, 
        "processed_data": os.path.join(DATASET_DIR, "processed", exp_id),
        "weights": os.path.join(MODELS_DIR, "weights", exp_id),
        "scalers": os.path.join(MODELS_DIR, "scalers", exp_id),
        "reports": os.path.join(REPORTS_DIR, exp_id)
    }
    
    # Menciptakan folder secara otomatis jika belum ada
    for key, path in paths.items():
        if key != "raw_data": 
            os.makedirs(path, exist_ok=True)
            
    # Pastikan folder logs/mlflow eksis untuk pelacakan
    os.makedirs(os.path.join(LOGS_DIR, 'mlflow'), exist_ok=True)
            
    return paths
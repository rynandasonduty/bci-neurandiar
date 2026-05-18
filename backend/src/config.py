import os
from pathlib import Path
from dotenv import load_dotenv

# =========================================================
# 1. ACQUISITION & HARDWARE CONFIGURATION (EMOTIV CORTEX)
# =========================================================
load_dotenv()
CLIENT_ID = os.getenv("EMOTIV_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("EMOTIV_CLIENT_SECRET", "")
CORTEX_URL = "wss://localhost:6868"

TARGET_WORDS = [
    "Makan", "Minum", "Berak", "Pipis", 
    "Mandi", "Bosan", "Lelah", "Sakit", "Tidur", "Sayang"
]

SLOT_1_DURATION = 5.0    
PAUSE_DURATION = 2.0     
SLOT_2_DURATION = 5.0    

TRIALS_PER_SUBJECT = 200 
BLOCK_SIZE = 20          

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BG_COLOR = (0, 0, 0)         
TEXT_COLOR = (255, 255, 255)

# =========================================================
# 2. ROOT DIRECTORY & PERMANENT PATHS
# =========================================================
# Absolute path to the backend root directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw dataset storage
DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")
RAW_DATA_DIR = os.path.join(DATASET_DIR, "raw")

# Primary artifact directories
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
LOGS_DIR = os.path.join(BACKEND_DIR, "logs")

# Static path for the MLflow tracking database
MLFLOW_DB_PATH = f"sqlite:///{os.path.join(LOGS_DIR, 'mlflow', 'mlruns.db')}"

# =========================================================
# 3. EXPERIMENT ORCHESTRATION ENGINE (GOLDEN STANDARD)
# =========================================================
def setup_experiment(exp_id: str, pilar: str = "P1_Global") -> dict:
    """
    Construct the canonical sub-directory structure for a given experiment and paradigm.

    All model weights, scalers, and processed test-set arrays are co-located in a single
    experiment directory to enforce the anti-leakage MLOps architecture. The directory
    is created automatically if it does not exist.

    Args:
        exp_id (str): Experiment identifier (e.g., 'E0_Baseline', 'E3_ERP_N400').
        pilar (str): Paradigm label ('P1_Global', 'P2_EEGNet', or 'P3_SVM').

    Returns:
        dict: A mapping of logical path keys to guaranteed-existent absolute directory paths.
    """
    # All artifacts are co-located at: models/weights/{paradigm}/{experiment_id}/
    pilar_base_dir = os.path.join(MODELS_DIR, "weights", pilar, exp_id)

    paths = {
        "raw_data": RAW_DATA_DIR,
        "processed_data": pilar_base_dir,
        "weights": pilar_base_dir,
        "scalers": pilar_base_dir,
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    return paths
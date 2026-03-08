import os
from dotenv import load_dotenv

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
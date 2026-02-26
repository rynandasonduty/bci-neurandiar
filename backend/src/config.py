import os
from dotenv import load_dotenv

# Memuat kredensial dari file .env
load_dotenv()

# ==========================================
# KREDENSIAL EMOTIV CORTEX
# ==========================================
CLIENT_ID = os.getenv("EMOTIV_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("EMOTIV_CLIENT_SECRET", "")
CORTEX_URL = "wss://localhost:6868"

# ==========================================
# PARAMETER PROTOKOL EKSPERIMEN BCI
# ==========================================
# 10 Kata target berdasarkan kebutuhan dasar
TARGET_WORDS = [
    "Makan", "Minum", "Buang Air Besar", "Buang Air Kecil", 
    "Mandi", "Bosan", "Lelah", "Sakit", "Tidur", "Sayang"
]

# Protokol Waktu 12-Detik (dalam detik)
SLOT_1_DURATION = 5.0    # Durasi pengulangan suku kata pertama
PAUSE_DURATION = 2.0     # Durasi jeda untuk menetralkan keadaan mental
SLOT_2_DURATION = 5.0    # Durasi pengulangan suku kata kedua

# Pengaturan Uji Coba (Trials)
TRIALS_PER_SUBJECT = 200 # Total uji coba (100 Terbuka, 100 Bayangan)
BLOCK_SIZE = 20          # Jumlah uji coba per 1 blok sebelum istirahat singkat

# ==========================================
# PENGATURAN ANTARMUKA VISUAL (PYGAME)
# ==========================================
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BG_COLOR = (0, 0, 0)         # Latar belakang hitam (mengurangi kelelahan mata)
TEXT_COLOR = (255, 255, 255) # Teks putih
import pygame
import time
import random
import sys
import os

# Memastikan jalur impor benar jika dijalankan dari root 'backend'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import (
    TARGET_WORDS, SLOT_1_DURATION, PAUSE_DURATION, SLOT_2_DURATION,
    SCREEN_WIDTH, SCREEN_HEIGHT, BG_COLOR, TEXT_COLOR, TRIALS_PER_SUBJECT, BLOCK_SIZE
)
from src.acquisition.cortex_client import CortexClient

class ExperimentRunner:
    def __init__(self):
        # 1. Inisialisasi Pygame (Visual & Audio)
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("Neurandiar BCI - Data Acquisition")
        self.font_large = pygame.font.SysFont("arial", 72, bold=True)
        self.font_small = pygame.font.SysFont("arial", 36)
        
        # Load Audio (Pastikan file ini ada di folder assets/)
        try:
            self.sound_bip = pygame.mixer.Sound(os.path.join("assets", "bip.wav"))
            self.sound_double_bip = pygame.mixer.Sound(os.path.join("assets", "double_bip.wav"))
        except:
            print("[!] Peringatan: File audio bip.wav atau double_bip.wav tidak ditemukan di folder assets/")
            print("[!] Eksperimen akan berjalan tanpa suara jika diteruskan.")
            self.sound_bip = None
            self.sound_double_bip = None

        # 2. Inisialisasi Emotiv Cortex ini perlu dihubungkan dulu ke EmotivPro Launcher 
        self.cortex = CortexClient()

    def draw_text(self, text, y_offset=0, size="large"):
        """Menampilkan teks di tengah layar"""
        self.screen.fill(BG_COLOR)
        font = self.font_large if size == "large" else self.font_small
        text_surface = font.render(text, True, TEXT_COLOR) 
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, (SCREEN_HEIGHT // 2) + y_offset))
        self.screen.blit(text_surface, text_rect)
        pygame.display.flip()

    def play_sound(self, sound_obj):
        if sound_obj:
            sound_obj.play()

    def wait_for_space(self):
        """Menunggu instruksi dari operator (tombol Spasi)"""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.cortex.close()
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting = False

    def run_trial(self, word, trial_number, phase_name):
        """Menjalankan protokol sinkron 12-detik untuk satu kata"""
        # Tanda Visual (t=0 s) - Menampilkan kata target
        self.draw_text(word.upper())
        
        # Jeda 1 detik agar subjek membaca kata sebelum bip pertama
        time.sleep(1) 

        # --- SLOT 1 (t=0 - 5 s) ---
        self.play_sound(self.sound_bip)
        # Inject Marker untuk Slot 1 (Value 1)
        self.cortex.inject_marker(marker_value=1, marker_label=f"{word}_slot1_{phase_name}")
        time.sleep(SLOT_1_DURATION)

        # --- JEDA ANTAR SLOT (t=5 - 7 s) ---
        self.draw_text("RELAX", size="small")
        time.sleep(PAUSE_DURATION)

        # --- SLOT 2 (t=7 - 12 s) ---
        self.draw_text(word.upper())
        self.play_sound(self.sound_bip)
        # Inject Marker untuk Slot 2 (Value 2)
        self.cortex.inject_marker(marker_value=2, marker_label=f"{word}_slot2_{phase_name}")
        time.sleep(SLOT_2_DURATION)

        # Akhir Uji Coba
        self.play_sound(self.sound_double_bip)
        self.draw_text("+", size="large") # Fixation cross
        time.sleep(1) # Jeda singkat antar trial

    def start_experiment(self):
        # Setup dan koneksi Emotiv
        self.cortex.setup()

        # Menyiapkan urutan trial agar acak tapi seimbang (10 kata x 10 pengulangan)
        trials_per_phase = TRIALS_PER_SUBJECT // 2
        words_list = TARGET_WORDS * (trials_per_phase // len(TARGET_WORDS))
        
        phases = [
            ("FASE 1: OVERT SPEECH (Diucapkan)", "overt"),
            ("FASE 2: IMAGINED SPEECH (Dibayangkan)", "imagined")
        ]

        for phase_title, phase_label in phases:
            random.shuffle(words_list)
            
            self.draw_text(phase_title, y_offset=-50)
            self.draw_text("Tekan SPASI untuk mulai...", y_offset=50, size="small")
            self.wait_for_space()

            for i, word in enumerate(words_list):
                # Mekanisme Istirahat per Blok (setiap 20 trial)
                if i > 0 and i % BLOCK_SIZE == 0:
                    self.draw_text("ISTIRAHAT SEJENAK", y_offset=-50)
                    self.draw_text("Tekan SPASI untuk lanjut...", y_offset=50, size="small")
                    self.wait_for_space()

                # Cek event quit saat loop berjalan
                pygame.event.pump() 

                print(f"[*] Menjalankan Trial {i+1}/{trials_per_phase} - Kata: {word} ({phase_label})")
                self.run_trial(word, i+1, phase_label)

        # Penutup
        self.draw_text("EKSPERIMEN SELESAI. TERIMA KASIH!", y_offset=-50)
        self.draw_text("Menyimpan data...", y_offset=50, size="small")
        time.sleep(3)
        self.cortex.close()
        pygame.quit()

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.start_experiment()
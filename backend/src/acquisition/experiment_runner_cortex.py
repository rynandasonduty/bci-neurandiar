import pygame
import time
import random
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import (
    TARGET_WORDS, SLOT_1_DURATION, PAUSE_DURATION, SLOT_2_DURATION,
    SCREEN_WIDTH, SCREEN_HEIGHT, BG_COLOR, TEXT_COLOR, TRIALS_PER_SUBJECT, BLOCK_SIZE
)
from src.acquisition.cortex_client import CortexClient

SYLLABLE_MAP = {
    "Makan": ("MA", "KAN"), "Minum": ("MI", "NUM"),
    "Berak": ("BE", "RAK"), "Pipis": ("PI", "PIS"),
    "Mandi": ("MAN", "DI"), "Bosan": ("BO", "SAN"),
    "Lelah": ("LE", "LAH"), "Sakit": ("SA", "KIT"),
    "Tidur": ("TI", "DUR"), "Sayang": ("SA", "YANG")
}

def setup_logger(subject_id):
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/{subject_id}_experiment_log.txt"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

class ExperimentRunner:
    def __init__(self, subject_id, logger):
        self.subject_id = subject_id
        self.logger = logger
        
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption(f"Neurandiar BCI - {self.subject_id}")
        self.clock = pygame.time.Clock() # Pengatur FPS
        
        self.font_large = pygame.font.SysFont("arial", 70, bold=True) 
        self.font_medium = pygame.font.SysFont("arial", 48, bold=True)
        self.font_small = pygame.font.SysFont("arial", 30)
        self.font_tracker = pygame.font.SysFont("arial", 20)
        
        # Waktu mulai absolut untuk Stopwatch
        self.experiment_start_time = time.time()
        
        try:
            self.sound_bip = pygame.mixer.Sound(os.path.join("assets", "bip.wav"))
            self.sound_double_bip = pygame.mixer.Sound(os.path.join("assets", "double_bip.wav"))
        except:
            self.logger.warning("Peringatan: File audio tidak ditemukan di assets/")
            self.sound_bip = None
            self.sound_double_bip = None

        self.cortex = CortexClient()

    def play_sound(self, sound_obj):
        if sound_obj:
            sound_obj.play()

    def format_stopwatch(self):
        """Menghitung dan memformat waktu berjalan [MM:SS]"""
        elapsed = int(time.time() - self.experiment_start_time)
        mins = elapsed // 60
        secs = elapsed % 60
        return f"DURASI: {mins:02d}:{secs:02d}"

    def check_events(self):
        """Mengecek input keyboard secara real-time tanpa delay"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.logger.info("Eksperimen dihentikan paksa (ESC).")
                self.cortex.close()
                pygame.quit()
                sys.exit()
                
            # TRIGGER JEDA DARURAT (Tombol P)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.logger.warning(">>> JEDA DARURAT DIAKTIFKAN <<<")
                self.trigger_emergency_pause()

    def trigger_emergency_pause(self):
        """Layar berhenti sampai SPASI ditekan"""
        paused = True
        while paused:
            self.screen.fill(BG_COLOR)
            pause_surf = self.font_medium.render("JEDA DARURAT (Tombol P)", True, (255, 100, 100))
            cont_surf = self.font_small.render("Tekan SPASI untuk melanjutkan...", True, TEXT_COLOR)
            
            # Tetap gambar stopwatch agar waktu terus jalan
            stopwatch_surf = self.font_tracker.render(self.format_stopwatch(), True, (100, 255, 100))
            self.screen.blit(stopwatch_surf, stopwatch_surf.get_rect(bottomleft=(20, SCREEN_HEIGHT - 20)))
            
            self.screen.blit(pause_surf, pause_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 40)))
            self.screen.blit(cont_surf, cont_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 40)))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = False
                    self.logger.info(">>> MELANJUTKAN EKSPERIMEN <<<")
            self.clock.tick(30)

    def draw_trial_screen(self, main_text, size="large", tracker_info=""):
        self.screen.fill(BG_COLOR)
        
        # 1. Teks Utama (Suku Kata / Teks Info)
        if main_text:
            font = self.font_large if size == "large" else self.font_medium
            text_surface = font.render(main_text, True, TEXT_COLOR) 
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(text_surface, text_rect)
        
        # 2. Tracker di Kanan Bawah (Fase, Blok, Trial)
        if tracker_info:
            tracker_surface = self.font_tracker.render(tracker_info, True, (100, 100, 100)) 
            self.screen.blit(tracker_surface, tracker_surface.get_rect(bottomright=(SCREEN_WIDTH - 20, SCREEN_HEIGHT - 20)))

        # 3. Stopwatch di Kiri Bawah
        stopwatch_surf = self.font_tracker.render(self.format_stopwatch(), True, (100, 255, 100))
        self.screen.blit(stopwatch_surf, stopwatch_surf.get_rect(bottomleft=(20, SCREEN_HEIGHT - 20)))

        # 4. Indikator Sinyal EEG di Kanan Atas (UI Placeholder untuk LSL)
        signal_surf = self.font_tracker.render("Sinyal EEG: STANDBY", True, (200, 200, 50))
        self.screen.blit(signal_surf, signal_surf.get_rect(topright=(SCREEN_WIDTH - 20, 20)))

        pygame.display.flip()

    def custom_wait(self, duration_sec, main_text, size="large", tracker_info=""):
        """Pengganti time.sleep() agar sistem tetap responsif dan stopwatch berjalan"""
        start_ticks = pygame.time.get_ticks()
        duration_ms = duration_sec * 1000
        while pygame.time.get_ticks() - start_ticks < duration_ms:
            self.check_events()
            self.draw_trial_screen(main_text, size, tracker_info)
            self.clock.tick(60) # Berjalan 60 FPS

    def run_trial(self, word, trial_number, phase_name, block_number):
        syl1, syl2 = SYLLABLE_MAP[word]
        tracker_text = f"SUBJEK: {self.subject_id}  |  FASE: {phase_name.upper()}  |  BLOK: {block_number}/5  |  TRIAL: {trial_number}/100"

        # Tanda Visual (t=0 s)
        self.custom_wait(1.0, word.upper(), "large", tracker_text)

        # --- SLOT 1 ---
        self.play_sound(self.sound_bip)
        self.cortex.inject_marker(marker_value=1, marker_label=f"{word}_slot1_{phase_name}")
        self.logger.info(f"Berhasil Inject Marker Slot 1: {syl1.upper()}")
        self.custom_wait(SLOT_1_DURATION, syl1.upper(), "large", tracker_text)

        # --- JEDA ANTAR SLOT ---
        self.custom_wait(PAUSE_DURATION, None, "large", tracker_text)

        # --- SLOT 2 ---
        self.play_sound(self.sound_bip)
        self.cortex.inject_marker(marker_value=2, marker_label=f"{word}_slot2_{phase_name}")
        self.logger.info(f"Berhasil Inject Marker Slot 2: {syl2.upper()}")
        self.custom_wait(SLOT_2_DURATION, syl2.upper(), "large", tracker_text)

        # Akhir Uji Coba
        self.play_sound(self.sound_double_bip)
        self.custom_wait(1.0, None, "large", tracker_text)

    def wait_for_space_interactive(self, main_text, sub_text, color=(100, 255, 100)):
        """Menunggu spasi, tapi layar tetap di-update (stopwatch terus jalan)"""
        waiting = True
        while waiting:
            self.screen.fill(BG_COLOR)
            main_surf = self.font_medium.render(main_text, True, TEXT_COLOR)
            sub_surf = self.font_small.render(sub_text, True, color)
            
            stopwatch_surf = self.font_tracker.render(self.format_stopwatch(), True, (100, 255, 100))
            self.screen.blit(stopwatch_surf, stopwatch_surf.get_rect(bottomleft=(20, SCREEN_HEIGHT - 20)))
            
            self.screen.blit(main_surf, main_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 40)))
            self.screen.blit(sub_surf, sub_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 40)))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.cortex.close()
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting = False
            self.clock.tick(30)

    def start_experiment(self):
        self.cortex.setup(record_title=self.subject_id)
        trials_per_phase = TRIALS_PER_SUBJECT // 2
        words_list = TARGET_WORDS * (trials_per_phase // len(TARGET_WORDS))
        
        phases = [
            ("FASE 1: OVERT SPEECH", "overt", "Tekan SPASI untuk mulai blok pertama..."),
            ("FASE 2: IMAGINED SPEECH", "imagined", "Tekan SPASI untuk mulai blok pertama...")
        ]

        for phase_title, phase_label, start_instruction in phases:
            random.shuffle(words_list)
            self.logger.info(f"=== MEMULAI {phase_title} ===")
            self.wait_for_space_interactive(phase_title, start_instruction)

            for i, word in enumerate(words_list):
                block_number = (i // BLOCK_SIZE) + 1
                trial_number = i + 1

                if i > 0 and i % BLOCK_SIZE == 0:
                    self.logger.info(f"--- Istirahat (Blok {block_number-1} Selesai) ---")
                    self.wait_for_space_interactive(f"ISTIRAHAT SEJENAK (Blok {block_number-1} Selesai)", "Tekan SPASI untuk lanjut ke blok berikutnya...", TEXT_COLOR)

                self.logger.info(f"Menjalankan Trial {trial_number}/{trials_per_phase} (Blok {block_number}) - Kata: {word} (Fase {phase_label.capitalize()})")
                self.run_trial(word, trial_number, phase_label, block_number)

        self.logger.info("=== EKSPERIMEN SELESAI ===")
        self.custom_wait(3.0, "EKSPERIMEN SELESAI. TERIMA KASIH!", "medium")
        self.cortex.close()
        pygame.quit()

if __name__ == "__main__":
    print("="*50)
    print(" SELAMAT DATANG DI SISTEM AKUISISI BCI NEURANDIAR ")
    print("="*50)
    subject_input = input("Masukkan ID Subjek (contoh: SUBJ01) dan tekan ENTER: ").strip()
    if not subject_input: subject_input = "SUBJ_TEST"
        
    exp_logger = setup_logger(subject_input)
    exp_logger.info(f"Sistem dimulai untuk subjek: {subject_input}")
    
    runner = ExperimentRunner(subject_id=subject_input, logger=exp_logger)
    runner.start_experiment()
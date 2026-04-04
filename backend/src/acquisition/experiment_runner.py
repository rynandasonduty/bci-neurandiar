import pygame
import time
import random
import sys
import os
import logging
from pylsl import StreamInfo, StreamOutlet


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)
# Menggunakan parameter fallback jika tidak ada di src.config
try:
    from src.config import (
        TARGET_WORDS, SLOT_1_DURATION, PAUSE_DURATION, SLOT_2_DURATION,
        SCREEN_WIDTH, SCREEN_HEIGHT, BG_COLOR, TEXT_COLOR, TRIALS_PER_SUBJECT, BLOCK_SIZE
    )
except ImportError:
    # Parameter default sesuai Metodologi Skripsi
    TARGET_WORDS = ["Makan", "Minum", "Berak", "Pipis", "Mandi", "Bosan", "Lelah", "Sakit", "Tidur", "Sayang"]
    SLOT_1_DURATION = 5.0
    PAUSE_DURATION = 2.0
    SLOT_2_DURATION = 5.0
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
    BG_COLOR = (30, 30, 30)
    TEXT_COLOR = (255, 255, 255)
    TRIALS_PER_SUBJECT = 200
    BLOCK_SIZE = 20

# Pemetaan Suku Kata dan ID (K=19)
SYLLABLE_MAP = {
    "Makan": ("MA", "KAN"), "Minum": ("MI", "NUM"),
    "Berak": ("BE", "RAK"), "Pipis": ("PI", "PIS"),
    "Mandi": ("MAN", "DI"), "Bosan": ("BO", "SAN"),
    "Lelah": ("LE", "LAH"), "Sakit": ("SA", "KIT"),
    "Tidur": ("TI", "DUR"), "Sayang": ("SA", "YANG")
}

SYLLABLE_IDS = {
    "MA": 1, "KAN": 2, "MI": 3, "NUM": 4, "BE": 5, "RAK": 6,
    "PI": 7, "PIS": 8, "MAN": 9, "DI": 10, "BO": 11, "SAN": 12,
    "LE": 13, "LAH": 14, "SA": 15, "KIT": 16, "TI": 17, "DUR": 18,
    "YANG": 19
}

def setup_logger(subject_id):
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{subject_id}_experiment_log.txt")
    
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
        
        # Layar Fullscreen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption(f"Neurandiar BCI - {self.subject_id}")
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.SysFont("arial", 90, bold=True) 
        self.font_medium = pygame.font.SysFont("arial", 48, bold=True)
        self.font_small = pygame.font.SysFont("arial", 30)
        self.font_tracker = pygame.font.SysFont("arial", 20)
        
        self.experiment_start_time = time.time()
        
        # Load Audio (Path Absolut)
        path_bip = os.path.join(BASE_DIR, "assets", "bip.wav")
        path_double_bip = os.path.join(BASE_DIR, "assets", "double_bip.wav")
        
        try:
            self.sound_bip = pygame.mixer.Sound(path_bip)
            self.sound_double_bip = pygame.mixer.Sound(path_double_bip)
            self.logger.info("[+] Audio berhasi dimuat.")
        except Exception as e:
            self.logger.warning(f"[-] Gagal memuat audio: {e}. Pastikan file ada di {path_bip}")
            self.sound_bip = None
            self.sound_double_bip = None

        # Setup LSL Marker Outlet
        self.logger.info("[*] Membangun aliran LSL Marker...")
        info = StreamInfo(name='Neurandiar_Markers', 
                          type='Markers', 
                          channel_count=1, 
                          nominal_srate=0, 
                          channel_format='int32', 
                          source_id=f'bci_markers_{self.subject_id}')
        self.lsl_outlet = StreamOutlet(info)
        self.logger.info("[+] LSL Marker Outlet Siap!")

    def play_sound(self, sound_obj):
        if sound_obj:
            sound_obj.play()

    def format_stopwatch(self):
        elapsed = int(time.time() - self.experiment_start_time)
        mins = elapsed // 60
        secs = elapsed % 60
        return f"DURASI: {mins:02d}:{secs:02d}"

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.logger.info("Eksperimen dihentikan paksa (ESC).")
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.logger.warning(">>> JEDA DARURAT DIAKTIFKAN <<<")
                self.trigger_emergency_pause()

    def trigger_emergency_pause(self):
        paused = True
        while paused:
            self.screen.fill(BG_COLOR)
            pause_surf = self.font_medium.render("JEDA DARURAT (Tombol P)", True, (255, 100, 100))
            cont_surf = self.font_small.render("Tekan SPASI untuk melanjutkan...", True, TEXT_COLOR)
            
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

    def draw_trial_screen(self, main_text, sub_text="", tracker_info=""):
        self.screen.fill(BG_COLOR)
        
        if main_text:
            text_surface = self.font_large.render(main_text, True, TEXT_COLOR) 
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(text_surface, text_rect)
            
        if sub_text:
            sub_surface = self.font_small.render(sub_text, True, (150, 150, 150))
            sub_rect = sub_surface.get_rect(center=(SCREEN_WIDTH // 2, (SCREEN_HEIGHT // 2) - 80))
            self.screen.blit(sub_surface, sub_rect)
        
        if tracker_info:
            tracker_surface = self.font_tracker.render(tracker_info, True, (100, 100, 100)) 
            self.screen.blit(tracker_surface, tracker_surface.get_rect(bottomright=(SCREEN_WIDTH - 20, SCREEN_HEIGHT - 20)))

        stopwatch_surf = self.font_tracker.render(self.format_stopwatch(), True, (100, 255, 100))
        self.screen.blit(stopwatch_surf, stopwatch_surf.get_rect(bottomleft=(20, SCREEN_HEIGHT - 20)))

        signal_surf = self.font_tracker.render("LSL Marker: AKTIF", True, (50, 200, 50))
        self.screen.blit(signal_surf, signal_surf.get_rect(topright=(SCREEN_WIDTH - 20, 20)))

        pygame.display.flip()

    def custom_wait(self, duration_sec, main_text, sub_text="", tracker_info=""):
        start_ticks = pygame.time.get_ticks()
        duration_ms = duration_sec * 1000
        while pygame.time.get_ticks() - start_ticks < duration_ms:
            self.check_events()
            self.draw_trial_screen(main_text, sub_text, tracker_info)
            self.clock.tick(60)

    def run_trial(self, word, trial_number, phase_name, block_number):
        syl1, syl2 = SYLLABLE_MAP[word]
        id_syl1 = SYLLABLE_IDS[syl1]
        id_syl2 = SYLLABLE_IDS[syl2]
        
        tracker_text = f"SUBJEK: {self.subject_id}  |  FASE: {phase_name.upper()}  |  BLOK: {block_number}/5  |  TRIAL: {trial_number}/100"

        # FASE VISUAL CUE: Menampilkan kata utuh sebagai instruksi awal
        self.custom_wait(1.5, word.upper(), "TARGET KATA", tracker_text)

        # FASE SLOT 1 (t=0-5s): Menampilkan Suku Kata Pertama, Bunyikan BIP, Inject Marker
        self.play_sound(self.sound_bip)
        self.lsl_outlet.push_sample([id_syl1])
        self.logger.info(f"Inject Marker Slot 1: {syl1} (ID: {id_syl1})")
        self.custom_wait(SLOT_1_DURATION, syl1.upper(), f"Ulangi Suku Kata Ini ({phase_name})", tracker_text)

        # FASE INTER-SLOT PAUSE (t=5-7s): Layar kosong (Hitam) untuk menetralkan keadaan mental
        self.custom_wait(PAUSE_DURATION, "", "", tracker_text)

        # FASE SLOT 2 (t=7-12s): Menampilkan Suku Kata Kedua, Bunyikan BIP, Inject Marker
        self.play_sound(self.sound_bip)
        self.lsl_outlet.push_sample([id_syl2])
        self.logger.info(f"Inject Marker Slot 2: {syl2} (ID: {id_syl2})")
        self.custom_wait(SLOT_2_DURATION, syl2.upper(), f"Ulangi Suku Kata Ini ({phase_name})", tracker_text)

        # AKHIR TRIAL: Double BIP dan jeda sejenak sebelum trial berikutnya
        self.play_sound(self.sound_double_bip)
        self.custom_wait(1.0, "", "", tracker_text)

    def wait_for_space_interactive(self, main_text, sub_text, color=(100, 255, 100)):
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
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting = False
            self.clock.tick(30)

    def start_experiment(self):
        self.logger.info("[*] Memulai Protokol Eksperimen...")
        
        trials_per_phase = TRIALS_PER_SUBJECT // 2
        words_list = TARGET_WORDS * (trials_per_phase // len(TARGET_WORDS))
        
        phases = [
            ("FASE 1: OVERT SPEECH", "overt", "Tekan SPASI untuk mulai Blok 1..."),
            ("FASE 2: IMAGINED SPEECH", "imagined", "Tekan SPASI untuk mulai Blok 1...")
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

                self.logger.info(f"Menjalankan Trial {trial_number}/{trials_per_phase} (Blok {block_number}) - Kata: {word}")
                self.run_trial(word, trial_number, phase_label, block_number)

        self.logger.info("=== EKSPERIMEN SELESAI ===")
        self.custom_wait(3.0, "EKSPERIMEN SELESAI. TERIMA KASIH!", "")
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
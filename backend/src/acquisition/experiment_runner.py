import pygame
import time
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import (
    TARGET_WORDS, SLOT_1_DURATION, PAUSE_DURATION, SLOT_2_DURATION,
    SCREEN_WIDTH, SCREEN_HEIGHT, BG_COLOR, TEXT_COLOR, TRIALS_PER_SUBJECT, BLOCK_SIZE
)
from src.acquisition.cortex_client import CortexClient

SYLLABLE_MAP = {
    "Makan": ("MA", "KAN"),
    "Minum": ("MI", "NUM"),
    "Berak": ("BE", "RAK"),
    "Pipis": ("PI", "PIS"),
    "Mandi": ("MAN", "DI"),
    "Bosan": ("BO", "SAN"),
    "Lelah": ("LE", "LAH"),
    "Sakit": ("SA", "KIT"),
    "Tidur": ("TI", "DUR"),
    "Sayang": ("SA", "YANG")
}

class ExperimentRunner:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("Neurandiar BCI - Data Acquisition")
        
        self.font_large = pygame.font.SysFont("arial", 70, bold=True) 
        self.font_medium = pygame.font.SysFont("arial", 48, bold=True)
        self.font_small = pygame.font.SysFont("arial", 30)
        self.font_tracker = pygame.font.SysFont("arial", 20)
        
        try:
            self.sound_bip = pygame.mixer.Sound(os.path.join("assets", "bip.wav"))
            self.sound_double_bip = pygame.mixer.Sound(os.path.join("assets", "double_bip.wav"))
        except:
            print("[!] Peringatan: File audio tidak ditemukan di assets/")
            self.sound_bip = None
            self.sound_double_bip = None

        self.cortex = CortexClient()

    def play_sound(self, sound_obj):
        if sound_obj:
            sound_obj.play()

    def wait_for_space(self):
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.cortex.close()
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting = False

    def draw_trial_screen(self, main_text, size="large", tracker_info=None):
        self.screen.fill(BG_COLOR)
        
        if main_text:
            font = self.font_large if size == "large" else self.font_small
            text_surface = font.render(main_text, True, TEXT_COLOR) 
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(text_surface, text_rect)
        
        if tracker_info:
            tracker_surface = self.font_tracker.render(tracker_info, True, (100, 100, 100)) 
            tracker_rect = tracker_surface.get_rect(bottomright=(SCREEN_WIDTH - 20, SCREEN_HEIGHT - 20))
            self.screen.blit(tracker_surface, tracker_rect)

        pygame.display.flip()

    def run_trial(self, word, trial_number, phase_name, block_number):
        syl1, syl2 = SYLLABLE_MAP[word]
        tracker_text = f"FASE: {phase_name.upper()}  |  BLOK: {block_number}/5  |  TRIAL: {trial_number}/100"

        self.draw_trial_screen(word.upper(), tracker_info=tracker_text)
        time.sleep(1) 

        # --- SLOT 1 (t=0 - 5 s) ---
        self.play_sound(self.sound_bip)
        self.cortex.inject_marker(marker_value=1, marker_label=f"{word}_slot1_{phase_name}")
        print(f"[-] Berhasil Inject Marker Slot 1: {syl1.upper()}")
        self.draw_trial_screen(syl1.upper(), tracker_info=tracker_text)
        time.sleep(SLOT_1_DURATION)

        # --- JEDA ANTAR SLOT (t=5 - 7 s) ---
        self.draw_trial_screen(None, tracker_info=tracker_text)
        time.sleep(PAUSE_DURATION)

        # --- SLOT 2 (t=7 - 12 s) ---
        self.play_sound(self.sound_bip)
        self.cortex.inject_marker(marker_value=2, marker_label=f"{word}_slot2_{phase_name}")
        print(f"[-] Berhasil Inject Marker Slot 2: {syl2.upper()}")
        self.draw_trial_screen(syl2.upper(), tracker_info=tracker_text)
        time.sleep(SLOT_2_DURATION)

        # Akhir Uji Coba
        self.play_sound(self.sound_double_bip)
        self.draw_trial_screen(None, tracker_info=tracker_text)
        time.sleep(1) 

    def start_experiment(self):
        self.cortex.setup()

        trials_per_phase = TRIALS_PER_SUBJECT // 2
        words_list = TARGET_WORDS * (trials_per_phase // len(TARGET_WORDS))
        
        phases = [
            ("FASE 1: OVERT SPEECH (Diucapkan)", "overt", "Instruksi: Ucapkan suku kata secara berulang-ulang dengan bersuara."),
            ("FASE 2: IMAGINED SPEECH (Dibayangkan)", "imagined", "Instruksi: Bayangkan mengucapkan suku kata dalam pikiran TANPA menggerakkan bibir/rahang.")
        ]

        for phase_title, phase_label, phase_instruction in phases:
            random.shuffle(words_list)
            
            self.screen.fill(BG_COLOR)
            title_surf = self.font_medium.render(phase_title, True, TEXT_COLOR)
            inst_surf = self.font_small.render(phase_instruction, True, (200, 200, 200))
            start_surf = self.font_small.render("Tekan SPASI untuk mulai blok pertama...", True, (100, 255, 100))
            
            self.screen.blit(title_surf, title_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 60)))
            self.screen.blit(inst_surf, inst_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 20)))
            self.screen.blit(start_surf, start_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 100)))
            pygame.display.flip()
            self.wait_for_space()

            for i, word in enumerate(words_list):
                block_number = (i // BLOCK_SIZE) + 1
                trial_number = i + 1

                if i > 0 and i % BLOCK_SIZE == 0:
                    self.screen.fill(BG_COLOR)
                    break_surf = self.font_medium.render(f"ISTIRAHAT SEJENAK (Blok {block_number-1} Selesai)", True, TEXT_COLOR)
                    cont_surf = self.font_small.render("Tekan SPASI untuk lanjut ke blok berikutnya...", True, TEXT_COLOR)
                    self.screen.blit(break_surf, break_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 40)))
                    self.screen.blit(cont_surf, cont_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 40)))
                    pygame.display.flip()
                    self.wait_for_space()

                pygame.event.pump() 
                
                print(f"\n[*] Menjalankan Trial {trial_number}/{trials_per_phase} (Blok {block_number}) - Kata: {word} (Fase {phase_label.capitalize()})")
                self.run_trial(word, trial_number, phase_label, block_number)

        self.screen.fill(BG_COLOR)
        end_surf = self.font_medium.render("EKSPERIMEN SELESAI. TERIMA KASIH!", True, TEXT_COLOR)
        self.screen.blit(end_surf, end_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)))
        pygame.display.flip()
        time.sleep(3)
        self.cortex.close()
        pygame.quit()

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.start_experiment()
import numpy as np
import os
import sys
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Menghubungkan ke Mesin Direktori di config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import setup_experiment

# Pemetaan 10 Kata Target ke Kelas Integer (0-9)
WORD_CLASSES = {
    "MAKAN": 0, "MINUM": 1, "BERAK": 2, "PIPIS": 3, "MANDI": 4,
    "BOSAN": 5, "LELAH": 6, "SAKIT": 7, "TIDUR": 8, "SAYANG": 9
}

# Pemetaan terbalik untuk mengubah tebakan angka kembali menjadi teks kata
REVERSE_WORD_CLASSES = {v: k for k, v in WORD_CLASSES.items()}

class WordAssembler:
    def __init__(self, exp_id=None):
        """
        Inisialisasi Word Assembler.
        Jika exp_id diberikan, ia akan otomatis menyimpan/memuat dari folder eksperimen tersebut.
        Jika None, ia akan menggunakan mode Simulasi/Dry-Run di folder default.
        """
        if exp_id:
            paths = setup_experiment(exp_id)
            self.model_dir = paths["weights"]
            self.model_path = os.path.join(self.model_dir, f"logreg_assembler_{exp_id}.pkl")
        else:
            # Mode Default/Simulasi (Mencegah error jika dipanggil tanpa konteks eksperimen)
            self.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'weights', 'E_Sim'))
            self.model_path = os.path.join(self.model_dir, "logreg_assembler_sim.pkl")
            os.makedirs(self.model_dir, exist_ok=True)
            
        # Inisialisasi model Regresi Logistik
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def train(self, X_probs, y_words):
        """
        Melatih Regresi Logistik.
        - X_probs: Matriks (Jumlah Sampel, 38) -> 19 probabilitas Slot 1 + 19 probabilitas Slot 2
        - y_words: Array (Jumlah Sampel,) -> Label integer 0-9 untuk 10 kata target
        """
        print(f"[*] Melatih Model Regresi Logistik (Word Assembler)...")
        
        # Membagi data untuk evaluasi internal model (80% Latih, 20% Uji)
        X_train, X_test, y_train, y_test = train_test_split(
            X_probs, y_words, test_size=0.2, random_state=42, stratify=y_words
        )

        # Proses melatih mesin
        self.model.fit(X_train, y_train)

        print("[*] Mengevaluasi Model Perakit Kata...")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"[+] Akurasi Word Assembler: {acc * 100:.2f}%\n")
        print("Laporan Klasifikasi (Confusion Matrix Metrics):")
        print(classification_report(y_test, y_pred, target_names=list(WORD_CLASSES.keys())))
        
        return acc # Kembalikan akurasi agar bisa dicatat oleh MLflow nantinya

    def save_model(self):
        """Membekukan dan menyimpan model ke dalam file .pkl"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[SUCCESS] Model Word Assembler disimpan di: {self.model_path}")

    def load_model(self):
        """Memuat model yang sudah dilatih dari file .pkl"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model tidak ditemukan di {self.model_path}. Latih model terlebih dahulu!")
            
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"[+] Model Word Assembler berhasil dimuat dari: {self.model_path}")

    def assemble_word(self, prob_slot1, prob_slot2):
        """
        Fungsi Inferensi Real-Time: Menebak kata utuh dari probabilitas dua suku kata.
        prob_slot1: Array (19,) dari EEGNet (Slot 1)
        prob_slot2: Array (19,) dari EEGNet (Slot 2)
        """
        # Gabungkan probabilitas menjadi 1 vektor berdimensi 38
        combined_probs = np.concatenate((prob_slot1, prob_slot2))
        
        # Ubah bentuk array (reshape) agar bisa dibaca oleh scikit-learn (1 baris, 38 kolom)
        X_input = combined_probs.reshape(1, -1)

        # Lakukan tebakan (mengembalikan angka 0-9)
        pred_idx = self.model.predict(X_input)[0]
        
        # Terjemahkan angka menjadi Kata Teks
        pred_word = REVERSE_WORD_CLASSES[pred_idx]

        return pred_word

if __name__ == "__main__":
    # ==========================================
    # DRY-RUN / SIMULASI PERAKITAN KATA
    # ==========================================
    print("="*50)
    print(" SIMULASI PELATIHAN WORD ASSEMBLER ")
    print("="*50)

    # Inisialisasi tanpa exp_id akan otomatis masuk mode simulasi
    assembler = WordAssembler()

    # 1. Membuat Dummy Data
    print("[*] Membuat 1000 data probabilitas simulasi...")
    X_dummy_probs = np.random.rand(1000, 38) 
    y_dummy_words = np.random.randint(0, 10, 1000)

    # 2. Melatih Regresi Logistik
    assembler.train(X_dummy_probs, y_dummy_words)
    assembler.save_model()

    # 3. Simulasi Pengujian Real-Time
    print("\n" + "="*50)
    print(" UJI COBA INFERENSI (REAL-TIME SIMULATION) ")
    print("="*50)
    
    dummy_p1 = np.random.rand(19)
    dummy_p2 = np.random.rand(19)

    # Memuat model (Simulasi)
    assembler.load_model()
    predicted = assembler.assemble_word(dummy_p1, dummy_p2)
    print(f"[+] HASIL DEKODE KATA AKHIR: {predicted}")
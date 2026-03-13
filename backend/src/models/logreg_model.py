import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Pemetaan 10 Kata Target ke Kelas Integer (0-9)
WORD_CLASSES = {
    "MAKAN": 0, "MINUM": 1, "BERAK": 2, "PIPIS": 3, "MANDI": 4,
    "BOSAN": 5, "LELAH": 6, "SAKIT": 7, "TIDUR": 8, "SAYANG": 9
}

# Pemetaan terbalik untuk mengubah tebakan angka kembali menjadi teks kata
REVERSE_WORD_CLASSES = {v: k for k, v in WORD_CLASSES.items()}

class WordAssembler:
    def __init__(self, model_dir="../../dataset/models"):
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, "logistic_regression_assembler.pkl")
        
        # Inisialisasi model Regresi Logistik
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, X_probs, y_words):
        """
        Melatih Regresi Logistik.
        - X_probs: Matriks (Jumlah Sampel, 38) -> 19 probabilitas Slot 1 + 19 probabilitas Slot 2
        - y_words: Array (Jumlah Sampel,) -> Label integer 0-9 untuk 10 kata target
        """
        print("[*] Melatih Model Regresi Logistik (Word Assembler)...")
        
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
        print(f"[+] Model Word Assembler berhasil dimuat.")

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

    assembler = WordAssembler()

    # 1. Membuat Dummy Data (Seolah-olah ini adalah probabilitas dari EEGNet)
    print("[*] Membuat 1000 data probabilitas simulasi...")
    # 1000 sampel percobaan x 38 nilai probabilitas
    X_dummy_probs = np.random.rand(1000, 38) 
    
    # 1000 label kata target acak (0 sampai 9)
    y_dummy_words = np.random.randint(0, 10, 1000)

    # 2. Melatih Regresi Logistik
    assembler.train(X_dummy_probs, y_dummy_words)
    assembler.save_model()

    # 3. Simulasi Pengujian Real-Time
    print("\n" + "="*50)
    print(" UJI COBA INFERENSI (REAL-TIME SIMULATION) ")
    print("="*50)
    
    # Seandainya EEGNet mengeluarkan probabilitas untuk Slot 1 ("MA")
    dummy_p1 = np.random.rand(19)
    
    # Seandainya EEGNet mengeluarkan probabilitas untuk Slot 2 ("KAN")
    dummy_p2 = np.random.rand(19)

    predicted = assembler.assemble_word(dummy_p1, dummy_p2)
    print(f"[+] Input Probabilitas Slot 1 (Shape): {dummy_p1.shape}")
    print(f"[+] Input Probabilitas Slot 2 (Shape): {dummy_p2.shape}")
    print(f"[+] HASIL DEKODE KATA AKHIR: {predicted}")
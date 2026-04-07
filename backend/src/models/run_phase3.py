import os
import sys
import numpy as np

# Beritahu Python letak folder preprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocessing')))

from build_logreg_dataset import LogRegDatasetBuilder
from logreg_model import WordAssembler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PROCESSED_DIR = "../../dataset/processed"

def main():
    print("\n" + "="*50)
    print(" FASE 3: MEMBANGUN DATASET & MELATIH REGRESI LOGISTIK ")
    print("="*50)
    
    try:
        # 1. Minta EEGNet mengekstrak fitur 38-dimensi
        print("[*] Mengekstrak fitur dengan EEGNet...")
        logreg_builder = LogRegDatasetBuilder()
        logreg_builder.build_dataset()
        
        # 2. Muat hasil ekstraksi
        X_word = np.load(os.path.join(PROCESSED_DIR, "X_word_features.npy"))
        y_word = np.load(os.path.join(PROCESSED_DIR, "y_word_labels.npy"))
        
        # 3. Latih Model Perakitan Kata
        print("[*] Melatih algoritma Regresi Logistik...")
        assembler = WordAssembler()
        X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(
            X_word, y_word, test_size=0.2, random_state=42, stratify=y_word
        )
        assembler.model.fit(X_w_train, y_w_train)
        assembler.save_model()
        
        # 4. Ujian Akhir AI
        y_w_pred = assembler.model.predict(X_w_test)
        logreg_acc = accuracy_score(y_w_test, y_w_pred)
        
        print("\n" + "="*50)
        print(f" 🎉 AKURASI FINAL TINGKAT KATA: {logreg_acc * 100:.2f}% 🎉")
        print("="*50)
        print("[SUCCESS] Seluruh Pipeline AI telah selesai!")
        
    except Exception as e:
        print(f"\n[-] GAGAL: {e}")

if __name__ == "__main__":
    main()
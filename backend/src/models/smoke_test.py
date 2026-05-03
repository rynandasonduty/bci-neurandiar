import os
import sys
import glob
import numpy as np

# Menghubungkan direktori agar bisa memanggil modul
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.build_dataset import DatasetBuilder
from utils.data_utils import three_way_split, fit_and_apply_scaler

def run_smoke_test():
    print("="*60)
    print(" 🚀 MEMULAI SMOKE TEST (VALIDASI PIPELINE ANTI-LEAKAGE) ")
    print("="*60)
    
    # 1. Inisialisasi DatasetBuilder dengan ID Eksperimen dummy
    EXP_ID = "E_SMOKE_TEST"
    builder = DatasetBuilder(exp_id=EXP_ID)
    
    # 2. Cari satu subjek secara otomatis dari data mentah
    log_files = glob.glob(os.path.join(builder.raw_data_dir, "logs", "*_experiment_log.txt"))
    if not log_files:
        print("[X] ERROR: Data mentah tidak ditemukan! Pastikan folder raw_data memiliki log dan csv.")
        return
        
    first_log = log_files[0]
    subject_id = os.path.basename(first_log).replace("_experiment_log.txt", "")
    csv_files = glob.glob(os.path.join(builder.raw_data_dir, f"{subject_id}*.csv"))
    
    if not csv_files:
        print(f"[X] ERROR: File CSV untuk subjek {subject_id} tidak ditemukan.")
        return

    print(f"\n[*] Menguji Ekstraksi pada Subjek: {subject_id}")
    
    # 3. Ekstraksi Data Murni (Tanpa Normalisasi)
    X_list, y_list = builder.process_subject(subject_id, csv_files[0], first_log)
    
    if len(X_list) == 0:
        print("[X] ERROR: Ekstraksi gagal. Tidak ada data yang dikembalikan.")
        return
        
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Transposisi ke format EEGNet (Samples, Channels, Time, Depth)
    X_eeg = np.transpose(X, (0, 2, 1))
    X_eeg = np.expand_dims(X_eeg, axis=3)
    
    print(f"[CHECK] Shape Data Mentah (Output build_dataset) : {X_eeg.shape}")
    
    # 4. Uji Pemisahan 3-Way Split (70/15/15)
    print("\n[*] Menguji Fungsi 3-Way Split (utils/data_utils.py)...")
    X_tr, X_v, X_te, y_tr, y_v, y_te = three_way_split(X_eeg, y, val_ratio=0.15, test_ratio=0.15)
    
    print(f"[CHECK] Shape Train (70%) : {X_tr.shape}")
    print(f"[CHECK] Shape Val   (15%) : {X_v.shape}")
    print(f"[CHECK] Shape Test  (15%) : {X_te.shape}")
    
    # 5. Uji Normalisasi (Hanya fit pada Train)
    print("\n[*] Menguji Fungsi Normalisasi (fit_and_apply_scaler)...")
    X_tr_s, X_v_s, X_te_s, scaler = fit_and_apply_scaler(X_tr, X_v, X_te)
    
    train_mean = X_tr_s.mean()
    val_mean = X_v_s.mean()
    
    print(f"[CHECK] Mean Train Scaled : {train_mean:.6f} (Harus sangat mendekati 0)")
    print(f"[CHECK] Mean Val Scaled   : {val_mean:.6f} (Tidak akan persis 0, ini normal)")
    
    if abs(train_mean) < 1e-4:
        print("\n" + "="*60)
        print(" ✅ SMOKE TEST LULUS! PIPELINE ANDA SUDAH KEBAL DATA LEAKAGE.")
        print("="*60)
        print("Anda sekarang aman untuk menjalankan 'run_master_experiments.py'.")
    else:
        print("\n[!] PERINGATAN: Mean Train tidak mendekati 0. Cek kembali fungsi scaler Anda.")

if __name__ == "__main__":
    run_smoke_test()
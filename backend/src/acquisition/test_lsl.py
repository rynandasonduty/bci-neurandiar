from pylsl import StreamInlet, resolve_byprop

def test_lsl_connection():
    print("[*] Mencari aliran data (stream) LSL dari EmotivPRO...")
    
    # Mencari stream dengan tipe 'EEG' (Menunggu maksimal 5 detik)
    print("[*] Menunggu pancaran sinyal... Pastikan EmotivPRO menyala.")
    streams = resolve_byprop('type', 'EEG', timeout=5)
    
    if not streams:
        print("[-] Tidak ada stream EEG yang ditemukan. Pastikan 'EEG Data Outlet' di EmotivPRO sudah ON!")
        return

    print(f"[+] Ditemukan Stream: {streams[0].name()} | ID: {streams[0].source_id()}")
    print("[*] Membuka koneksi... (Tekan CTRL+C untuk berhenti)")
    
    # Membuat StreamInlet untuk menarik data dari jaringan
    inlet_eeg = StreamInlet(streams[0])
    
    try:
        while True:
            # Mengambil satu sampel data dan timestamp-nya
            sample, timestamp = inlet_eeg.pull_sample()
            
            # Print 3 channel pertama agar terminal tidak terlalu penuh
            print(f"[{timestamp:.3f}] AF3: {sample[0]:.2f} µV | F7: {sample[1]:.2f} µV | F3: {sample[2]:.2f} µV")
            
    except KeyboardInterrupt:
        print("\n[+] Uji coba LSL dihentikan.")

if __name__ == "__main__":
    test_lsl_connection()
import json
import ssl
import time
from websocket import create_connection
from src.config import CORTEX_URL, CLIENT_ID, CLIENT_SECRET

class CortexClient:
    def __init__(self):
        self.ws = None
        self.auth_token = None
        self.session_id = None
        self.headset_id = None
        self.req_id = 1  # ID increment untuk JSON RPC

    def connect(self):
        print("[*] Menghubungkan ke Emotiv Cortex API...")
        self.ws = create_connection(CORTEX_URL, sslopt={"cert_reqs": ssl.CERT_NONE})
        print("[+] Terhubung ke WebSocket Emotiv!")

    def send_request(self, method, params=None):
        """Fungsi helper untuk mengirim dan menerima JSON RPC ke Cortex"""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.req_id
        }
        self.req_id += 1
        self.ws.send(json.dumps(payload))
        
        result = json.loads(self.ws.recv())
        return result

    def request_access(self):
        print("[*] Meminta Hak Akses (Request Access)...")
        res = self.send_request("requestAccess", {
            "clientId": CLIENT_ID,
            "clientSecret": CLIENT_SECRET
        })
        if res.get('result', {}).get('accessGranted') == False:
            print("\n[!] PERHATIAN: Silakan buka aplikasi Emotiv Launcher")
            print("[!] Klik tombol 'Approve' pada permintaan akses Neurandiar BCI.\n")
            time.sleep(5)

    def authorize(self):
        print("[*] Melakukan Otorisasi...")
        res = self.send_request("authorize", {
            "clientId": CLIENT_ID,
            "clientSecret": CLIENT_SECRET,
            "debit": 1  
        })
        
        if 'error' in res:
            res = self.send_request("authorize", {
                "clientId": CLIENT_ID,
                "clientSecret": CLIENT_SECRET,
                "debit": 0
            })
        
        self.auth_token = res.get('result', {}).get('cortexToken')
        if self.auth_token:
            print("[+] Otorisasi Berhasil!")
        else:
            raise Exception(f"[-] Otorisasi Gagal: {res}")

    def query_headset(self):
        print("[*] Mencari Headset Emotiv EPOC X...")
        res = self.send_request("queryHeadsets")
        for headset in res.get('result', []):
            if headset['status'] in ['connected', 'discovered']:
                self.headset_id = headset['id']
                print(f"[+] Headset Ditemukan: {self.headset_id} ({headset['status']})")
                return True
        raise Exception("[-] Headset tidak ditemukan. Pastikan headset menyala dan terhubung via Dongle/Bluetooth!")

    def create_session(self):
        print("[*] Membuat Sesi Eksperimen...")
        res = self.send_request("createSession", {
            "cortexToken": self.auth_token,
            "headset": self.headset_id,
            "status": "active"
        })
        self.session_id = res.get('result', {}).get('id')
        if self.session_id:
            print(f"[+] Sesi Berhasil Dibuat: {self.session_id}")
        else:
            raise Exception(f"[-] Gagal Membuat Sesi: {res}")

    def start_record(self, record_title="BCI_Neurandiar"):
        """Memulai penulisan data ke file Record di Emotiv"""
        print(f"[*] Menyiapkan perekaman data (Record: {record_title})...")
        
        time.sleep(2) 
        
        res = self.send_request("updateSession", {
            "cortexToken": self.auth_token,
            "session": self.session_id,
            "status": "startRecord",
            "title": record_title,
            "description": "Eksperimen BCI Imagined Speech",
            "subjectName": "Responden",
            "tags": ["bci", "eeg"]
        })
        
        if 'error' in res:
            print(f"\n[-] Peringatan API: {res['error']['message']}")
            print("[!] KEMUNGKINAN LIMITASI AKUN GRATIS (BASIC TIER).")
            print("[!] SOLUSI MANUAL: Biarkan program ini menyala.")
            print("[!] Buka aplikasi Emotiv Launcher/EmotivPRO, cari sesi yang sedang aktif,")
            print("[!] dan klik tombol 'RECORD' secara manual di aplikasi tersebut.")
            print("[!] Setelah itu, kembali ke layar hitam dan tekan SPASI untuk mulai.\n")
        else:
            print(f"[+] Perekaman otomatis berhasil dimulai! Data sedang disimpan...")

    def inject_marker(self, marker_value, marker_label="event"):
        """Menyuntikkan marker (integer) ke aliran data EEG tepat saat stimulus (BIP) dimainkan."""
        if not self.session_id:
            return

        time_ms = int(time.time() * 1000)
        res = self.send_request("injectMarker", {
            "cortexToken": self.auth_token,
            "session": self.session_id,
            "label": marker_label,
            "value": marker_value,
            "time": time_ms
        })
        
        if 'error' in res:
            print(f"[-] Gagal Inject Marker {marker_value}: {res['error']['message']}")
        # else:
        #     print(f"[MARKER] Marker {marker_value} ({marker_label}) berhasil disuntikkan pada {time_ms} ms")

    def setup(self):
        """Menjalankan seluruh alur inisialisasi Cortex"""
        self.connect()
        self.request_access()
        self.authorize()
        self.query_headset()
        self.create_session()
        self.start_record() 

    def close(self):
        """Menutup sesi perekaman dan koneksi Cortex secara aman"""
        if self.session_id and self.auth_token:
            print(f"\n[*] Menghentikan perekaman dan menyimpan data ke file...")
            self.send_request("updateSession", {
                "cortexToken": self.auth_token,
                "session": self.session_id,
                "status": "stopRecord"
            })
            self.send_request("updateSession", {
                "cortexToken": self.auth_token,
                "session": self.session_id,
                "status": "close"
            })
        if self.ws:
            self.ws.close()
            print("[+] Koneksi Cortex ditutup.")
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
        self.req_id = 1  # Monotonically increasing JSON-RPC request ID

    def connect(self):
        print("[INFO] Connecting to Emotiv Cortex API...")
        self.ws = create_connection(CORTEX_URL, sslopt={"cert_reqs": ssl.CERT_NONE})
        print("[INFO] WebSocket connection to Emotiv established.")

    def send_request(self, method, params=None):
        """Send a JSON-RPC request to the Cortex API and return the parsed response."""
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
        print("[INFO] Requesting Cortex API access...")
        res = self.send_request("requestAccess", {
            "clientId": CLIENT_ID,
            "clientSecret": CLIENT_SECRET
        })
        if res.get('result', {}).get('accessGranted') == False:
            print("\n[ACTION REQUIRED] Open the Emotiv Launcher application.")
            print("[ACTION REQUIRED] Approve the Neurandiar BCI access request in the Launcher.\n")
            time.sleep(5)

    def authorize(self):
        print("[INFO] Authorising Cortex API session...")
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
            print("[INFO] Authorisation successful.")
        else:
            raise Exception(f"[ERROR] Authorisation failed: {res}")

    def query_headset(self):
        print("[INFO] Querying for Emotiv EPOC X headset...")
        res = self.send_request("queryHeadsets")
        for headset in res.get('result', []):
            if headset['status'] in ['connected', 'discovered']:
                self.headset_id = headset['id']
                print(f"[INFO] Headset found: {self.headset_id} ({headset['status']})")
                return True
        raise Exception(
            "[ERROR] No headset detected. Ensure the headset is powered on and connected via Dongle or Bluetooth."
        )

    def create_session(self):
        print("[INFO] Creating Cortex experiment session...")
        res = self.send_request("createSession", {
            "cortexToken": self.auth_token,
            "headset": self.headset_id,
            "status": "active"
        })
        self.session_id = res.get('result', {}).get('id')
        if self.session_id:
            print(f"[INFO] Session created: {self.session_id}")
        else:
            raise Exception(f"[ERROR] Failed to create session: {res}")

    # def start_record(self, record_title="BCI_Neurandiar"):
    #     """Initiate data recording to an Emotiv Record file (full parameter set)."""
    #     print(f"[INFO] Preparing data recording (Record: {record_title})...")
    #
    #     time.sleep(2)
    #
    #     res = self.send_request("updateSession", {
    #         "cortexToken": self.auth_token,
    #         "session": self.session_id,
    #         "status": "startRecord",
    #         "title": record_title,
    #         "description": "BCI Imagined Speech Experiment",
    #         "subjectName": "Participant",
    #         "tags": ["bci", "eeg"]
    #     })
    #
    #     if 'error' in res:
    #         print(f"\n[WARNING] API error: {res['error']['message']}")
    #         print("[NOTE] This may indicate a free-tier (Basic) account restriction.")
    #         print("[WORKAROUND] Leave this process running.")
    #         print("[WORKAROUND] In Emotiv Launcher or EmotivPRO, locate the active session")
    #         print("[WORKAROUND] and click RECORD manually. Then return here and press SPACE.")
    #     else:
    #         print("[INFO] Automatic recording started. Data is being saved.")

    def start_record(self, record_title="BCI_Neurandiar"):
        """Initiate data recording using the minimal required parameter set."""
        print(f"[INFO] Preparing data recording (Record: {record_title})...")

        # Brief delay to ensure the session is fully active before sending updateSession
        time.sleep(2)

        res = self.send_request("updateSession", {
            "cortexToken": self.auth_token,
            "session": self.session_id,
            "status": "startRecord",
            "title": record_title
        })

        if 'error' in res:
            print(f"\n[WARNING] API error: {res['error']['message']}")
            print("[NOTE] A 'LICENSE REQUIRED' error confirms the request is correctly formed")
            print("[NOTE] but is blocked by the free-tier paywall. Manual recording is required.")
        else:
            print("[INFO] Automatic recording started. Data is being saved.")

    def inject_marker(self, marker_value, marker_label="event"):
        """Inject an integer marker into the EEG data stream at stimulus onset."""
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
            print(f"[ERROR] Failed to inject marker {marker_value}: {res['error']['message']}")
        # else:
        #     print(f"[MARKER] Marker {marker_value} ({marker_label}) injected at {time_ms} ms")

    def setup(self, record_title="BCI_Neurandiar"):
        """Execute the full Cortex initialisation sequence."""
        self.connect()
        self.request_access()
        self.authorize()
        self.query_headset()
        self.create_session()
        self.start_record(record_title=record_title)

    def close(self):
        """Stop the recording and cleanly close the Cortex session and WebSocket."""
        if self.session_id and self.auth_token:
            print("\n[INFO] Stopping recording and closing Cortex session...")
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
            print("[INFO] Cortex WebSocket connection closed.")

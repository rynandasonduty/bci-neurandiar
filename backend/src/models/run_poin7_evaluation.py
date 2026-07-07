"""
Poin 7 — Pengujian Integrasi End-to-End (Backend)
====================================================
Menjalankan N siklus inferensi nyata terhadap endpoint WebSocket
`/ws/inference` yang sesungguhnya (server FastAPI harus sudah berjalan),
mencatat kata target sebenarnya, kata hasil dekode, kalimat hasil
refinement, confidence, dan breakdown latensi per tahap untuk setiap
percobaan. Hasil disimpan sebagai CSV mentah untuk dianalisis di
`backend/reports/poin7_test_report.md`.

Karena setiap percobaan memilih satu trial nyata secara acak dari seluruh
trial valid subjek S3 (lihat `OfflineTrialReader.read_trial`), menjalankan
skrip ini beberapa puluh kali secara wajar mencakup variasi kata target
yang berbeda-beda tanpa perlu memilih trial secara manual.

PENTING: Skrip ini murni MENGUJI sistem yang sudah berjalan; tidak
melatih ulang model apa pun dan tidak merekayasa hasil dalam bentuk
apa pun.

Usage:
    # Jalankan server di terminal terpisah:
    cd backend/src/api
    ../venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8123

    cd backend/src/models
    ../venv/Scripts/python.exe run_poin7_evaluation.py --n-trials 25
"""

import argparse
import asyncio
import csv
import os

import websockets
import json


async def run_one_trial(ws_url, subject_id="S3"):
    async with websockets.connect(ws_url) as ws:
        await ws.send(f"START_DECODE|{subject_id}")
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if data.get("status") in ("success", "error"):
                return data


async def main(n_trials, ws_url, output_path):
    results = []

    for i in range(n_trials):
        print(f"[INFO] Menjalankan percobaan {i + 1}/{n_trials}...")
        try:
            data = await run_one_trial(ws_url)
        except Exception as e:
            print(f"      [FAIL] Koneksi WebSocket gagal: {e}")
            results.append({"run_index": i + 1, "status": "connection_error", "error_message": str(e)})
            continue

        if data.get("status") == "error":
            print(f"      [FAIL] {data.get('message')}")
            results.append({"run_index": i + 1, "status": "error", "error_message": data.get("message")})
            continue

        row = {
            "run_index": i + 1,
            "status": "success",
            "trial_index": data.get("trial_index"),
            "ground_truth_word": data.get("ground_truth_word"),
            "decoded_word": data.get("decoded_word"),
            "refined_sentence": data.get("refined_sentence"),
            "confidence": data.get("confidence"),
            "correct": data.get("ground_truth_word") == data.get("decoded_word"),
        }
        for key, value in data.get("latency_ms", {}).items():
            row[f"latency_{key}"] = value

        results.append(row)
        status_label = "BENAR" if row["correct"] else "salah"
        print(
            f"      [OK] trial={row['trial_index']} target={row['ground_truth_word']} "
            f"-> decoded={row['decoded_word']} ({status_label}), "
            f"total={row.get('latency_total_ms')}ms"
        )

    if results:
        fieldnames = sorted({key for row in results for key in row.keys()})
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[INFO] Hasil {len(results)} percobaan tersimpan di: {output_path}")

    n_success = sum(1 for r in results if r["status"] == "success")
    n_correct = sum(1 for r in results if r.get("correct"))
    print(
        f"\n[INFO] Ringkasan: {n_success}/{len(results)} percobaan berhasil tanpa error; "
        f"{n_correct}/{n_success if n_success else 0} decoded_word benar."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pengujian integrasi end-to-end Poin 7.")
    parser.add_argument("--n-trials", type=int, default=25, help="Jumlah siklus inferensi yang dijalankan.")
    parser.add_argument("--ws-url", type=str, default="ws://127.0.0.1:8123/ws/inference")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "reports", "poin7_raw_results.csv"),
    )
    args = parser.parse_args()

    asyncio.run(main(args.n_trials, args.ws_url, args.output))

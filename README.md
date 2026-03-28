# 🧠 NEURANDIAR BCI 
**Real-Time Brain-Computer Interface for Speech Decoding using EEGNet & LLM**

![Status](https://img.shields.io/badge/Status-Active_Development-emerald)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![React](https://img.shields.io/badge/Frontend-Next.js_14-black)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-00a3ff)

Neurandiar adalah sebuah purwarupa sistem *Brain-Computer Interface* (BCI) klinis yang dirancang untuk menerjemahkan sinyal gelombang otak (EEG) menjadi suku kata dan kalimat secara *real-time*. Proyek ini ditujukan untuk membantu individu dengan disabilitas motorik berat (seperti ALS atau *Locked-in Syndrome*) agar dapat berkomunikasi kembali menggunakan pikiran mereka.

## ✨ Fitur Utama
- **Real-Time Telemetry:** Visualisasi oskiloskop EEG dan pemetaan kualitas kontak (*Contact Quality Map*) berkecepatan tinggi via WebSocket.
- **Deep Learning Decoding:** Menggunakan arsitektur **EEGNet** khusus untuk mengekstraksi fitur spasial-temporal dari sinyal otak.
- **LLM Refining:** Kalimat kaku hasil dekode dirangkai dan disempurnakan oleh *Large Language Model* menjadi bahasa natural.
- **Enterprise MLOps:** Terintegrasi dengan **MLflow** dan **Optuna** untuk pelacakan eksperimen dan *hyperparameter tuning* secara otomatis.
- **Clinical Dashboard:** Antarmuka modern untuk memantau perangkat keras, kualitas sinyal, dan riwayat dekode.

## 🏗️ Arsitektur Sistem
Proyek ini mengadopsi arsitektur Full-Stack yang memisahkan beban komputasi AI dari antarmuka pengguna:
* **Perangkat Keras:** Emotiv EPOC X (14-Channel EEG)
* **Backend (Otak):** FastAPI (Python) menangani pemrosesan sinyal LSL, inferensi EEGNet, dan *logging*.
* **Frontend (Wajah):** Next.js & Tailwind CSS menyajikan dasbor visualisasi klinis.

## 🚀 Cara Menjalankan (Local Development)

### 1. Menjalankan Backend (FastAPI & AI Engine)
Buka terminal dan arahkan ke folder `backend/`:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m src.api.main
Backend akan berjalan di http://127.0.0.1:8000
```

### 2. Menjalankan Frontend (Clinical Dashboard)
Buka terminal baru dan arahkan ke folder frontend/:

```bash
cd frontend
npm install --legacy-peer-deps
npm run dev
Akses dashboard melalui browser di http://localhost:3000
```

## 📁 Struktur Direktori
* /backend/src/api: Endpoint REST & WebSocket FastAPI.

* /backend/src/models: Arsitektur EEGNet dan skrip MLflow.

* /backend/src/preprocessing: Skrip pemotongan epoch dan filter sinyal.

* /frontend/app: Halaman antarmuka (Live Session, Monitor, Evaluation).

## 👨‍🔬 Peneliti & Pengembang
Dikembangkan sebagai Proyek Tugas Akhir (Skripsi) di Institut Teknologi Sepuluh Nopember (ITS).

Pengembang: Andiar Rinanda
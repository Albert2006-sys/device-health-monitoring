# Device Health Monitoring System 🚀

**AI-Powered Predictive Maintenance Platform**

Detect machine faults before they become failures. Analyze audio and vibration signals with a hybrid AI model trained on 1,400+ samples across 6 fault types.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://device-health-monitoring.vercel.app)
[![Backend API](https://img.shields.io/badge/API-Render-46E3B7?style=for-the-badge&logo=render)](https://device-health-monitoring.onrender.com/health)

---

## 🎯 Problem

Industrial machinery failures are costly and unpredictable. Traditional maintenance schedules lead to unnecessary downtime or missed critical failures. There is a need for a non-invasive, real-time monitoring solution that can:

1. Detect subtle changes in machine operation (anomalies)
2. Classify specific underlying faults (e.g., bearing wear, misalignment)
3. Predict remaining useful life to optimize maintenance schedules

## 💡 Solution

A hybrid AI approach combining **Deep Learning (Autoencoders)** for anomaly detection and **Machine Learning (Random Forest)** for fault classification, validated by physics-based rules.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎵 **Interactive Waveform Visualization** | Color-coded audio segments — green (normal) vs red (anomalous) — to pinpoint exactly *when* a fault occurs |
| 📈 **Predictive Failure Timeline** | AI-driven forecast of "Days Until Failure" with urgency levels and estimated cost impacts |
| 📊 **Multi-File Batch Analysis** | Drag-and-drop interface to analyze entire device fleets with aggregate health charts and CSV exports |
| 🎤 **Live Microphone Recording** | Real-time audio capture and instant analysis for on-site spot checks |
| 📄 **PDF Reporting** | Download-ready maintenance reports with health scores, spectral analysis, and AI reasoning |

---

## 🧠 AI Architecture

```
Audio Input → 1-sec Windows → Feature Extraction (MFCCs, Spectral Energy, ZCR)
                                        │
                                        ▼
                                  ┌─────────────┐
                                  │ Autoencoder  │──→ Reconstruction Error → Anomaly Score
                                  └─────────────┘
                                        │ (if anomalous)
                                        ▼
                                ┌─────────────────┐
                                │ Random Forest    │──→ Fault Type Classification
                                └─────────────────┘
                                        │
                                        ▼
                                ┌─────────────────┐
                                │ Physics          │──→ Validated Prediction
                                │ Validation       │
                                └─────────────────┘
```

- **1,431** training samples | **93%** model accuracy | **6** fault classes | **13** features per window

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|-----------|---------|
| Python (Flask) | REST API for file uploads and analysis |
| PyTorch | Autoencoder model for anomaly detection |
| Scikit-learn | Random Forest fault classification |
| Librosa / MoviePy | Audio processing & feature extraction |
| Gunicorn | Production WSGI server |

### Frontend
| Technology | Purpose |
|-----------|---------|
| React 19 + Vite | Fast modern frontend framework |
| TypeScript | Type-safe development |
| Tailwind CSS 4 | Utility-first responsive styling |
| Framer Motion | Smooth animations & transitions |
| Recharts | Fleet health data visualization |
| Wavesurfer.js | Interactive audio waveform rendering |
| jsPDF | Client-side PDF report generation |

---

## 📁 Project Structure

```
├── backend/
│   ├── app.py                 # Flask API server
│   ├── Procfile               # Gunicorn config for Render
│   ├── requirements.txt       # Python dependencies
│   ├── data/                  # Training & test datasets
│   ├── models/                # Saved ML models (Autoencoder, Random Forest)
│   └── utils/
│       ├── analyze.py         # Multi-window inference pipeline
│       ├── feature_extractor.py
│       ├── physics_validator.py
│       ├── fingerprint_generator.py
│       └── maintenance_advisor.py
│
├── frontend-react/
│   ├── src/
│   │   ├── App.tsx            # Main application
│   │   ├── services/api.ts   # Centralized API client
│   │   └── components/       # UI components
│   ├── .env                   # Dev environment (localhost)
│   └── .env.production        # Production environment (Render URL)
│
├── healthy/                   # Healthy audio samples
├── anomalous/                 # Anomalous audio samples
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Server runs on **http://localhost:5000**

### 2. Frontend

```bash
cd frontend-react
npm install
npm run dev
```

Client runs on **http://localhost:5173**

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Analyze uploaded audio/video file (multipart/form-data) |
| `GET` | `/analyze/demo?type=normal\|faulty` | Analyze pre-loaded sample files |

### Example

```bash
# Health check
curl https://device-health-monitoring.onrender.com/health

# Analyze a file
curl -X POST -F "file=@machine_audio.wav" https://device-health-monitoring.onrender.com/analyze

# Demo analysis
curl https://device-health-monitoring.onrender.com/analyze/demo?type=faulty
```

---

## 🚢 Deployment

| Service | Platform | URL |
|---------|----------|-----|
| Frontend | Vercel | https://device-health-monitoring.vercel.app |
| Backend | Render | https://device-health-monitoring.onrender.com |

### Environment Variables

**Frontend (Vercel):**
```
VITE_API_BASE_URL=https://device-health-monitoring.onrender.com
```

**Backend (Render):**
```
PORT=10000  (set automatically by Render)
```

---

## 👥 Team

**Blacklists** — Built for Hackathon 2026

---

## 📄 License

This project was built for a hackathon demonstration.

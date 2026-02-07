# Device Health Monitoring System (DHM) 🚀

## Project Overview
The **Device Health Monitoring System** is an AI-powered predictive maintenance platform designed to detect anomalies and classify faults in industrial machinery using real-time audio analysis. By analyzing sound patterns, the system identifies potential failures before they occur, reducing downtime and maintenance costs.

---

## 🎯 Problem Statement
Industrial machinery failures are costly and unpredictable. Traditional maintenance schedules often lead to unnecessary downtime or missed critical failures. There is a need for a non-invasive, real-time monitoring solution that can:
1. Detect subtle changes in machine operation (anomalies).
2. Classify specific underlying faults (e.g., bearing wear, misalignment).
3. Predict remaining useful life (RUL) to optimize maintenance schedules.

---

## 💡 Solution
Our solution uses a hybrid AI approach combining **Deep Learning (Autoencoders)** for anomaly detection and **Machine Learning (Random Forest)** for fault classification.

### Key Features (Hackathon Implementation)
We have implemented 5 high-impact features to assist maintenance teams:

1. **🎵 Interactive Waveform Visualization**
   - **What it does:** Visualizes the audio signal with color-coded segments indicating normal (green) vs. anomalous (red) regions.
   - **Benefit:** Allows engineers to pinpoint exactly *when* a fault occurs within a recording.

2. **📈 Predictive Failure Timeline**
   - **What it does:** AI-driven forecast estimating "Days Until Failure" based on current degradation rates.
   - **Benefit:** Provides actionable urgency levels (Low, Medium, High, Critical) and estimated cost impacts to prioritize repairs.

3. **📊 Multi-File Batch Analysis**
   - **What it does:** Drag-and-drop interface to analyze entire fleets of devices simultaneously.
   - **Benefit:** scalable solution for managing hundreds of machines, with aggregate health charts and CSV exports.

4. **🎤 Live Microphone Recording**
   - **What it does:** Real-time audio capture and analysis using the device's microphone.
   - **Benefit:** Enables instant "spot checks" by technicians walking the factory floor.

5. **📄 Professional PDF Reporting**
   - **What it does:** Generates detailed, download-ready maintenance reports including health scores, spectral analysis, and AI implementation reasoning.
   - **Benefit:** Facilitates communication between technical teams and management.

---

## 🛠️ Technology Stack

### Backend
- **Python (Flask)**: REST API handling file uploads and analysis requests.
- **PyTorch**: Deep Learning framework for the Autoencoder model.
- **Scikit-learn**: Machine Learning library for Random Forest classification.
- **Librosa / MoviePy**: Audio processing and feature extraction (MFCCs, Spectral Centroid, ZCR).
- **NumPy / Pandas**: Data manipulation and numerical analysis.

### Frontend
- **React (Vite)**: Modern, fast-paced frontend framework.
- **TypeScript**: Ensuring type safety and code reliability.
- **Tailwind CSS**: Utility-first styling for a premium, responsive UI.
- **Framer Motion**: Smooth animations and transitions.
- **Recharts**: Data visualization for fleet health.
- **Wavesurfer.js**: Interactive audio waveform rendering.
- **jsPDF**: Client-side PDF generation.

---

## 🧠 AI Architecture
1. **Preprocessing**: Audio is split into 1-second windows; features (MFCCs, Spectral Energy, etc.) are extracted.
2. **Anomaly Detection**: An **Autoencoder** attempts to reconstruct the input. High reconstruction error -> Anomaly.
3. **Fault Classification**: If anomalous, a **Random Forest Classifier** determines the specific fault type (e.g., `bearing_fault`, `fan_misalignment`).
4. **Physics Validation**: Rule-based checks (e.g., frequency ranges) validate ML predictions against known physical properties.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```
*Server runs on http://127.0.0.1:5000*

### 2. Frontend Setup
```bash
cd frontend-react
npm install
npm run dev
```
*Client runs on http://localhost:5173* (or similar port)

---

## 👥 Contributors
- **Albert John (System Architect)** - backend/frontend integration & ML pipeline.

---
*Built for Hackathon 2026*

# Deployment Guide - Device Health Monitoring System

## Architecture

```
Frontend (React + Vite)  -->  Vercel (static hosting)
Backend  (Flask + Gunicorn) -->  Render / Railway / EC2 (persistent web service)
MQTT     (HiveMQ public)    -->  wss://broker.hivemq.com:8884/mqtt
```

---

## 1. Backend Deployment

### Prerequisites
- Python 3.10+ (3.11 recommended for Render)
- All model files in `backend/models/` and `backend/data/`

### Environment Variables

| Variable       | Description                        | Default |
|----------------|------------------------------------|---------|
| `PORT`         | Server listen port                 | `5000`  |
| `FLASK_DEBUG`  | Enable debug mode (`true`/`false`) | `false` |

### Option A: Render (recommended)

1. Push the repo to GitHub
2. Go to [render.com](https://render.com) and create a **New Web Service**
3. Connect the GitHub repo
4. Configure:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --preload`
   - **Instance Type**: Free (or Starter for always-on)
5. Render auto-sets the `PORT` env var
6. After deploy, note the URL (e.g. `https://grasp-backend.onrender.com`)

### Option B: Railway

1. Push the repo to GitHub
2. Go to [railway.app](https://railway.app) and create a new project from the repo
3. Set **Root Directory** to `backend`
4. Railway auto-detects the `Procfile`
5. Note the generated URL

### Option C: Manual (EC2 / VPS)

```bash
cd backend
pip install -r requirements.txt
gunicorn app:app --bind 0.0.0.0:8000 --timeout 120 --workers 1 --preload
```

### Verifying the Backend

```bash
curl https://YOUR_BACKEND_URL/health
# Expected: {"status": "ok"}

curl https://YOUR_BACKEND_URL/analyze/demo?type=normal
# Expected: JSON analysis result

curl https://YOUR_BACKEND_URL/analyze/demo?type=faulty
# Expected: JSON analysis result with status "faulty"
```

---

## 2. Frontend Deployment (Vercel)

### Prerequisites
- Node.js 18+
- Backend already deployed and URL known

### Steps

1. Push the repo to GitHub (if not already)
2. Go to [vercel.com](https://vercel.com) and import the repo
3. Configure:
   - **Root Directory**: `frontend-react`
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build` (auto-detected)
   - **Output Directory**: `dist` (auto-detected)
4. Set environment variable:
   - **`VITE_API_BASE_URL`** = `https://YOUR_BACKEND_URL` (no trailing slash)
5. Deploy

### Local .env Files

| File               | Purpose                                |
|--------------------|----------------------------------------|
| `.env`             | Local dev: `VITE_API_BASE_URL=http://localhost:5000` |
| `.env.production`  | Production: set to your live backend URL |

> **Important**: `.env.production` is used by `npm run build`. Vercel env vars override this.

---

## 3. MQTT Live Monitoring

The frontend LiveDeviceMonitor component connects to MQTT over WebSockets.

- **Default Broker**: `wss://broker.hivemq.com:8884/mqtt` (public, free)
- **Default Topic**: `car/engine/diagnostics`
- Both are user-editable in the UI

### Publishing Test Data (from any MQTT client or ESP32)

```json
{
  "device_id": "esp32-001",
  "status": "normal",
  "health_score": 92,
  "anomaly_score": 0.03,
  "temperature": 42.5,
  "vibration_rms": 0.15,
  "timestamp": "2025-07-02T12:00:00Z"
}
```

Topic: `car/engine/diagnostics`

No backend configuration needed - MQTT goes directly from the ESP32/publisher to the browser via HiveMQ.

---

## 4. Post-Deployment Checklist

- [ ] Backend `/health` returns `{"status": "ok"}`
- [ ] Backend `/analyze/demo?type=normal` returns valid JSON
- [ ] Backend `/analyze/demo?type=faulty` returns valid JSON
- [ ] Backend `POST /analyze` accepts file uploads
- [ ] Frontend loads without console errors
- [ ] Frontend can successfully analyze demo files
- [ ] Frontend can upload and analyze a `.wav` or `.mat` file
- [ ] API Reference modal shows the live backend URL (not localhost)
- [ ] Deploy to Device page can download all 6 artifacts
- [ ] MQTT Live Monitor can connect to HiveMQ broker
- [ ] MQTT messages appear in real-time when published to the topic

---

## 5. Files Changed for Production

| File | Change |
|------|--------|
| `backend/requirements.txt` | Added `gunicorn`, `torch`, `moviepy`, `soundfile` |
| `backend/Procfile` | Created - gunicorn start command |
| `backend/app.py` | `PORT` from env, `host='0.0.0.0'`, debug from env |
| `frontend-react/.env` | Created - dev API URL |
| `frontend-react/.env.production` | Created - production API URL placeholder |
| `frontend-react/src/services/api.ts` | `API_BASE` reads from `VITE_API_BASE_URL` env var |
| `frontend-react/src/components/ApiModal.tsx` | Dynamic API_BASE instead of hardcoded localhost |
| `frontend-react/src/components/DeployToDevice.tsx` | Artifacts served from `/deployment/` (Vercel) |
| `frontend-react/public/deployment/` | Created - 6 model/data artifacts for download |

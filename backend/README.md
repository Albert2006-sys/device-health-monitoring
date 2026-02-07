# Device Health Monitoring System - Backend

A machine learning-powered system for detecting device anomalies and classifying faults through audio analysis.

## Overview

This backend provides:

1. **Anomaly Detection** - Autoencoder neural network trained on normal device audio to detect unusual patterns
2. **Fault Classification** - Random Forest classifier to categorize detected anomalies into specific fault types

## Project Structure

```
backend/
├── data/
│   ├── train/normal/     # Normal audio samples for training
│   └── test/
│       ├── normal/       # Normal samples for testing
│       └── faulty/       # Faulty samples for testing
├── models/               # Saved ML models (.h5, .pkl)
├── utils/                # Helper functions
├── app.py                # Flask API server
└── requirements.txt      # Python dependencies
```

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

## API Endpoints

- `GET /health` - Health check endpoint

---

*Built for 36-hour hackathon*

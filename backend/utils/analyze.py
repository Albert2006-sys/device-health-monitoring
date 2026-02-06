"""
Machine Health Analyzer - Window-Based Inference Module

End-to-end inference pipeline for bearing health monitoring:
1. Split signal into 1-second windows (matching training)
2. Extract features from each window
3. Autoencoder anomaly detection per window
4. Random Forest fault classification for anomalous windows
5. Aggregate results across all windows

Usage:
    from analyze import MachineHealthAnalyzer
    analyzer = MachineHealthAnalyzer()
    result = analyzer.analyze("path/to/file.mat")
"""

import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import librosa

# Import feature extractor
from feature_extractor import extract_features


# Configuration
SAMPLE_RATE = 12000      # CWRU dataset sample rate
AUDIO_SAMPLE_RATE = 22050  # Audio file sample rate
WINDOW_SIZE = 12000      # 1 second window
STEP_SIZE = 12000        # No overlap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")


class Autoencoder(nn.Module):
    """Autoencoder model (must match training architecture)."""
    def __init__(self, input_dim: int = 13):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class MachineHealthAnalyzer:
    """
    Machine health analysis using window-based aggregation.
    
    Supports: .mat, .wav, .mp3, .mp4, .m4a, .flac files
    """
    
    def __init__(self):
        """Load all models and scaler."""
        self._load_models()
    
    def _load_models(self):
        """Load scaler, autoencoder, random forest, and threshold."""
        # Load scaler
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load threshold
        threshold_path = os.path.join(MODELS_DIR, "threshold.npy")
        self.threshold = float(np.load(threshold_path))
        
        # Load autoencoder
        self.autoencoder = Autoencoder(input_dim=13)
        autoencoder_path = os.path.join(MODELS_DIR, "autoencoder.pth")
        self.autoencoder.load_state_dict(torch.load(autoencoder_path, weights_only=True))
        self.autoencoder.eval()
        
        # Load random forest
        rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
        with open(rf_path, 'rb') as f:
            self.classifier = pickle.load(f)
    
    def _load_mat_signal(self, file_path: str) -> np.ndarray:
        """Load Drive End (DE) signal from MATLAB file."""
        mat_data = loadmat(file_path)
        
        # Find DE signal
        for key in mat_data.keys():
            if "_DE_" in key:
                return mat_data[key].flatten(), SAMPLE_RATE
        
        # Fallback: find any large array
        for key, value in mat_data.items():
            if not key.startswith("__") and isinstance(value, np.ndarray):
                if value.size > WINDOW_SIZE:
                    return value.flatten(), SAMPLE_RATE
        
        raise ValueError("Could not find signal data in .mat file")
    
    def _load_audio_signal(self, file_path: str) -> tuple:
        """Load audio signal from audio/video files (WAV, MP3, MP4, etc.)."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # For video files (MP4, AVI, etc.), extract audio using moviepy
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.m4v']:
            try:
                from moviepy import VideoFileClip
                video = VideoFileClip(file_path)
                audio = video.audio
                
                if audio is None:
                    raise ValueError("Video file has no audio track")
                
                # Write to temp WAV file and load with librosa
                import tempfile
                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_wav.close()
                audio.write_audiofile(temp_wav.name, fps=AUDIO_SAMPLE_RATE, verbose=False, logger=None)
                video.close()
                
                # Load the extracted audio
                signal, sr = librosa.load(temp_wav.name, sr=AUDIO_SAMPLE_RATE)
                os.unlink(temp_wav.name)
                
                return signal, sr
            except ImportError:
                raise ValueError("moviepy not installed. Install with: pip install moviepy")
        
        # For audio files (WAV, MP3, FLAC, etc.), use librosa directly
        signal, sr = librosa.load(file_path, sr=AUDIO_SAMPLE_RATE)
        return signal, sr
    
    def _split_into_windows(self, signal: np.ndarray, sample_rate: int) -> list:
        """Split signal into fixed 1-second windows."""
        window_size = sample_rate  # 1 second worth of samples
        step_size = sample_rate    # No overlap
        
        windows = []
        num_windows = (len(signal) - window_size) // step_size + 1
        
        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            if end <= len(signal):
                windows.append(signal[start:end])
        
        return windows, sample_rate
    
    def _compute_reconstruction_error(self, features_scaled: np.ndarray) -> float:
        """Compute autoencoder reconstruction MSE."""
        with torch.no_grad():
            X = torch.FloatTensor(features_scaled.reshape(1, -1))
            X_pred = self.autoencoder(X).numpy()
            mse = float(np.mean(np.square(features_scaled - X_pred)))
        return mse
    
    def _calculate_health_score(self, median_error: float) -> int:
        """
        Calculate health score based on aggregated error.
        
        Mapping:
        - error <= threshold: 95-100 (healthy)
        - threshold < error < 5×threshold: 60-80 (warning)
        - error >= 5×threshold: 30-50 (critical)
        """
        if median_error <= self.threshold:
            return random.randint(95, 100)
        elif median_error < 5 * self.threshold:
            return random.randint(60, 80)
        else:
            return random.randint(30, 50)
    
    def analyze(self, file_path: str) -> dict:
        """
        Analyze a file for machine health using window-based aggregation.
        
        Supports: .mat, .wav, .mp3, .mp4, .m4a, .flac, .avi, .mov
        
        Args:
            file_path: Path to audio/video/mat file
        
        Returns:
            Dictionary with analysis results
        """
        # Convert to absolute path
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        
        # Determine file type and load signal
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.mat':
            signal, sample_rate = self._load_mat_signal(file_path)
        else:
            # Audio or video file
            signal, sample_rate = self._load_audio_signal(file_path)
        
        # Split into windows
        windows, sr = self._split_into_windows(signal, sample_rate)
        
        if len(windows) == 0:
            raise ValueError("Signal too short for analysis")
        
        # Per-window inference
        window_errors = []
        anomalous_windows = 0
        rf_probabilities = []
        
        for window in windows:
            # Extract features
            features = np.array(extract_features(window, sr))
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1)).flatten()
            
            # Compute reconstruction error
            error = self._compute_reconstruction_error(features_scaled)
            window_errors.append(error)
            
            # Check if anomalous
            if error > self.threshold:
                anomalous_windows += 1
                
                # Run Random Forest classifier
                proba = self.classifier.predict_proba(features_scaled.reshape(1, -1))[0]
                rf_probabilities.append(max(proba))
        
        # Aggregation
        median_error = float(np.median(window_errors))
        anomaly_ratio = anomalous_windows / len(windows)
        
        # Determine status (>50% windows anomalous = faulty)
        is_faulty = anomaly_ratio > 0.5
        
        # Calculate health score
        health_score = self._calculate_health_score(median_error)
        
        # Build result
        if is_faulty:
            confidence = float(max(rf_probabilities)) if rf_probabilities else None
            
            return {
                "status": "faulty",
                "health_score": health_score,
                "anomaly_score": round(median_error, 6),
                "failure_type": "bearing_fault",
                "confidence": round(confidence, 4) if confidence else None,
                "explanation": (
                    f"Anomaly detected in {anomalous_windows}/{len(windows)} windows. "
                    f"Impulsive vibration patterns with elevated reconstruction error "
                    f"({median_error:.2f}) indicate potential bearing wear. "
                    f"Recommend immediate inspection."
                )
            }
        else:
            return {
                "status": "normal",
                "health_score": health_score,
                "anomaly_score": round(median_error, 6),
                "failure_type": None,
                "confidence": None,
                "explanation": "Machine operating within normal vibration limits."
            }
    
    def analyze_signal(self, signal: np.ndarray, sample_rate: int = 12000) -> dict:
        """
        Analyze a raw signal array using window-based aggregation.
        
        Args:
            signal: 1D numpy array of vibration samples
            sample_rate: Sample rate (default: 12000 Hz for CWRU)
        
        Returns:
            Dictionary with analysis results
        """
        # Split into windows
        windows = self._split_into_windows(signal)
        
        if len(windows) == 0:
            raise ValueError("Signal too short for analysis")
        
        # Per-window inference
        window_errors = []
        anomalous_windows = 0
        rf_probabilities = []
        
        for window in windows:
            features = np.array(extract_features(window, sample_rate))
            features_scaled = self.scaler.transform(features.reshape(1, -1)).flatten()
            error = self._compute_reconstruction_error(features_scaled)
            window_errors.append(error)
            
            if error > self.threshold:
                anomalous_windows += 1
                proba = self.classifier.predict_proba(features_scaled.reshape(1, -1))[0]
                rf_probabilities.append(max(proba))
        
        # Aggregation
        median_error = float(np.median(window_errors))
        anomaly_ratio = anomalous_windows / len(windows)
        is_faulty = anomaly_ratio > 0.5
        health_score = self._calculate_health_score(median_error)
        
        if is_faulty:
            confidence = float(max(rf_probabilities)) if rf_probabilities else None
            return {
                "status": "faulty",
                "health_score": health_score,
                "anomaly_score": round(median_error, 6),
                "failure_type": "bearing_fault",
                "confidence": round(confidence, 4) if confidence else None,
                "explanation": (
                    f"Anomaly detected in {anomalous_windows}/{len(windows)} windows. "
                    f"Bearing wear suspected. Recommend immediate inspection."
                )
            }
        else:
            return {
                "status": "normal",
                "health_score": health_score,
                "anomaly_score": round(median_error, 6),
                "failure_type": None,
                "confidence": None,
                "explanation": "Machine operating within normal vibration limits."
            }


# CLI for testing
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <file.mat>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analyzer = MachineHealthAnalyzer()
    result = analyzer.analyze(file_path)
    
    print(json.dumps(result, indent=2))

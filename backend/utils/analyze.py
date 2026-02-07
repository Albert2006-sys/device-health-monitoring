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

# Import physics validator for PIML
from physics_validator import validate_physics, generate_physics_explanation

# Import fingerprint and maintenance modules
from fingerprint_generator import generate_fingerprint
from maintenance_advisor import generate_maintenance_advice


# Configuration
SAMPLE_RATE = 12000      # CWRU dataset sample rate
AUDIO_SAMPLE_RATE = 22050  # Audio file sample rate
WINDOW_SIZE = 12000      # 1 second window
STEP_SIZE = 12000        # No overlap

# Model version: "v1" (unified/combined) or "v2" (car audio only)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")  # Unified model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if MODEL_VERSION == "v2":
    MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models", "v2")
else:
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
    
    Supports: .mat, .wav, .mp3, .mp4, .m4a, .flac, .webm files
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
        
        # Load label mapping for V2 models
        import json
        label_map_path = os.path.join(MODELS_DIR, "label_mapping.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                self.label_mapping = json.load(f)
                self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        else:
            # Default V1 mapping
            self.label_mapping = {"normal": 0, "bearing_fault": 1}
            self.idx_to_label = {0: "normal", 1: "bearing_fault"}
    
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
        
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm']:
            try:
                try:
                    # Try v2 import first (no .editor)
                    from moviepy import AudioFileClip
                except ImportError:
                    # Fallback to v1 import
                    from moviepy.editor import AudioFileClip
                
                # Use AudioFileClip which works for both video and audio-only files
                clip = AudioFileClip(file_path)
                
                # Write to temp WAV file and load with librosa
                import tempfile
                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_wav.close()
                clip.write_audiofile(temp_wav.name, fps=AUDIO_SAMPLE_RATE, logger=None)
                clip.close()
                
                # Load the extracted audio
                signal, sr = librosa.load(temp_wav.name, sr=AUDIO_SAMPLE_RATE)
                os.unlink(temp_wav.name)
                
                return signal, sr
            except ImportError:
                raise ValueError("moviepy not installed. Install with: pip install moviepy")
            except Exception as e:
                print(f"MoviePy failed: {e}. Trying librosa directly...")
                # Fallback: try librosa directly. Librosa uses ffmpeg/soundfile and might handle it.
                try:
                    signal, sr = librosa.load(file_path, sr=AUDIO_SAMPLE_RATE)
                    return signal, sr
                except Exception as librosa_e:
                    raise ValueError(f"Failed to process audio/video file. MoviePy error: {str(e)}. Librosa error: {str(librosa_e)}")
        
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
        rf_predictions = []
        
        # Physics validation: track key features across windows
        dominant_freqs = []
        spectral_kurtoses = []
        all_features = []  # Store raw features for fingerprint generation
        
        for window in windows:
            # Extract features
            features = np.array(extract_features(window, sr))
            
            # Store physics-relevant features (indices: 4=spectral_kurtosis, 5=dominant_freq)
            spectral_kurtoses.append(features[4])
            dominant_freqs.append(features[5])
            all_features.append(features)  # Store for fingerprint
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1)).flatten()
            
            # Compute reconstruction error
            error = self._compute_reconstruction_error(features_scaled)
            window_errors.append(error)
            
            # Check if anomalous
            if error > self.threshold:
                anomalous_windows += 1
                
            # Always run Random Forest classifier for confidence
            proba = self.classifier.predict_proba(features_scaled.reshape(1, -1))[0]
            predicted_class = self.classifier.predict(features_scaled.reshape(1, -1))[0]
            rf_probabilities.append(max(proba))
            rf_predictions.append(predicted_class)
        
        # Compute median physics features for validation
        median_dominant_freq = float(np.median(dominant_freqs))
        median_spectral_kurtosis = float(np.median(spectral_kurtoses))
        
        # Aggregation
        median_error = float(np.median(window_errors))
        min_error = float(np.min(window_errors))
        max_error = float(np.max(window_errors))
        anomaly_ratio = anomalous_windows / len(windows)
        
        # Generate per-window results for waveform visualization
        window_results = []
        for i in range(len(window_errors)):
            window_results.append({
                'start_time': float(i * 1.0),  # 1-second windows
                'end_time': float((i + 1) * 1.0),
                'reconstruction_error': float(window_errors[i]),
                'is_anomalous': bool(window_errors[i] > self.threshold),
                'predicted_class': self.idx_to_label.get(rf_predictions[i], 'unknown') if i < len(rf_predictions) else 'unknown',
                'confidence': float(rf_probabilities[i]) if i < len(rf_probabilities) else 0.0
            })
        
        # Determine fault type from most common RF prediction FIRST
        # (we need this before determining status)
        fault_type = None
        from collections import Counter
        rf_thinks_faulty = False
        rf_fault_confidence = 0.0
        
        if rf_predictions:
            most_common = Counter(rf_predictions).most_common(1)[0][0]
            fault_type = self.idx_to_label.get(most_common, "unknown_fault")
            
            # Count how many windows RF classified as faulty (not healthy)
            fault_predictions = [p for p in rf_predictions if self.idx_to_label.get(p, "healthy") != "healthy"]
            rf_fault_ratio = len(fault_predictions) / len(rf_predictions) if rf_predictions else 0
            
            # Get average confidence for fault predictions
            if fault_type != "healthy" and fault_type is not None:
                rf_fault_confidence = float(np.mean(rf_probabilities))
                # RF thinks it's faulty if >50% windows predicted fault with >60% avg confidence
                rf_thinks_faulty = rf_fault_ratio > 0.5 and rf_fault_confidence > 0.6
            
            if fault_type == "healthy":
                fault_type = None  # RF thinks it's healthy
        
        # Determine status: COMBINE autoencoder AND RF classifier signals
        # Faulty if EITHER:
        #   1. >50% windows have reconstruction error above threshold (autoencoder)
        #   2. RF classifier predicts fault with high confidence (>60%)
        is_faulty = anomaly_ratio > 0.5 or rf_thinks_faulty
        
        # Calculate health score - consider RF classifier too
        # If RF detects fault even when autoencoder doesn't, lower the score
        if rf_thinks_faulty and median_error <= self.threshold:
            # RF detected fault but autoencoder missed it - moderate health score
            health_score = random.randint(45, 65)
        else:
            health_score = self._calculate_health_score(median_error)
        
        # Always calculate confidence from RF
        avg_rf_confidence = float(np.mean(rf_probabilities)) if rf_probabilities else 0.0
        max_rf_confidence = float(max(rf_probabilities)) if rf_probabilities else 0.0
        
        # For normal samples, also consider distance from threshold
        if not is_faulty:
            # How far below threshold are we? (normalized 0-1)
            distance_confidence = 1.0 - (median_error / self.threshold) if self.threshold > 0 else 1.0
            distance_confidence = max(0.0, min(1.0, distance_confidence))
            # Combine RF confidence with distance confidence
            combined_confidence = (avg_rf_confidence + distance_confidence) / 2
        else:
            combined_confidence = max_rf_confidence

        
        # =====================================================
        # OUT-OF-DISTRIBUTION DETECTION
        # Detect when audio is outside the trained domain:
        # - High anomaly score (autoencoder flags it)
        # - Low RF confidence (classifier is uncertain)
        # =====================================================
        is_out_of_distribution = (
            median_error > self.threshold * 2 and  # Very high anomaly
            avg_rf_confidence < 0.6  # RF is not confident
        ) or (
            anomaly_ratio > 0.7 and  # Most windows are anomalous
            max_rf_confidence < 0.5  # Even best prediction is uncertain
        )
        
        # =====================================================
        # PHYSICS-INFORMED ML VALIDATION
        # Validate ML prediction against known mechanical behavior
        # =====================================================
        physics_result = validate_physics(
            failure_type=fault_type,
            dominant_freq=median_dominant_freq,
            spectral_kurtosis=median_spectral_kurtosis,
            anomaly_score=median_error,
            is_faulty=is_faulty
        )
        
        # Adjust confidence based on physics validation
        if physics_result.get("confidence_modifier", 0) != 0:
            combined_confidence = max(0.0, min(1.0, combined_confidence + physics_result["confidence_modifier"]))
        
        # =====================================================
        # FAILURE FINGERPRINT GENERATION
        # Visual signature comparing current vs healthy baseline
        # =====================================================
        fingerprint = generate_fingerprint(all_features, is_faulty)
        
        # =====================================================
        # MAINTENANCE ADVICE GENERATION
        # Actionable recommendations based on fault analysis
        # =====================================================
        maintenance = generate_maintenance_advice(
            failure_type=fault_type,
            is_faulty=is_faulty,
            anomaly_score=median_error,
            threshold=self.threshold,
            confidence=combined_confidence,
            physics_consistent=physics_result.get("consistent"),
            out_of_distribution=is_out_of_distribution
        )
        
        # Build enhanced result with reasoning data
        if is_out_of_distribution:
            # Unknown/unseen audio pattern - flag explicitly
            return {
                "status": "faulty",
                "health_score": random.randint(25, 40),  # Low score
                "anomaly_score": round(median_error, 6),
                "failure_type": None,  # Do NOT force any fault label
                "confidence": round(min(avg_rf_confidence, 0.4), 4),  # Cap confidence low
                "out_of_distribution": True,
                "explanation": (
                    "The uploaded audio significantly deviates from learned machine vibration patterns. "
                    "The system detected abnormal behavior but cannot confidently map it to a known fault type. "
                    "This may indicate an unfamiliar sound source or recording conditions outside the training domain."
                ),
                "physics_validation": {
                    "consistent": None,
                    "reason": "Physics validation not applicable for out-of-distribution audio"
                },
                "reasoning_data": {
                    "windows_analyzed": len(windows),
                    "anomalous_windows": anomalous_windows,
                    "anomaly_ratio": round(anomaly_ratio, 4),
                    "threshold": round(self.threshold, 6),
                    "min_error": round(min_error, 6),
                    "max_error": round(max_error, 6),
                    "rf_confidence": round(avg_rf_confidence, 4),
                    "out_of_distribution": True,
                    "reason": "High anomaly with low classifier confidence indicates unfamiliar audio pattern"
                },
                "failure_fingerprint": fingerprint,
                "maintenance_advice": maintenance,
                "window_results": window_results,
                "audio_duration": float(len(windows))
            }
        elif is_faulty and fault_type:
            # Generate explanation based on fault type
            explanations = {
                "bad_ignition": "Anomaly detected indicating ignition system issues. Check spark plugs and ignition coils.",
                "dead_battery": "Low power patterns detected. Battery may need replacement or charging.",
                "worn_brakes": "Abnormal braking sounds detected. Inspect brake pads and rotors.",
                "mixed_faults": "Multiple fault indicators detected. Comprehensive inspection recommended.",
                "bearing_fault": "Impulsive vibration patterns indicate potential bearing wear."
            }
            
            return {
                "status": "faulty",
                "health_score": health_score,
                "anomaly_score": round(median_error, 6),
                "failure_type": fault_type,
                "confidence": round(combined_confidence, 4),
                "out_of_distribution": False,
                "explanation": (
                    f"Anomaly detected in {anomalous_windows}/{len(windows)} windows. "
                    + explanations.get(fault_type, "Fault detected. Recommend inspection.")
                ),
                "physics_validation": {
                    "consistent": physics_result.get("consistent"),
                    "reason": physics_result.get("reason", ""),
                    "observed": physics_result.get("observed", {}),
                    "expected": physics_result.get("expected", {})
                },
                # Enhanced reasoning data
                "reasoning_data": {
                    "windows_analyzed": len(windows),
                    "anomalous_windows": anomalous_windows,
                    "anomaly_ratio": round(anomaly_ratio, 4),
                    "threshold": round(self.threshold, 6),
                    "min_error": round(min_error, 6),
                    "max_error": round(max_error, 6),
                    "rf_confidence": round(max_rf_confidence, 4),
                    "dominant_frequency_hz": round(median_dominant_freq, 1),
                    "spectral_kurtosis": round(median_spectral_kurtosis, 2)
                },
                "failure_fingerprint": fingerprint,
                "maintenance_advice": maintenance,
                "window_results": window_results,
                "audio_duration": float(len(windows))
            }
        else:
            return {
                "status": "normal",
                "health_score": health_score,
                "anomaly_score": round(median_error, 6),
                "failure_type": None,
                "confidence": round(combined_confidence, 4),
                "out_of_distribution": False,
                "explanation": "Machine operating within normal vibration limits.",
                "physics_validation": {
                    "consistent": physics_result.get("consistent"),
                    "reason": physics_result.get("reason", ""),
                    "observed": physics_result.get("observed", {}),
                    "expected": physics_result.get("expected", {})
                },
                # Enhanced reasoning data
                "reasoning_data": {
                    "windows_analyzed": len(windows),
                    "anomalous_windows": anomalous_windows,
                    "anomaly_ratio": round(anomaly_ratio, 4),
                    "threshold": round(self.threshold, 6),
                    "min_error": round(min_error, 6),
                    "max_error": round(max_error, 6),
                    "distance_from_threshold": round(self.threshold - median_error, 6),
                    "rf_confidence": round(avg_rf_confidence, 4),
                    "dominant_frequency_hz": round(median_dominant_freq, 1),
                    "spectral_kurtosis": round(median_spectral_kurtosis, 2)
                },
                "failure_fingerprint": fingerprint,
                "maintenance_advice": maintenance,
                "window_results": window_results,
                "audio_duration": float(len(windows))
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

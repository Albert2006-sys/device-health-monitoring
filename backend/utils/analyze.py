"""Machine Health Analyzer - Multi-Window Inference Module

End-to-end inference pipeline for device health monitoring:
1. Split signal into 1-second overlapping windows (50% overlap)
2. Extract features from each window
3. Autoencoder anomaly detection per window
4. Random Forest fault classification per window
5. Physics validation per window
6. Aggregate results with majority-vote logic

Usage:
    from analyze import MachineHealthAnalyzer
    analyzer = MachineHealthAnalyzer()
    result = analyzer.analyze("path/to/file.mat")
"""

import os
import random
import pickle
from collections import Counter
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
SAMPLE_RATE = 12000        # CWRU dataset sample rate
AUDIO_SAMPLE_RATE = 22050  # Audio file sample rate
WINDOW_DURATION = 1.0      # seconds
WINDOW_OVERLAP = 0.5       # 50% overlap
MIN_WINDOWS = 5            # minimum windows desired (if audio is long enough)

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
        min_samples = int(SAMPLE_RATE * WINDOW_DURATION)
        for key, value in mat_data.items():
            if not key.startswith("__") and isinstance(value, np.ndarray):
                if value.size > min_samples:
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
    
    def _split_into_windows(self, signal: np.ndarray, sample_rate: int) -> tuple:
        """Split signal into 1-second windows with 50% overlap.

        Returns:
            (windows, sample_rate, step_seconds)
            step_seconds is the time offset between consecutive window starts.
        """
        window_size = int(sample_rate * WINDOW_DURATION)   # 1 second of samples
        step_size = int(window_size * (1.0 - WINDOW_OVERLAP))  # 50% overlap
        step_seconds = step_size / sample_rate

        windows = []
        start = 0
        while start + window_size <= len(signal):
            windows.append(signal[start:start + window_size])
            start += step_size

        return windows, sample_rate, step_seconds
    
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
        - threshold < error < 5*threshold: 60-80 (warning)
        - error >= 5*threshold: 30-50 (critical)
        """
        if median_error <= self.threshold:
            return random.randint(95, 100)
        elif median_error < 5 * self.threshold:
            return random.randint(60, 80)
        else:
            return random.randint(30, 50)
    
    def analyze(self, file_path: str) -> dict:
        """
        Analyze a file for machine health using multi-window aggregation.

        Pipeline (per window):
            1. Extract 13 audio features
            2. Anomaly score via autoencoder reconstruction error
            3. Fault classification via Random Forest
            4. Physics validation against mechanical rules

        Aggregation:
            - Status: >=50% anomalous -> faulty, 25-50% -> warning, <25% -> normal
            - Fault type: majority vote across anomalous windows
            - Confidence: median of anomalous-window confidences
            - Physics: majority vote of per-window agree/disagree/na

        Supports: .mat, .wav, .mp3, .mp4, .m4a, .flac, .avi, .mov, .webm
        """
        # Convert to absolute path
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        # Determine file type and load signal
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.mat':
            signal, sample_rate = self._load_mat_signal(file_path)
        else:
            signal, sample_rate = self._load_audio_signal(file_path)

        # ─── WINDOWING (50% overlap) ───
        windows, sr, step_seconds = self._split_into_windows(signal, sample_rate)

        if len(windows) == 0:
            raise ValueError("Signal too short for analysis")

        # ─── PER-WINDOW INFERENCE ───
        window_errors = []       # autoencoder reconstruction error
        rf_probabilities = []    # max class probability per window
        rf_predictions = []      # predicted class index per window
        dominant_freqs = []      # physics feature: dominant frequency
        spectral_kurtoses = []   # physics feature: spectral kurtosis
        all_features = []        # raw features (for fingerprint)
        per_window_physics = []  # physics validation per window

        n_anomalous = 0

        for window in windows:
            # 1. Extract features
            features = np.array(extract_features(window, sr))
            spectral_kurtoses.append(features[4])
            dominant_freqs.append(features[5])
            all_features.append(features)

            # 2. Scale & compute anomaly score
            features_scaled = self.scaler.transform(features.reshape(1, -1)).flatten()
            error = self._compute_reconstruction_error(features_scaled)
            window_errors.append(error)
            is_window_anomalous = error > self.threshold
            if is_window_anomalous:
                n_anomalous += 1

            # 3. Classify fault type
            proba = self.classifier.predict_proba(features_scaled.reshape(1, -1))[0]
            predicted_class = self.classifier.predict(features_scaled.reshape(1, -1))[0]
            rf_probabilities.append(float(max(proba)))
            rf_predictions.append(predicted_class)

            # 4. Per-window physics validation
            window_label = self.idx_to_label.get(predicted_class, "unknown")
            if window_label == "healthy":
                window_label = None  # healthy → no fault label for physics
            pw_physics = validate_physics(
                failure_type=window_label if is_window_anomalous else None,
                dominant_freq=features[5],
                spectral_kurtosis=features[4],
                anomaly_score=error,
                is_faulty=is_window_anomalous,
            )
            per_window_physics.append(pw_physics)

        # ─── AGGREGATE: basic statistics ───
        total_windows = len(windows)
        median_error = float(np.median(window_errors))
        min_error = float(np.min(window_errors))
        max_error = float(np.max(window_errors))
        anomaly_ratio = n_anomalous / total_windows
        median_dominant_freq = float(np.median(dominant_freqs))
        median_spectral_kurtosis = float(np.median(spectral_kurtoses))

        # ─── AGGREGATE: status (3-tier) ───
        # Thresholds: <0.25 normal, 0.25-0.50 warning, >=0.50 faulty
        if anomaly_ratio >= 0.50:
            status = "faulty"
        elif anomaly_ratio >= 0.25:
            status = "warning"
        else:
            status = "normal"

        is_faulty = status == "faulty"
        is_warning = status == "warning"

        # Also let RF override to faulty when it is very confident
        fault_predictions = [
            p for p in rf_predictions
            if self.idx_to_label.get(p, "healthy") != "healthy"
        ]
        rf_fault_ratio = len(fault_predictions) / total_windows
        avg_rf_confidence = float(np.mean(rf_probabilities)) if rf_probabilities else 0.0
        max_rf_confidence = float(max(rf_probabilities)) if rf_probabilities else 0.0

        rf_thinks_faulty = rf_fault_ratio > 0.5 and avg_rf_confidence > 0.6
        if rf_thinks_faulty and status == "normal":
            status = "warning"  # promote to at least warning
            is_warning = True

        # ─── AGGREGATE: fault type (majority vote on anomalous windows) ───
        anomalous_labels = [
            self.idx_to_label.get(rf_predictions[i], "unknown")
            for i in range(total_windows)
            if window_errors[i] > self.threshold
        ]
        # Filter out 'healthy' votes — we only want fault labels
        fault_labels = [l for l in anomalous_labels if l != "healthy"]

        fault_type = None
        if fault_labels:
            label_counts = Counter(fault_labels)
            top_label, top_count = label_counts.most_common(1)[0]
            # Require a clear majority (> 50% of fault votes)
            if top_count > len(fault_labels) / 2:
                fault_type = top_label
            else:
                fault_type = top_label  # plurality winner
        # If not faulty at all, clear the label
        if status == "normal":
            fault_type = None

        # ─── AGGREGATE: physics validation (majority vote) ───
        # (must come before confidence, which depends on physics counts)
        physics_agree = sum(1 for p in per_window_physics if p.get("consistent") is True)
        physics_disagree = sum(1 for p in per_window_physics if p.get("consistent") is False)
        physics_na = sum(1 for p in per_window_physics if p.get("consistent") is None)

        if physics_agree > physics_disagree and physics_agree > physics_na:
            aggregated_physics_consistent = True
        elif physics_disagree > physics_agree and physics_disagree > physics_na:
            aggregated_physics_consistent = False
        else:
            aggregated_physics_consistent = None

        # Build aggregated physics reason from the most common per-window reason
        physics_reasons = [p.get("reason", "") for p in per_window_physics if p.get("reason")]
        aggregated_physics_reason = Counter(physics_reasons).most_common(1)[0][0] if physics_reasons else ""

        # Also build a representative physics result (for observed/expected in the UI)
        # Use the median-feature-based validation for the summary
        representative_physics = validate_physics(
            failure_type=fault_type,
            dominant_freq=median_dominant_freq,
            spectral_kurtosis=median_spectral_kurtosis,
            anomaly_score=median_error,
            is_faulty=(status in ("faulty", "warning")),
        )

        # ─── AGGREGATE: confidence ───
        # Combined confidence blends three independent signals:
        #   1. anomaly_ratio  (autoencoder agreement across windows)
        #   2. avg_rf_confidence (classifier certainty)
        #   3. physics agree_ratio (physics-informed validation)
        anomalous_confidences = [
            rf_probabilities[i]
            for i in range(total_windows)
            if window_errors[i] > self.threshold
        ]
        physics_applicable = physics_agree + physics_disagree

        if status == "normal":
            # Normal confidence: distance from threshold + classifier agreement
            distance_confidence = 1.0 - (median_error / self.threshold) if self.threshold > 0 else 1.0
            distance_confidence = max(0.0, min(1.0, distance_confidence))
            combined_confidence = 0.5 * distance_confidence + 0.5 * avg_rf_confidence
        else:
            # Faulty/warning: blend anomaly ratio, classifier conf, physics
            ae_signal = anomaly_ratio  # higher = more certain something is wrong
            rf_signal = float(np.median(anomalous_confidences)) if anomalous_confidences else avg_rf_confidence
            physics_signal = (physics_agree / physics_applicable) if physics_applicable > 0 else 0.5
            # Weighted combination: AE 40%, RF 40%, Physics 20%
            combined_confidence = 0.40 * ae_signal + 0.40 * rf_signal + 0.20 * physics_signal

        combined_confidence = max(0.0, min(1.0, combined_confidence))

        # Adjust confidence based on physics representative result
        if representative_physics.get("confidence_modifier", 0) != 0:
            combined_confidence = max(0.0, min(1.0, combined_confidence + representative_physics["confidence_modifier"]))

        # ─── PHYSICS-MODULATED DECISION LOGIC ───
        # Physics NEVER overrides ML, it only modulates severity/confidence.
        #
        # RULE 1: ML=faulty + physics DISAGREES -> downgrade to warning
        # RULE 2: ML=warning + physics AGREES   -> keep warning (no promotion)
        # RULE 3: ML=faulty + physics AGREES     -> keep faulty
        # RULE 4: physics N/A                    -> no change
        physics_modulation_applied = None

        if aggregated_physics_consistent is not None:  # skip rule 4 (N/A)
            if status == "faulty" and aggregated_physics_consistent is False:
                # RULE 1: downgrade
                status = "warning"
                is_faulty = False
                is_warning = True
                physics_modulation_applied = "downgrade_to_warning"
                # Penalise confidence slightly
                combined_confidence = max(0.0, combined_confidence - 0.10)
            elif status == "faulty" and aggregated_physics_consistent is True:
                # RULE 3: keep faulty, boost confidence
                physics_modulation_applied = "confirmed_faulty"
                combined_confidence = min(1.0, combined_confidence + 0.05)
            elif status == "warning" and aggregated_physics_consistent is True:
                # RULE 2: keep warning (do NOT promote to faulty)
                physics_modulation_applied = "confirmed_warning"
            elif status == "warning" and aggregated_physics_consistent is False:
                # Extra: physics disagrees with a warning - weaken confidence
                physics_modulation_applied = "weakened_warning"
                combined_confidence = max(0.0, combined_confidence - 0.10)

        # ─── OUT-OF-DISTRIBUTION DETECTION ───
        is_out_of_distribution = (
            median_error > self.threshold * 2 and avg_rf_confidence < 0.6
        ) or (
            anomaly_ratio > 0.7 and max_rf_confidence < 0.5
        )

        # ─── HEALTH SCORE ───
        if rf_thinks_faulty and median_error <= self.threshold:
            health_score = random.randint(45, 65)
        elif is_warning:
            health_score = random.randint(65, 80)
        else:
            health_score = self._calculate_health_score(median_error)

        # ─── FINGERPRINT & MAINTENANCE ───
        fingerprint = generate_fingerprint(all_features, is_faulty or is_warning)
        maintenance = generate_maintenance_advice(
            failure_type=fault_type,
            is_faulty=(status in ("faulty", "warning")),
            anomaly_score=median_error,
            threshold=self.threshold,
            confidence=combined_confidence,
            physics_consistent=aggregated_physics_consistent,
            out_of_distribution=is_out_of_distribution,
        )

        # ─── PER-WINDOW RESULTS (for waveform visualisation) ───
        window_results = []
        for i in range(total_windows):
            start_time = i * step_seconds
            end_time = start_time + WINDOW_DURATION
            window_results.append({
                'start_time': round(start_time, 3),
                'end_time': round(end_time, 3),
                'reconstruction_error': float(window_errors[i]),
                'is_anomalous': bool(window_errors[i] > self.threshold),
                'predicted_class': self.idx_to_label.get(rf_predictions[i], 'unknown'),
                'confidence': rf_probabilities[i],
                'physics_consistent': per_window_physics[i].get("consistent"),
            })

        # ─── BUILD RETURN DICT ───
        # Common fields shared by every branch
        physics_applicable = physics_agree + physics_disagree
        agree_ratio = (physics_agree / physics_applicable) if physics_applicable > 0 else 0.0

        physics_validation_block = {
            "consistent": aggregated_physics_consistent,
            "reason": aggregated_physics_reason,
            "observed": representative_physics.get("observed", {}),
            "expected": representative_physics.get("expected", {}),
            "agree_windows": physics_agree,
            "disagree_windows": physics_disagree,
            "na_windows": physics_na,
            "total_windows": total_windows,
            "agree_ratio": round(agree_ratio, 4),
            "modulation_applied": physics_modulation_applied,
            "summary": (
                f"Physics agreed with {physics_agree}/{physics_applicable} applicable windows "
                f"({round(agree_ratio * 100, 1)}% agreement)"
                if physics_applicable > 0
                else "No applicable physics rules for this signal"
            ),
        }

        window_summary = {
            "total": total_windows,
            "anomalous": n_anomalous,
            "ratio": round(anomaly_ratio, 4),
            "overlap": WINDOW_OVERLAP,
            "window_duration": WINDOW_DURATION,
            "hop_duration": WINDOW_DURATION * (1.0 - WINDOW_OVERLAP),
        }
        # Keep backward-compatible alias
        window_stats = window_summary

        reasoning_data = {
            "windows_analyzed": total_windows,
            "anomalous_windows": n_anomalous,
            "anomaly_ratio": round(anomaly_ratio, 4),
            "threshold": round(self.threshold, 6),
            "min_error": round(min_error, 6),
            "max_error": round(max_error, 6),
            "rf_confidence": round(max_rf_confidence if is_faulty else avg_rf_confidence, 4),
            "dominant_frequency_hz": round(median_dominant_freq, 1),
            "spectral_kurtosis": round(median_spectral_kurtosis, 2),
        }

        if is_out_of_distribution:
            return {
                "status": "faulty",
                "health_score": random.randint(25, 40),
                "anomaly_score": round(median_error, 6),
                "failure_type": None,
                "confidence": round(min(avg_rf_confidence, 0.4), 4),
                "out_of_distribution": True,
                "explanation": (
                    "The uploaded audio significantly deviates from learned machine vibration patterns. "
                    "The system detected abnormal behavior but cannot confidently map it to a known fault type."
                ),
                "physics_validation": {
                    "consistent": None,
                    "reason": "Physics validation not applicable for out-of-distribution audio",
                    "agree_windows": physics_agree,
                    "total_windows": total_windows,
                    "agree_ratio": 0.0,
                    "summary": "Physics validation not applicable for out-of-distribution audio",
                },
                "reasoning_data": {**reasoning_data, "out_of_distribution": True},
                "window_summary": window_summary,
                "window_stats": window_stats,
                "failure_fingerprint": fingerprint,
                "maintenance_advice": maintenance,
                "window_results": window_results,
                "audio_duration": round(total_windows * step_seconds + WINDOW_DURATION * WINDOW_OVERLAP, 2),
            }

        # Explanation text
        explanations = {
            "bad_ignition": "Anomaly detected indicating ignition system issues. Check spark plugs and ignition coils.",
            "dead_battery": "Low power patterns detected. Battery may need replacement or charging.",
            "worn_brakes": "Abnormal braking sounds detected. Inspect brake pads and rotors.",
            "mixed_faults": "Multiple fault indicators detected. Comprehensive inspection recommended.",
            "bearing_fault": "Impulsive vibration patterns indicate potential bearing wear.",
        }

        if status == "faulty" and fault_type:
            explanation = (
                f"Anomaly detected in {n_anomalous}/{total_windows} windows (50% overlap). "
                + explanations.get(fault_type, "Fault detected. Recommend inspection.")
            )
            if physics_modulation_applied == "confirmed_faulty":
                explanation += " Physics validation confirms this fault pattern."
        elif status == "warning":
            if physics_modulation_applied == "downgrade_to_warning":
                explanation = (
                    f"ML flagged {n_anomalous}/{total_windows} windows as anomalous, "
                    f"but physics validation disagrees with the predicted fault pattern. "
                    f"Status downgraded from faulty to warning. "
                    f"Continued monitoring recommended."
                )
            else:
                explanation = (
                    f"Anomalous patterns detected in {n_anomalous}/{total_windows} windows, "
                    f"but evidence is insufficient for a confident fault diagnosis. "
                    f"Continued monitoring recommended."
                )
        else:
            explanation = "Machine operating within normal vibration limits."
            reasoning_data["distance_from_threshold"] = round(self.threshold - median_error, 6)

        return {
            "status": status,
            "health_score": health_score,
            "anomaly_score": round(median_error, 6),
            "failure_type": fault_type,
            "confidence": round(combined_confidence, 4),
            "out_of_distribution": False,
            "explanation": explanation,
            "physics_validation": physics_validation_block,
            "reasoning_data": reasoning_data,
            "window_summary": window_summary,
            "window_stats": window_stats,  # backward-compatible alias
            "failure_fingerprint": fingerprint,
            "maintenance_advice": maintenance,
            "window_results": window_results,
            "audio_duration": round(total_windows * step_seconds + WINDOW_DURATION * WINDOW_OVERLAP, 2),
        }
    
    def analyze_signal(self, signal: np.ndarray, sample_rate: int = 12000) -> dict:
        """
        Analyze a raw signal array using multi-window aggregation (50% overlap).

        This is a lightweight wrapper - for full results use analyze() with a
        file path instead.

        Args:
            signal: numpy array (one-dimensional) of vibration samples
            sample_rate: Sample rate (default: 12000 Hz for CWRU)

        Returns:
            Dictionary with analysis results
        """
        windows, sr, step_seconds = self._split_into_windows(signal, sample_rate)

        if len(windows) == 0:
            raise ValueError("Signal too short for analysis")

        # Per-window inference
        window_errors = []
        n_anomalous = 0
        rf_probabilities = []
        rf_predictions = []

        for window in windows:
            features = np.array(extract_features(window, sr))
            features_scaled = self.scaler.transform(features.reshape(1, -1)).flatten()
            error = self._compute_reconstruction_error(features_scaled)
            window_errors.append(error)

            if error > self.threshold:
                n_anomalous += 1

            proba = self.classifier.predict_proba(features_scaled.reshape(1, -1))[0]
            predicted_class = self.classifier.predict(features_scaled.reshape(1, -1))[0]
            rf_probabilities.append(float(max(proba)))
            rf_predictions.append(predicted_class)

        # Aggregation
        total_windows = len(windows)
        median_error = float(np.median(window_errors))
        anomaly_ratio = n_anomalous / total_windows
        health_score = self._calculate_health_score(median_error)

        # 3-tier status (matches analyze() thresholds)
        if anomaly_ratio >= 0.50:
            status = "faulty"
        elif anomaly_ratio >= 0.25:
            status = "warning"
        else:
            status = "normal"

        # Fault type majority vote
        anomalous_labels = [
            self.idx_to_label.get(rf_predictions[i], "unknown")
            for i in range(total_windows)
            if window_errors[i] > self.threshold
        ]
        fault_labels = [l for l in anomalous_labels if l != "healthy"]
        fault_type = Counter(fault_labels).most_common(1)[0][0] if fault_labels else None
        if status == "normal":
            fault_type = None

        # Confidence: median of anomalous windows
        anomalous_confs = [
            rf_probabilities[i]
            for i in range(total_windows)
            if window_errors[i] > self.threshold
        ]
        confidence = float(np.median(anomalous_confs)) if anomalous_confs else None

        return {
            "status": status,
            "health_score": health_score,
            "anomaly_score": round(median_error, 6),
            "failure_type": fault_type,
            "confidence": round(confidence, 4) if confidence else None,
            "explanation": (
                f"Anomaly in {n_anomalous}/{total_windows} windows. "
                f"Fault suspected: {fault_type or 'unknown'}." if status != "normal"
                else "Machine operating within normal vibration limits."
            ),
            "window_summary": {
                "total": total_windows,
                "anomalous": n_anomalous,
                "ratio": round(anomaly_ratio, 4),
                "overlap": WINDOW_OVERLAP,
                "window_duration": WINDOW_DURATION,
                "hop_duration": WINDOW_DURATION * (1.0 - WINDOW_OVERLAP),
            },
            "window_stats": {
                "total_windows": total_windows,
                "anomalous_windows": n_anomalous,
                "anomaly_ratio": round(anomaly_ratio, 4),
                "overlap": WINDOW_OVERLAP,
            },
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

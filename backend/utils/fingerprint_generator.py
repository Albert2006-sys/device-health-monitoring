"""
Failure Fingerprint Generator

Creates normalized feature fingerprints for visual comparison between
the current sample and a healthy baseline.

Features used (normalized 0-1):
- RMS Energy
- Spectral Kurtosis  
- Dominant Frequency
- Spectral Centroid
- MFCC 1-3
"""

import numpy as np
from typing import Dict, List, Optional

# Healthy baseline values (precomputed from training data)
# These represent typical values for healthy machine operation
HEALTHY_BASELINE = {
    "rms_energy": 0.035,        # Low, stable power
    "spectral_kurtosis": 1.5,   # Low impulsiveness
    "dominant_frequency": 200,   # Low frequency operation
    "spectral_centroid": 800,   # Moderate brightness
    "mfcc_1": -150,             # Typical MFCC range
    "mfcc_2": 40,
    "mfcc_3": 10
}

# Feature normalization ranges (min, max) based on observed data
FEATURE_RANGES = {
    "rms_energy": (0.0, 0.5),
    "spectral_kurtosis": (0.0, 20.0),
    "dominant_frequency": (0.0, 8000.0),
    "spectral_centroid": (0.0, 4000.0),
    "mfcc_1": (-300, 100),
    "mfcc_2": (-50, 150),
    "mfcc_3": (-60, 80)
}


def normalize_value(value: float, feature_name: str) -> float:
    """Normalize a feature value to 0-1 range."""
    if feature_name not in FEATURE_RANGES:
        return 0.5
    
    min_val, max_val = FEATURE_RANGES[feature_name]
    if max_val == min_val:
        return 0.5
    
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def generate_fingerprint(
    features_list: List[np.ndarray],
    is_faulty: bool
) -> Dict:
    """
    Generate failure fingerprint from extracted features.
    
    Args:
        features_list: List of feature arrays from each window
                       Each array has 13 features in order:
                       [0] RMS, [1] ZCR, [2] Centroid, [3] Bandwidth,
                       [4] Kurtosis, [5] Dominant Freq, [6-12] MFCC 1-7
        is_faulty: Whether the sample was classified as faulty
    
    Returns:
        Dictionary with current and baseline fingerprints
    """
    if not features_list:
        return None
    
    # Convert to numpy array for easier manipulation
    features_array = np.array(features_list)
    
    # Compute median across all windows for stability
    median_features = np.median(features_array, axis=0)
    
    # Extract relevant features for fingerprint
    current_raw = {
        "rms_energy": float(median_features[0]),
        "spectral_kurtosis": float(median_features[4]),
        "dominant_frequency": float(median_features[5]),
        "spectral_centroid": float(median_features[2]),
        "mfcc_1": float(median_features[6]),
        "mfcc_2": float(median_features[7]),
        "mfcc_3": float(median_features[8])
    }
    
    # Normalize current features to 0-1
    current_normalized = {
        name: round(normalize_value(value, name), 3)
        for name, value in current_raw.items()
    }
    
    # Normalize baseline features
    baseline_normalized = {
        name: round(normalize_value(value, name), 3)
        for name, value in HEALTHY_BASELINE.items()
    }
    
    # Calculate deviation score (how different from baseline)
    deviations = [
        abs(current_normalized[name] - baseline_normalized[name])
        for name in current_normalized.keys()
    ]
    deviation_score = round(np.mean(deviations), 3)
    
    return {
        "features": current_normalized,
        "baseline": baseline_normalized,
        "deviation_score": deviation_score,
        "interpretation": _get_interpretation(deviation_score, is_faulty)
    }


def _get_interpretation(deviation_score: float, is_faulty: bool) -> str:
    """Generate human-readable interpretation of the fingerprint."""
    if deviation_score < 0.15:
        return "Vibration signature closely matches healthy baseline."
    elif deviation_score < 0.30:
        return "Minor deviations from healthy baseline detected."
    elif deviation_score < 0.50:
        return "Moderate deviation from healthy baseline - monitoring recommended."
    else:
        return "Significant deviation from healthy baseline - investigation required."

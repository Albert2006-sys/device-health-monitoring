"""
Physics-Informed ML Validation Module

This module provides lightweight physics-based validation for ML predictions.
It does NOT replace ML decisions - it validates them against known mechanical behavior.

Key Concepts:
- Bearing faults excite harmonics in 2-6 kHz range (BPFO, BPFI, BSF, FTF)
- Ignition issues produce low-frequency irregular patterns
- Brake wear creates mid-frequency impulsive sounds
"""

from typing import Dict, Optional, Tuple


# Physics constants for mechanical fault patterns
PHYSICS_RULES = {
    "bearing_fault": {
        "frequency_range": (2000, 6000),  # Hz - bearing fault harmonic range
        "description": "Bearing faults typically excite harmonics between 2-6 kHz (BPFO, BPFI, BSF, FTF)",
        "spectral_kurtosis_threshold": 3.0,  # Impulsive signals have high kurtosis
    },
    "bad_ignition": {
        "frequency_range": (50, 500),  # Hz - low frequency engine patterns
        "description": "Ignition issues create irregular low-frequency patterns (50-500 Hz)",
        "spectral_kurtosis_threshold": 2.0,
    },
    "worn_brakes": {
        "frequency_range": (1000, 4000),  # Hz - brake squeal range
        "description": "Brake wear produces mid-frequency squeals (1-4 kHz)",
        "spectral_kurtosis_threshold": 4.0,  # Very impulsive
    },
    "dead_battery": {
        "frequency_range": (0, 200),  # Hz - very low frequency
        "description": "Dead battery patterns show weak, low-frequency signals",
        "spectral_kurtosis_threshold": 1.5,
    },
    "mixed_faults": {
        "frequency_range": (100, 5000),  # Hz - broad range
        "description": "Multiple faults create complex broadband patterns",
        "spectral_kurtosis_threshold": 2.5,
    },
}


def validate_physics(
    failure_type: Optional[str],
    dominant_freq: float,
    spectral_kurtosis: float,
    anomaly_score: float,
    is_faulty: bool
) -> Dict:
    """
    Validate ML prediction against known physics of mechanical faults.
    
    Args:
        failure_type: Predicted fault type from RF classifier
        dominant_freq: Dominant frequency from feature extraction (Hz)
        spectral_kurtosis: Spectral kurtosis value indicating impulsiveness
        anomaly_score: Reconstruction error from autoencoder
        is_faulty: Whether the system marked this as faulty
    
    Returns:
        Dict with physics validation results
    """
    result = {
        "consistent": None,  # True, False, or None (unknown)
        "reason": "",
        "observed": {},
        "expected": {},
        "confidence_modifier": 0.0,  # Can boost or reduce confidence
    }
    
    # Store observed values
    result["observed"] = {
        "dominant_frequency_hz": round(dominant_freq, 1),
        "spectral_kurtosis": round(spectral_kurtosis, 2),
        "anomaly_score": round(anomaly_score, 6),
    }
    
    # If healthy/normal, check if physics agrees
    if not is_faulty or failure_type is None:
        # For healthy signals, dominant frequency is often in machine operating range
        # and spectral kurtosis should be low (non-impulsive)
        if spectral_kurtosis < 3.0 and dominant_freq < 8000:
            result["consistent"] = True
            result["reason"] = (
                "Signal characteristics align with normal machine operation. "
                f"Low impulsiveness (kurtosis={spectral_kurtosis:.2f}) and stable frequency patterns."
            )
            result["expected"] = {
                "pattern": "Non-impulsive, stable vibration",
                "kurtosis_range": "< 3.0",
            }
            result["confidence_modifier"] = 0.05  # Slight boost
        else:
            result["consistent"] = None
            result["reason"] = "Insufficient physics data for healthy validation."
        return result
    
    # Get physics rules for this fault type
    rules = PHYSICS_RULES.get(failure_type)
    
    if rules is None:
        result["consistent"] = None
        result["reason"] = f"No physics rules defined for fault type: {failure_type}"
        return result
    
    freq_min, freq_max = rules["frequency_range"]
    kurtosis_threshold = rules["spectral_kurtosis_threshold"]
    
    # Store expected values
    result["expected"] = {
        "frequency_range_hz": f"{freq_min}-{freq_max}",
        "description": rules["description"],
        "kurtosis_threshold": kurtosis_threshold,
    }
    
    # Check frequency consistency
    freq_consistent = freq_min <= dominant_freq <= freq_max
    
    # Check kurtosis (impulsiveness) for fault types that need it
    kurtosis_consistent = spectral_kurtosis >= kurtosis_threshold * 0.5  # Allow some tolerance
    
    # Combined physics consistency
    if freq_consistent and kurtosis_consistent:
        result["consistent"] = True
        result["reason"] = (
            f"Dominant frequency ({dominant_freq:.0f} Hz) aligns with expected "
            f"{failure_type.replace('_', ' ')} patterns ({freq_min}-{freq_max} Hz). "
            f"{rules['description']}"
        )
        result["confidence_modifier"] = 0.1  # Boost confidence
        
    elif freq_consistent or kurtosis_consistent:
        result["consistent"] = True  # Partially consistent is still valid
        if freq_consistent:
            result["reason"] = (
                f"Frequency pattern ({dominant_freq:.0f} Hz) matches expected range for "
                f"{failure_type.replace('_', ' ')} ({freq_min}-{freq_max} Hz)."
            )
        else:
            result["reason"] = (
                f"Signal impulsiveness (kurtosis={spectral_kurtosis:.2f}) suggests "
                f"mechanical irregularity consistent with {failure_type.replace('_', ' ')}."
            )
        result["confidence_modifier"] = 0.05
        
    else:
        result["consistent"] = False
        result["reason"] = (
            f"Observed frequency ({dominant_freq:.0f} Hz) does not align with expected "
            f"{failure_type.replace('_', ' ')} patterns ({freq_min}-{freq_max} Hz). "
            "ML prediction may require manual verification."
        )
        result["confidence_modifier"] = -0.1  # Reduce confidence
    
    return result


def generate_physics_explanation(physics_result: Dict, failure_type: Optional[str]) -> str:
    """
    Generate human-readable physics explanation for the AI reasoning section.
    
    Returns a formatted string for display in the UI.
    """
    if physics_result["consistent"] is None:
        return ""
    
    observed = physics_result.get("observed", {})
    expected = physics_result.get("expected", {})
    
    lines = []
    lines.append("🔬 Physics Validation")
    
    # What was observed
    if "dominant_frequency_hz" in observed:
        lines.append(f"• Detected peak at {observed['dominant_frequency_hz']:.0f} Hz")
    
    if "spectral_kurtosis" in observed:
        kurtosis = observed["spectral_kurtosis"]
        if kurtosis > 4:
            lines.append(f"• High impulsiveness detected (kurtosis={kurtosis:.1f})")
        elif kurtosis > 2:
            lines.append(f"• Moderate impulsiveness (kurtosis={kurtosis:.1f})")
    
    # What physics expects
    if expected.get("frequency_range_hz"):
        lines.append(f"• Expected range for this fault: {expected['frequency_range_hz']} Hz")
    
    if expected.get("description"):
        lines.append(f"• {expected['description']}")
    
    # Conclusion
    if physics_result["consistent"]:
        lines.append("• ✓ Signal matches mechanical expectations")
    else:
        lines.append("• ⚠ Signal does not match typical mechanical patterns")
    
    return "\n".join(lines)

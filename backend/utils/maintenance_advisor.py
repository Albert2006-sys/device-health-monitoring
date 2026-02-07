"""
Maintenance Advice Generator

Generates actionable maintenance recommendations based on:
- Fault type detected
- Anomaly severity
- Physics validation results
- Confidence level
"""

from typing import Dict, List, Optional


# Maintenance recommendations by fault type
FAULT_RECOMMENDATIONS = {
    "bearing_fault": {
        "high": {
            "urgency": "high",
            "actions": [
                "Inspect bearings within 24 hours",
                "Check lubrication levels immediately",
                "Monitor vibration trend continuously",
                "Prepare replacement parts"
            ]
        },
        "medium": {
            "urgency": "medium",
            "actions": [
                "Schedule bearing inspection within 1 week",
                "Check and refill lubricant",
                "Increase monitoring frequency",
                "Document vibration levels"
            ]
        },
        "low": {
            "urgency": "low",
            "actions": [
                "Include in next scheduled maintenance",
                "Monitor for changes in vibration pattern",
                "Verify lubrication schedule"
            ]
        }
    },
    "worn_brakes": {
        "high": {
            "urgency": "high",
            "actions": [
                "Inspect brake pads immediately",
                "Check friction surfaces for wear",
                "Measure brake pad thickness",
                "Schedule replacement if wear exceeds limits"
            ]
        },
        "medium": {
            "urgency": "medium",
            "actions": [
                "Inspect brake system within 1 week",
                "Check for unusual wear patterns",
                "Verify brake fluid levels"
            ]
        },
        "low": {
            "urgency": "low",
            "actions": [
                "Monitor for brake noise changes",
                "Include brakes in next routine inspection"
            ]
        }
    },
    "bad_ignition": {
        "high": {
            "urgency": "high",
            "actions": [
                "Check spark plugs immediately",
                "Inspect ignition coil connections",
                "Test ignition timing",
                "Verify fuel system pressure"
            ]
        },
        "medium": {
            "urgency": "medium",
            "actions": [
                "Schedule ignition system diagnostic",
                "Check spark plug condition",
                "Inspect ignition cables"
            ]
        },
        "low": {
            "urgency": "low",
            "actions": [
                "Monitor engine performance",
                "Include ignition check in next service"
            ]
        }
    },
    "dead_battery": {
        "high": {
            "urgency": "high",
            "actions": [
                "Test battery voltage immediately",
                "Check charging system output",
                "Inspect battery terminals for corrosion",
                "Prepare replacement battery"
            ]
        },
        "medium": {
            "urgency": "medium",
            "actions": [
                "Load test battery within 1 week",
                "Clean battery terminals",
                "Check alternator belt tension"
            ]
        },
        "low": {
            "urgency": "low",
            "actions": [
                "Monitor battery condition",
                "Check electrolyte levels (if applicable)"
            ]
        }
    },
    "mixed_faults": {
        "high": {
            "urgency": "high",
            "actions": [
                "Conduct comprehensive system inspection",
                "Check multiple component groups",
                "Document all anomalies found",
                "Prioritize repairs by severity"
            ]
        },
        "medium": {
            "urgency": "medium",
            "actions": [
                "Schedule detailed diagnostic session",
                "Check for common failure modes",
                "Review maintenance history"
            ]
        },
        "low": {
            "urgency": "low",
            "actions": [
                "Continue monitoring",
                "Document baseline measurements",
                "Review at next scheduled maintenance"
            ]
        }
    }
}

# Default recommendations for unknown/OOD audio
UNKNOWN_RECOMMENDATIONS = {
    "urgency": "medium",
    "actions": [
        "Verify sensor placement and connection",
        "Re-record under controlled conditions",
        "Manual inspection recommended",
        "Compare with known reference recordings"
    ]
}

# Normal machine recommendations
NORMAL_RECOMMENDATIONS = {
    "urgency": "low",
    "actions": [
        "Continue normal operation",
        "Maintain regular maintenance schedule",
        "Document baseline for future comparison"
    ]
}


def determine_severity(
    anomaly_score: float,
    threshold: float,
    confidence: float,
    physics_consistent: Optional[bool]
) -> str:
    """
    Determine fault severity based on multiple factors.
    
    Returns: 'high', 'medium', or 'low'
    """
    # Base severity on how far above threshold
    if threshold > 0:
        severity_ratio = anomaly_score / threshold
    else:
        severity_ratio = 1.0
    
    # Factor in confidence
    if confidence and confidence > 0.85:
        confidence_factor = 1.2
    elif confidence and confidence > 0.6:
        confidence_factor = 1.0
    else:
        confidence_factor = 0.8
    
    # Physics consistency affects urgency
    physics_factor = 1.0 if physics_consistent else 0.9
    
    # Combined severity score
    severity_score = severity_ratio * confidence_factor * physics_factor
    
    if severity_score > 2.0:
        return "high"
    elif severity_score > 1.2:
        return "medium"
    else:
        return "low"


def generate_maintenance_advice(
    failure_type: Optional[str],
    is_faulty: bool,
    anomaly_score: float,
    threshold: float,
    confidence: float,
    physics_consistent: Optional[bool],
    out_of_distribution: bool = False
) -> Dict:
    """
    Generate maintenance advice based on analysis results.
    
    Args:
        failure_type: Detected fault type (or None)
        is_faulty: Whether classified as faulty
        anomaly_score: Reconstruction error from autoencoder
        threshold: Anomaly threshold
        confidence: ML confidence level
        physics_consistent: Whether physics validation passed
        out_of_distribution: Whether audio is OOD
    
    Returns:
        Dictionary with urgency and recommended actions
    """
    # Handle normal operation
    if not is_faulty:
        return {
            **NORMAL_RECOMMENDATIONS,
            "note": "Machine operating within normal parameters.",
            "generated_from": "status: normal"
        }
    
    # Handle out-of-distribution
    if out_of_distribution or failure_type is None:
        return {
            **UNKNOWN_RECOMMENDATIONS,
            "note": "Unable to determine specific fault. Manual verification needed.",
            "generated_from": "out_of_distribution detection"
        }
    
    # Determine severity
    severity = determine_severity(
        anomaly_score, threshold, confidence, physics_consistent
    )
    
    # Get recommendations for this fault type and severity
    fault_advice = FAULT_RECOMMENDATIONS.get(failure_type, {})
    advice = fault_advice.get(severity, UNKNOWN_RECOMMENDATIONS)
    
    # Build physics note if available
    physics_note = ""
    if physics_consistent is True:
        physics_note = "Physics validation confirms mechanical fault pattern."
    elif physics_consistent is False:
        physics_note = "Atypical vibration pattern - verify with manual inspection."
    
    return {
        "urgency": advice["urgency"],
        "recommended_actions": advice["actions"],
        "severity": severity,
        "note": physics_note if physics_note else f"Based on {failure_type.replace('_', ' ')} detection.",
        "generated_from": f"fault_type: {failure_type}, severity: {severity}"
    }

"""
Physics-Informed ML Validation Module
======================================

Lightweight, rule-based validation layer for ML predictions in a Device
Health Monitoring (DHM) system.

Design principles
-----------------
* **Never override** ML predictions — only annotate with physics reasoning.
* **Never invent** new fault labels — only validate the ones the ML produces.
* Purely deterministic — no external dependencies, no simulations.

Authoritative physics rules implemented
----------------------------------------
1. Bearing fault      → 2 000–6 000 Hz dominant, kurtosis > 3.5  (impulsive)
2. Imbalance / Misalignment → < 500 Hz dominant  (low-frequency)
3. Healthy operation  → no strong impulsive behaviour, kurtosis near baseline
4. Unknown / unseen   → physics not applicable  → consistent = None
"""

from __future__ import annotations

import math
from typing import Dict, Optional


# ──────────────────────────────────────────────
# Physics constants (single source of truth)
# ──────────────────────────────────────────────

# Bearing fault: high-frequency harmonics + impulsive energy
BEARING_FREQ_MIN: float = 2000.0   # Hz
BEARING_FREQ_MAX: float = 6000.0   # Hz
BEARING_KURTOSIS_MIN: float = 3.5  # impulsive threshold (vibration domain)

# Imbalance / misalignment: low-frequency vibration
IMBALANCE_FREQ_MAX: float = 500.0  # Hz

# Healthy baseline: non-impulsive, no dominant spikes
HEALTHY_KURTOSIS_MAX: float = 3.5  # below this → non-impulsive (vibration domain)

# Domain detection threshold.
# Our feature extractor computes kurtosis on the FFT magnitude spectrum
# (scipy.stats.kurtosis), which yields values of 5–6000+ for audio files
# vs. 1–10 for raw vibration signals (.mat).  When kurtosis >> 10 the
# reading is *always* high regardless of machine health, so the kurtosis
# rule becomes uninformative and we must rely on frequency alone.
AUDIO_DOMAIN_KURTOSIS_FLOOR: float = 10.0  # above this → audio domain

# Fault aliases — map every ML label the system can produce to one
# of the physics categories above (or None if no rule applies).
_FAULT_TO_PHYSICS_CATEGORY: Dict[str, Optional[str]] = {
    "bearing_fault":  "bearing",
    "bad_ignition":   "imbalance",    # ignition → low-freq irregular patterns
    "dead_battery":   None,           # no strong physics rule
    "worn_brakes":    None,           # no strong physics rule
    "mixed_faults":   None,           # too broad for a single rule
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _is_finite(value: object) -> bool:
    """Return True if *value* is a finite number (not None, NaN, or ±inf)."""
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_round(value: object, ndigits: int = 2) -> Optional[float]:
    """Round *value* if it is finite, otherwise return None."""
    if _is_finite(value):
        return round(float(value), ndigits)
    return None


def _is_audio_domain(kurtosis_value: float) -> bool:
    """Detect whether the kurtosis value comes from the audio feature domain.

    Our feature extractor computes ``scipy.stats.kurtosis`` on the FFT
    magnitude spectrum, which yields values of 5-6 000+ for audio files
    but 1-10 for raw vibration (.mat) data.  When the value is clearly
    above the vibration-domain range the kurtosis check becomes
    uninformative and the validator should rely on frequency alone.
    """
    return kurtosis_value > AUDIO_DOMAIN_KURTOSIS_FLOOR


# ──────────────────────────────────────────────
# Core public API (task requirement)
# ──────────────────────────────────────────────

def physics_validate(
    dominant_frequency: float,
    spectral_kurtosis: float,
    failure_type: Optional[str],
) -> Dict:
    """
    Validate an ML prediction against known mechanical physics.

    Parameters
    ----------
    dominant_frequency : float
        Dominant frequency of the signal in Hz.
    spectral_kurtosis : float
        Spectral kurtosis (impulsiveness measure).
    failure_type : str | None
        ML-predicted fault label, or None for healthy / unknown.

    Returns
    -------
    dict  ``{"consistent": bool | None, "reason": str}``
        * ``True``  — physics agrees with the ML result.
        * ``False`` — physics contradicts the ML result.
        * ``None``  — physics is not applicable.
    """
    # ---- guard: unusable inputs → physics not applicable ----
    if not _is_finite(dominant_frequency) or not _is_finite(spectral_kurtosis):
        return {
            "consistent": None,
            "reason": "Physics not applicable: missing or invalid signal measurements.",
        }

    freq = float(dominant_frequency)
    kurt = float(spectral_kurtosis)

    # ---- healthy / no-fault prediction ----
    if failure_type is None:
        return _validate_healthy(freq, kurt)

    # ---- map ML label to a physics category ----
    category = _FAULT_TO_PHYSICS_CATEGORY.get(failure_type)

    if category is None:
        return {
            "consistent": None,
            "reason": (
                f"Physics not applicable for fault type '{failure_type}'. "
                "No physics rule is defined for this category."
            ),
        }

    # ---- dispatch to the appropriate rule ----
    if category == "bearing":
        return _validate_bearing(freq, kurt, failure_type)

    if category == "imbalance":
        return _validate_imbalance(freq, failure_type)

    # Fallback (should not be reached with current categories)
    return {
        "consistent": None,
        "reason": "Physics not applicable: unrecognised physics category.",
    }


# ──────────────────────────────────────────────
# Rule implementations (private)
# ──────────────────────────────────────────────

def _validate_bearing(freq: float, kurt: float, label: str) -> Dict:
    """
    Bearing fault rule.

    Physics expectation:
      - Dominant frequency in 2 000-6 000 Hz
      - Spectral kurtosis > 3.5 (impulsive harmonics)

    In the audio domain (kurtosis >> 10) the kurtosis is always high
    regardless of fault type, so the frequency check alone determines
    consistency.
    """
    freq_ok = BEARING_FREQ_MIN <= freq <= BEARING_FREQ_MAX
    audio_domain = _is_audio_domain(kurt)
    kurt_ok = kurt > BEARING_KURTOSIS_MIN  # always True in audio domain

    # ── frequency matches ──
    if freq_ok and (kurt_ok or audio_domain):
        reason = (
            f"Dominant frequency ({freq:.0f} Hz) falls within the expected "
            f"bearing-fault harmonic range ({BEARING_FREQ_MIN:.0f}-{BEARING_FREQ_MAX:.0f} Hz)."
        )
        if kurt_ok and not audio_domain:
            reason += f" Spectral kurtosis ({kurt:.2f}) confirms impulsive behaviour."
        return {"consistent": True, "reason": reason}

    # ── frequency does NOT match ──
    if audio_domain:
        # Kurtosis is uninformative → verdict rests entirely on frequency
        return {
            "consistent": False,
            "reason": (
                f"Dominant frequency ({freq:.0f} Hz) is outside the bearing-fault range "
                f"({BEARING_FREQ_MIN:.0f}-{BEARING_FREQ_MAX:.0f} Hz). "
                "ML prediction may warrant manual verification."
            ),
        }

    # Vibration domain: can use both signals
    if kurt_ok:
        return {
            "consistent": False,
            "reason": (
                f"Kurtosis ({kurt:.2f}) is impulsive, but dominant frequency ({freq:.0f} Hz) "
                f"is outside the bearing-fault range ({BEARING_FREQ_MIN:.0f}-{BEARING_FREQ_MAX:.0f} Hz). "
                "ML prediction may warrant manual verification."
            ),
        }

    return {
        "consistent": False,
        "reason": (
            f"Neither frequency ({freq:.0f} Hz) nor kurtosis ({kurt:.2f}) match the "
            f"expected bearing-fault profile ({BEARING_FREQ_MIN:.0f}-{BEARING_FREQ_MAX:.0f} Hz, "
            f"kurtosis > {BEARING_KURTOSIS_MIN}). ML prediction may warrant manual verification."
        ),
    }


def _validate_imbalance(freq: float, label: str) -> Dict:
    """
    Imbalance / misalignment rule.

    Physics expectation:
      • Dominant frequency < 500 Hz (energy near fundamental)
    """
    if freq < IMBALANCE_FREQ_MAX:
        return {
            "consistent": True,
            "reason": (
                f"Dominant frequency ({freq:.0f} Hz) is below {IMBALANCE_FREQ_MAX:.0f} Hz, "
                f"consistent with low-frequency vibration expected for '{label.replace('_', ' ')}'."
            ),
        }

    return {
        "consistent": False,
        "reason": (
            f"Dominant frequency ({freq:.0f} Hz) is above the expected "
            f"{IMBALANCE_FREQ_MAX:.0f} Hz ceiling for '{label.replace('_', ' ')}'. "
            "ML prediction may warrant manual verification."
        ),
    }


def _validate_healthy(freq: float, kurt: float) -> Dict:
    """
    Healthy-operation rule.

    Physics expectation:
      - No strong impulsive behaviour (kurtosis <= baseline)
      - No dominant harmonic spikes

    In the audio domain the kurtosis is always high (feature-scale artefact),
    so we fall back to checking that the dominant frequency does NOT fall
    inside a known fault-characteristic range.
    """
    audio_domain = _is_audio_domain(kurt)

    # ── Vibration domain: kurtosis is informative ──
    if not audio_domain:
        if kurt <= HEALTHY_KURTOSIS_MAX:
            return {
                "consistent": True,
                "reason": (
                    f"Spectral kurtosis ({kurt:.2f}) is within the non-impulsive baseline "
                    f"(<= {HEALTHY_KURTOSIS_MAX}), consistent with normal operation."
                ),
            }
        return {
            "consistent": False,
            "reason": (
                f"Spectral kurtosis ({kurt:.2f}) exceeds the healthy baseline "
                f"(> {HEALTHY_KURTOSIS_MAX}), suggesting impulsive behaviour that may "
                "indicate an undetected fault."
            ),
        }

    # ── Audio domain: kurtosis uninformative, check frequency ──
    in_bearing_range = BEARING_FREQ_MIN <= freq <= BEARING_FREQ_MAX

    if not in_bearing_range:
        return {
            "consistent": True,
            "reason": (
                f"Dominant frequency ({freq:.0f} Hz) does not fall into a known "
                "fault-characteristic range. No impulsive anomaly detected; "
                "consistent with normal operation."
            ),
        }

    # Dominant freq IS in bearing-fault territory — flag it
    return {
        "consistent": False,
        "reason": (
            f"Dominant frequency ({freq:.0f} Hz) falls within the bearing-fault "
            f"range ({BEARING_FREQ_MIN:.0f}-{BEARING_FREQ_MAX:.0f} Hz), which is unusual "
            "for healthy operation. Manual verification recommended."
        ),
    }


# ──────────────────────────────────────────────
# Backward-compatible wrapper used by analyze.py
# ──────────────────────────────────────────────

def validate_physics(
    failure_type: Optional[str],
    dominant_freq: float,
    spectral_kurtosis: float,
    anomaly_score: float,
    is_faulty: bool,
) -> Dict:
    """
    Legacy wrapper that maps the old call-site signature
    (``analyze.py``) to the new ``physics_validate()`` core.

    Returned dict includes the extra keys the rest of the pipeline
    already expects: ``observed``, ``expected``, ``confidence_modifier``.
    """
    # Determine the effective failure_type for physics_validate:
    # If the ML says "healthy" (not faulty / no label) → pass None.
    effective_type = failure_type if is_faulty else None

    core = physics_validate(
        dominant_frequency=dominant_freq,
        spectral_kurtosis=spectral_kurtosis,
        failure_type=effective_type,
    )

    # Build the enriched result expected downstream
    consistent = core["consistent"]

    observed = {
        "dominant_frequency_hz": _safe_round(dominant_freq, 1),
        "spectral_kurtosis": _safe_round(spectral_kurtosis, 2),
        "anomaly_score": _safe_round(anomaly_score, 6),
    }

    expected: Dict = {}
    confidence_modifier = 0.0

    if consistent is True:
        confidence_modifier = 0.1 if is_faulty else 0.05
    elif consistent is False:
        confidence_modifier = -0.1

    # Populate expected context for bearing faults
    category = _FAULT_TO_PHYSICS_CATEGORY.get(failure_type or "")
    if category == "bearing":
        expected = {
            "frequency_range_hz": f"{BEARING_FREQ_MIN:.0f}-{BEARING_FREQ_MAX:.0f}",
            "description": "Bearing faults generate high-frequency harmonics (2000-6000 Hz) with impulsive kurtosis > 3.5",
            "kurtosis_threshold": BEARING_KURTOSIS_MIN,
        }
    elif category == "imbalance":
        expected = {
            "frequency_range_hz": f"0-{IMBALANCE_FREQ_MAX:.0f}",
            "description": f"Low-frequency vibration below {IMBALANCE_FREQ_MAX:.0f} Hz, energy near fundamental",
        }
    elif effective_type is None:
        expected = {
            "pattern": "Non-impulsive, stable vibration",
            "kurtosis_range": f"<= {HEALTHY_KURTOSIS_MAX}",
        }

    return {
        "consistent": consistent,
        "reason": core["reason"],
        "observed": observed,
        "expected": expected,
        "confidence_modifier": confidence_modifier,
    }


# ──────────────────────────────────────────────
# Human-readable explanation (used by UI)
# ──────────────────────────────────────────────

def generate_physics_explanation(physics_result: Dict, failure_type: Optional[str]) -> str:
    """
    Generate a concise, human-readable physics explanation for the
    AI-reasoning section in the frontend.

    Returns an empty string when physics is not applicable.
    """
    consistent = physics_result.get("consistent")
    if consistent is None:
        return ""

    observed = physics_result.get("observed", {})
    expected = physics_result.get("expected", {})

    lines: list[str] = ["[Physics Validation]"]

    # --- observed measurements ---
    freq = observed.get("dominant_frequency_hz")
    if freq is not None:
        lines.append(f"* Detected peak at {freq:.0f} Hz")

    kurt = observed.get("spectral_kurtosis")
    if kurt is not None:
        if kurt > 4:
            lines.append(f"* High impulsiveness detected (kurtosis = {kurt:.1f})")
        elif kurt > HEALTHY_KURTOSIS_MAX:
            lines.append(f"* Moderate impulsiveness (kurtosis = {kurt:.1f})")
        else:
            lines.append(f"* Low impulsiveness (kurtosis = {kurt:.1f})")

    # --- expected physics ---
    if expected.get("frequency_range_hz"):
        lines.append(f"* Expected range for this fault: {expected['frequency_range_hz']} Hz")

    if expected.get("description"):
        lines.append(f"* {expected['description']}")

    # --- verdict ---
    if consistent:
        lines.append("* PASS: Signal matches mechanical expectations")
    else:
        lines.append("* WARN: Signal does not match typical mechanical patterns")

    return "\n".join(lines)

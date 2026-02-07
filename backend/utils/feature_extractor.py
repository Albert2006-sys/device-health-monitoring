"""
Feature Extractor Module for Device Health Monitoring

Extracts 13 audio features for machine learning-based fault detection.
Features are returned in a fixed order for model compatibility.

Features (in order):
1. RMS Energy - Average signal power/loudness
2. Zero Crossing Rate - How often signal crosses zero (indicates noisiness)
3. Spectral Centroid - "Center of mass" of spectrum (brightness)
4. Spectral Bandwidth - Spread of frequencies around centroid
5. Spectral Kurtosis - Peakedness of frequency distribution
6. Dominant Frequency - Strongest frequency component (main vibration)
7-13. MFCC 1-7 - Mel-frequency cepstral coefficients (timbral texture)
"""

import numpy as np
import librosa
from scipy.stats import kurtosis


def extract_features(y: np.ndarray, sr: int = 22050) -> list:
    """
    Extract 13 audio features from a raw audio signal.

    Args:
        y: Audio time series (1D numpy array)
        sr: Sample rate (default: 22050 Hz)

    Returns:
        List of 13 float features in fixed order
    """
    features = []

    # 1. RMS Energy - measures average signal power
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms)) if len(rms) > 0 else 0.0
    features.append(rms_mean)

    # 2. Zero Crossing Rate - indicates signal noisiness
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr_mean = float(np.mean(zcr)) if len(zcr) > 0 else 0.0
    features.append(zcr_mean)

    # 3. Spectral Centroid - brightness of sound
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(spectral_centroid)) if len(spectral_centroid) > 0 else 0.0
    features.append(centroid_mean)

    # 4. Spectral Bandwidth - frequency spread
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    bandwidth_mean = float(np.mean(spectral_bandwidth)) if len(spectral_bandwidth) > 0 else 0.0
    features.append(bandwidth_mean)

    # 5. Spectral Kurtosis - peakedness of spectrum
    spectrum = np.abs(np.fft.rfft(y))
    if len(spectrum) > 0 and np.std(spectrum) > 0:
        spectral_kurt = float(kurtosis(spectrum, fisher=True, nan_policy='omit'))
        if np.isnan(spectral_kurt):
            spectral_kurt = 0.0
    else:
        spectral_kurt = 0.0
    features.append(spectral_kurt)

    # 6. Dominant Frequency - strongest frequency via FFT
    fft_freqs = np.fft.rfftfreq(len(y), d=1.0/sr)
    if len(spectrum) > 0:
        dominant_idx = np.argmax(spectrum)
        dominant_freq = float(fft_freqs[dominant_idx]) if dominant_idx < len(fft_freqs) else 0.0
    else:
        dominant_freq = 0.0
    features.append(dominant_freq)

    # 7-13. MFCCs 1-7 - timbral texture features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=7)
    for i in range(7):
        mfcc_mean = float(np.mean(mfccs[i])) if mfccs.shape[1] > 0 else 0.0
        if np.isnan(mfcc_mean):
            mfcc_mean = 0.0
        features.append(mfcc_mean)

    return features


def extract_features_from_file(file_path: str, sr: int = 22050) -> list:
    """
    Load an audio file and extract features.

    Args:
        file_path: Path to audio file (WAV, MP3, etc.)
        sr: Target sample rate (default: 22050 Hz)

    Returns:
        List of 13 float features in fixed order
    """
    y, sr_loaded = librosa.load(file_path, sr=sr)
    return extract_features(y, sr_loaded)

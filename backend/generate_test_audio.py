"""Generate a synthetic test audio file for feature extractor testing."""
import numpy as np
from scipy.io import wavfile
import os

# Create output directory if needed
os.makedirs("data/test/normal", exist_ok=True)

# Generate synthetic audio (1 second, 22050 Hz)
sr = 22050
duration = 1.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# Simulate machine sound: base frequency + harmonics + noise
base_freq = 60  # 60 Hz hum (like a motor)
signal = (
    0.5 * np.sin(2 * np.pi * base_freq * t) +       # Fundamental
    0.3 * np.sin(2 * np.pi * base_freq * 2 * t) +   # 2nd harmonic
    0.1 * np.sin(2 * np.pi * base_freq * 3 * t) +   # 3rd harmonic
    0.05 * np.random.randn(len(t))                   # Noise
)

# Normalize to 16-bit range
signal = np.int16(signal / np.max(np.abs(signal)) * 32767)

# Save as WAV file
output_path = "data/test/normal/sample.wav"
wavfile.write(output_path, sr, signal)
print(f"Created: {output_path}")

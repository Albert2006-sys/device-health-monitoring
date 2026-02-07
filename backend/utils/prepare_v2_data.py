"""
V2 Data Preparation Script

Prepares car audio WAV files for V2 model training:
- Loads all WAV files from healthy/ and anomalous/ directories
- Extracts 13 features per file
- Saves processed data to backend/data/v2/
"""

import os
import sys
import numpy as np
import librosa
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_features

# Configuration
SAMPLE_RATE = 22050
BASE_DIR = Path(__file__).parent.parent.parent  # GRASP/
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "v2"

# Directory mappings
HEALTHY_DIR = BASE_DIR / "healthy"
ANOMALOUS_DIR = BASE_DIR / "anomalous"

# Excluded directories (noise, not faults)
EXCLUDED_DIRS = ["cars"]

# Label mapping for faults
FAULT_LABELS = {
    "worn_out_brakes": "worn_brakes",
    "bad_ignition": "bad_ignition",
    "dead_battery": "dead_battery",
    "ai_mechanic_export_training": "mixed_faults",
    "ai_mechanic_export_testing": "mixed_faults",
}


def get_wav_files(directory: Path, recursive: bool = True) -> list:
    """Get all WAV files in directory."""
    if recursive:
        return list(directory.rglob("*.wav"))
    return list(directory.glob("*.wav"))


def extract_features_from_wav(file_path: Path) -> np.ndarray:
    """Load WAV and extract 13 features."""
    try:
        # Load audio
        y, sr = librosa.load(str(file_path), sr=SAMPLE_RATE)
        
        if len(y) < SAMPLE_RATE:  # Less than 1 second
            # Pad with zeros
            y = np.pad(y, (0, SAMPLE_RATE - len(y)))
        
        # Use first 1 second only
        y = y[:SAMPLE_RATE]
        
        # Extract features
        features = extract_features(y, SAMPLE_RATE)
        return np.array(features)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_healthy_data():
    """Process all healthy directory files."""
    print("\n=== Processing Healthy Data ===")
    
    all_features = []
    file_count = 0
    
    for subdir in HEALTHY_DIR.iterdir():
        if not subdir.is_dir():
            continue
        
        wav_files = get_wav_files(subdir)
        print(f"  {subdir.name}: {len(wav_files)} files")
        
        for wav_file in wav_files:
            features = extract_features_from_wav(wav_file)
            if features is not None:
                all_features.append(features)
                file_count += 1
    
    print(f"Total healthy samples: {file_count}")
    return np.array(all_features)


def process_anomalous_data():
    """Process all anomalous directory files with labels."""
    print("\n=== Processing Anomalous Data ===")
    
    all_features = []
    all_labels = []
    file_count = 0
    
    for subdir in ANOMALOUS_DIR.iterdir():
        if not subdir.is_dir():
            continue
        
        # Skip excluded directories
        if subdir.name in EXCLUDED_DIRS:
            print(f"  SKIPPING {subdir.name} (excluded)")
            continue
        
        # Determine label
        label = "unknown_fault"
        for key, value in FAULT_LABELS.items():
            if key in subdir.name:
                label = value
                break
        
        wav_files = get_wav_files(subdir)
        print(f"  {subdir.name}: {len(wav_files)} files -> label: {label}")
        
        for wav_file in wav_files:
            features = extract_features_from_wav(wav_file)
            if features is not None:
                all_features.append(features)
                all_labels.append(label)
                file_count += 1
    
    print(f"Total anomalous samples: {file_count}")
    return np.array(all_features), all_labels


def main():
    """Main data preparation pipeline."""
    print("=" * 60)
    print("V2 DATA PREPARATION")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process healthy data
    X_healthy = process_healthy_data()
    
    # Process anomalous data
    X_anomalous, y_anomalous = process_anomalous_data()
    
    # Create labels for classification
    # healthy = 0, faults = 1+ (encoded)
    unique_labels = sorted(set(y_anomalous))
    label_to_idx = {label: idx + 1 for idx, label in enumerate(unique_labels)}
    label_to_idx["healthy"] = 0
    
    # Print label mapping
    print("\n=== Label Mapping ===")
    for label, idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx}: {label}")
    
    # Create combined dataset for Random Forest
    y_healthy = np.zeros(len(X_healthy), dtype=int)
    y_anomalous_encoded = np.array([label_to_idx[l] for l in y_anomalous])
    
    X_all = np.vstack([X_healthy, X_anomalous])
    y_all = np.concatenate([y_healthy, y_anomalous_encoded])
    
    # Save data
    print("\n=== Saving Data ===")
    
    # Healthy only (for autoencoder)
    np.save(OUTPUT_DIR / "X_healthy.npy", X_healthy)
    print(f"  Saved X_healthy.npy: shape {X_healthy.shape}")
    
    # All data with labels (for Random Forest)
    np.save(OUTPUT_DIR / "X_all.npy", X_all)
    np.save(OUTPUT_DIR / "y_all.npy", y_all)
    print(f"  Saved X_all.npy: shape {X_all.shape}")
    print(f"  Saved y_all.npy: shape {y_all.shape}")
    
    # Save label mapping
    import json
    with open(OUTPUT_DIR / "label_mapping.json", "w") as f:
        json.dump(label_to_idx, f, indent=2)
    print(f"  Saved label_mapping.json")
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    
    return X_healthy, X_all, y_all, label_to_idx


if __name__ == "__main__":
    main()

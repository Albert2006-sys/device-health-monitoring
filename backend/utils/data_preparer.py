"""
Data Preparer for CWRU Bearing Dataset

Loads MATLAB vibration files, extracts features using sliding windows,
and saves feature matrices for training and testing.

Dataset: CWRU Bearing Dataset
Sampling rate: 12,000 Hz
Window size: 1 second (12,000 samples)
"""

import os
import numpy as np
from scipy.io import loadmat
from feature_extractor import extract_features


# Configuration
SAMPLE_RATE = 12000
WINDOW_SIZE = 12000  # 1 second at 12kHz
STEP_SIZE = 12000    # No overlap

# File paths
NORMAL_FILE = "../data/train/normal/Normal_0.mat"
FAULT_FILE = "../data/test/faulty/IR007_0_1797.mat"
OUTPUT_DIR = "../data"


def find_de_signal(mat_data: dict) -> np.ndarray:
    """
    Find and extract the Drive End (DE) accelerometer signal.
    
    Args:
        mat_data: Dictionary from loadmat()
    
    Returns:
        1D numpy array of vibration signal
    """
    for key in mat_data.keys():
        if "_DE_" in key:
            signal = mat_data[key].flatten()
            return signal
    
    # Fallback: find any array that looks like signal data
    for key, value in mat_data.items():
        if not key.startswith("__") and isinstance(value, np.ndarray):
            if value.size > WINDOW_SIZE:
                return value.flatten()
    
    raise ValueError("Could not find DE signal in .mat file")


def extract_windows(signal: np.ndarray) -> list:
    """
    Split signal into fixed-size windows.
    
    Args:
        signal: 1D vibration signal array
    
    Returns:
        List of 1D window arrays
    """
    windows = []
    num_windows = (len(signal) - WINDOW_SIZE) // STEP_SIZE + 1
    
    for i in range(num_windows):
        start = i * STEP_SIZE
        end = start + WINDOW_SIZE
        window = signal[start:end]
        windows.append(window)
    
    return windows


def process_file(file_path: str) -> np.ndarray:
    """
    Load a .mat file and extract features from all windows.
    
    Args:
        file_path: Path to MATLAB file
    
    Returns:
        2D numpy array of shape (num_windows, 13)
    """
    mat_data = loadmat(file_path)
    signal = find_de_signal(mat_data)
    windows = extract_windows(signal)
    
    features_list = []
    for window in windows:
        features = extract_features(window, SAMPLE_RATE)
        features_list.append(features)
    
    return np.array(features_list)


def main():
    """Main execution: process normal and fault data, save outputs."""
    
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    normal_path = os.path.join(script_dir, NORMAL_FILE)
    fault_path = os.path.join(script_dir, FAULT_FILE)
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    
    # Process normal data
    print(f"Processing normal data: {NORMAL_FILE}")
    X_train = process_file(normal_path)
    print(f"  Shape: {X_train.shape}")
    
    # Process fault data
    print(f"Processing fault data: {FAULT_FILE}")
    X_fault = process_file(fault_path)
    print(f"  Shape: {X_fault.shape}")
    
    # Save outputs
    train_output = os.path.join(output_dir, "X_train.npy")
    fault_output = os.path.join(output_dir, "X_fault.npy")
    
    np.save(train_output, X_train)
    np.save(fault_output, X_fault)
    
    print(f"\nSaved: {train_output}")
    print(f"Saved: {fault_output}")
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_fault shape: {X_fault.shape}")


if __name__ == "__main__":
    main()

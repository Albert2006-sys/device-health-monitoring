"""
Random Forest Training for Bearing Fault Classification

Classifies bearing condition as Normal (0) or Faulty (1)
after anomaly detection by autoencoder.

Input: X_train.npy + X_fault.npy
Output: random_forest.pkl
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Configuration
N_ESTIMATORS = 100
MAX_DEPTH = 10
RANDOM_STATE = 42
TEST_SIZE = 0.2

# File paths (relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
X_FAULT_PATH = os.path.join(DATA_DIR, "X_fault.npy")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")

# Feature names for importance display
FEATURE_NAMES = [
    "RMS Energy",
    "Zero Crossing Rate",
    "Spectral Centroid",
    "Spectral Bandwidth",
    "Spectral Kurtosis",
    "Dominant Frequency",
    "MFCC 1",
    "MFCC 2",
    "MFCC 3",
    "MFCC 4",
    "MFCC 5",
    "MFCC 6",
    "MFCC 7"
]


def main():
    """Train Random Forest classifier for fault detection."""
    
    # Load data
    print("Loading data...")
    X_normal = np.load(X_TRAIN_PATH)
    X_fault = np.load(X_FAULT_PATH)
    print(f"  Normal samples: {X_normal.shape}")
    print(f"  Faulty samples: {X_fault.shape}")
    
    # Load existing scaler
    print("\nLoading scaler...")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    # Normalize features
    X_normal_scaled = scaler.transform(X_normal)
    X_fault_scaled = scaler.transform(X_fault)
    
    # Create labels
    y_normal = np.zeros(len(X_normal))  # Label 0 = Normal
    y_fault = np.ones(len(X_fault))     # Label 1 = Faulty
    
    # Combine datasets
    X = np.vstack([X_normal_scaled, X_fault_scaled])
    y = np.concatenate([y_normal, y_fault])
    print(f"\nCombined dataset: {X.shape}")
    print(f"  Class 0 (Normal): {np.sum(y == 0)}")
    print(f"  Class 1 (Faulty): {np.sum(y == 1)}")
    
    # Shuffle and split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y
    )
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Faulty"]))
    
    # Class probabilities for test samples
    print("Test Sample Probabilities:")
    print("-" * 40)
    for i, (true, pred, proba) in enumerate(zip(y_test, y_pred, y_proba)):
        label = "Normal" if true == 0 else "Faulty"
        print(f"  Sample {i+1}: True={label:6s} | P(Normal)={proba[0]:.3f} | P(Faulty)={proba[1]:.3f}")
    
    # Feature importance (top 5)
    print("\n" + "="*50)
    print("TOP 5 FEATURE IMPORTANCE")
    print("="*50)
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1][:5]
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank}. {FEATURE_NAMES[idx]}: {importance[idx]:.4f}")
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    print(f"\nSaved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()

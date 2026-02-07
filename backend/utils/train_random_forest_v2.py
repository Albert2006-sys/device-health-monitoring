"""
V2 Random Forest Training Script

Trains multi-class Random Forest for car fault classification.
Classes: healthy, worn_brakes, bad_ignition, dead_battery, mixed_faults
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Configuration
N_ESTIMATORS = 100
MAX_DEPTH = 10
RANDOM_STATE = 42

DATA_DIR = Path(__file__).parent.parent / "data" / "v2"
MODELS_DIR = Path(__file__).parent.parent / "models" / "v2"


def train_random_forest():
    """Train Random Forest on all labeled data."""
    print("=" * 60)
    print("V2 RANDOM FOREST TRAINING")
    print("=" * 60)
    
    # Load data
    X_all = np.load(DATA_DIR / "X_all.npy")
    y_all = np.load(DATA_DIR / "y_all.npy")
    print(f"Loaded data: X={X_all.shape}, y={y_all.shape}")
    
    # Handle NaN/Inf values
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Load label mapping
    with open(DATA_DIR / "label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    idx_to_label = {v: k for k, v in label_mapping.items()}
    print(f"Classes: {list(label_mapping.keys())}")
    
    # Load V2 scaler
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Scale features
    X_scaled = scaler.transform(X_all)
    
    # Show class distribution
    print("\n--- Class Distribution ---")
    unique, counts = np.unique(y_all, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {idx_to_label[u]}: {c} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train classifier with balanced weights
    print("\n--- Training ---")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("Training complete!")
    
    # Evaluate
    print("\n--- Evaluation ---")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Feature importances
    print("\n--- Top 5 Feature Importances ---")
    feature_names = [
        "rms_energy", "zcr", "spec_centroid", "spec_bandwidth", "spec_kurtosis",
        "dom_freq", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7"
    ]
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:5]
    for i in indices:
        print(f"  {feature_names[i]}: {importances[i]:.4f}")
    
    # Save model
    print("\n--- Saving Model ---")
    with open(MODELS_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump(clf, f)
    print(f"  Saved random_forest.pkl")
    
    # Save label mapping to models dir
    with open(MODELS_DIR / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)
    print(f"  Saved label_mapping.json")
    
    print("\n" + "=" * 60)
    print("V2 RANDOM FOREST TRAINING COMPLETE")
    print("=" * 60)
    
    return clf


if __name__ == "__main__":
    train_random_forest()

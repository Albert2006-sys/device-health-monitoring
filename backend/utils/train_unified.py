"""
Unified Model Training - Combines V1 (CWRU) + V2 (Car Audio) Datasets

Creates a single model that handles both industrial bearing faults 
AND car audio diagnostics.
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
MODELS_DIR = SCRIPT_DIR.parent / "models"

# Training params
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
THRESHOLD_PERCENTILE = 95


class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection."""
    def __init__(self, input_dim: int = 13):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_v1_data():
    """Load V1 (CWRU) preprocessed data."""
    print("\n=== Loading V1 Data (CWRU Bearing) ===")
    
    X_train_path = DATA_DIR / "X_train.npy"
    X_fault_path = DATA_DIR / "X_fault.npy"
    
    if not X_train_path.exists():
        print("  V1 training data not found, skipping...")
        return None, None
    
    X_healthy = np.load(X_train_path)
    print(f"  V1 healthy: {X_healthy.shape}")
    
    if X_fault_path.exists():
        X_fault = np.load(X_fault_path)
        print(f"  V1 faulty: {X_fault.shape}")
    else:
        X_fault = None
        
    return X_healthy, X_fault


def load_v2_data():
    """Load V2 (Car Audio) preprocessed data."""
    print("\n=== Loading V2 Data (Car Audio) ===")
    
    v2_dir = DATA_DIR / "v2"
    
    if not v2_dir.exists():
        print("  V2 data not found, skipping...")
        return None, None, None, None
    
    X_healthy = np.load(v2_dir / "X_healthy.npy")
    X_all = np.load(v2_dir / "X_all.npy")
    y_all = np.load(v2_dir / "y_all.npy")
    
    with open(v2_dir / "label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    
    print(f"  V2 healthy: {X_healthy.shape}")
    print(f"  V2 all: {X_all.shape}, labels: {len(set(y_all))}")
    
    return X_healthy, X_all, y_all, label_mapping


def train_unified_autoencoder(X_healthy_combined: np.ndarray):
    """Train autoencoder on combined healthy data."""
    print("\n" + "=" * 60)
    print("TRAINING UNIFIED AUTOENCODER")
    print("=" * 60)
    
    # Handle NaN/Inf
    X_healthy_combined = np.nan_to_num(X_healthy_combined, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_healthy_combined)
    print(f"Total healthy samples: {len(X_scaled)}")
    
    # Split train/val
    n_train = int(0.8 * len(X_scaled))
    X_train = X_scaled[:n_train]
    X_val = X_scaled[n_train:]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Create loaders
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), 
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val)), 
                            batch_size=BATCH_SIZE)
    
    # Train
    model = Autoencoder(13)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n--- Training ---")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            X = batch[0]
            optimizer.zero_grad()
            loss = criterion(model(X), X)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                X = batch[0]
                val_loss += criterion(model(X), X).item()
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS}: Train={train_loss/len(train_loader):.6f}, Val={val_loss/len(val_loader):.6f}")
    
    # Compute threshold
    print("\n--- Computing Threshold ---")
    model.eval()
    with torch.no_grad():
        X_all = torch.FloatTensor(X_scaled)
        X_pred = model(X_all).numpy()
        errors = np.mean(np.square(X_scaled - X_pred), axis=1)
    
    threshold = np.percentile(errors, THRESHOLD_PERCENTILE)
    print(f"Threshold ({THRESHOLD_PERCENTILE}th percentile): {threshold:.6f}")
    
    return model, scaler, threshold


def train_unified_random_forest(X_all: np.ndarray, y_all: np.ndarray, 
                                 scaler: StandardScaler, label_mapping: dict):
    """Train random forest on combined labeled data."""
    print("\n" + "=" * 60)
    print("TRAINING UNIFIED RANDOM FOREST")
    print("=" * 60)
    
    # Handle NaN/Inf
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale
    X_scaled = scaler.transform(X_all)
    
    # Show class distribution
    idx_to_label = {v: k for k, v in label_mapping.items()}
    print("\n--- Class Distribution ---")
    unique, counts = np.unique(y_all, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {idx_to_label[u]}: {c} samples")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    print("\n--- Training ---")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("\n--- Evaluation ---")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return clf


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("UNIFIED MODEL TRAINING (V1 + V2)")
    print("=" * 60)
    
    # Load V1 data
    v1_healthy, v1_fault = load_v1_data()
    
    # Load V2 data
    v2_healthy, v2_all, v2_labels, v2_label_map = load_v2_data()
    
    # Combine healthy data for autoencoder
    healthy_datasets = []
    if v1_healthy is not None:
        healthy_datasets.append(v1_healthy)
    if v2_healthy is not None:
        healthy_datasets.append(v2_healthy)
    
    if not healthy_datasets:
        print("ERROR: No healthy data found!")
        return
    
    X_healthy_combined = np.vstack(healthy_datasets)
    print(f"\nCombined healthy samples: {X_healthy_combined.shape}")
    
    # Train autoencoder
    model, scaler, threshold = train_unified_autoencoder(X_healthy_combined)
    
    # Build unified label mapping
    # Start from V2 mapping, add V1 bearing_fault
    unified_label_map = v2_label_map.copy() if v2_label_map else {"healthy": 0}
    max_label = max(unified_label_map.values()) if unified_label_map else 0
    
    if v1_fault is not None and "bearing_fault" not in unified_label_map:
        unified_label_map["bearing_fault"] = max_label + 1
    
    print(f"\nUnified label mapping: {unified_label_map}")
    
    # Combine all labeled data for random forest
    all_features = []
    all_labels = []
    
    # Add V1 healthy
    if v1_healthy is not None:
        all_features.append(v1_healthy)
        all_labels.extend([unified_label_map["healthy"]] * len(v1_healthy))
    
    # Add V1 faulty
    if v1_fault is not None:
        all_features.append(v1_fault)
        all_labels.extend([unified_label_map["bearing_fault"]] * len(v1_fault))
    
    # Add V2 data (already has labels)
    if v2_all is not None:
        all_features.append(v2_all)
        all_labels.extend(v2_labels.tolist())
    
    X_all_combined = np.vstack(all_features)
    y_all_combined = np.array(all_labels)
    
    print(f"Combined labeled samples: {X_all_combined.shape}")
    
    # Train random forest
    clf = train_unified_random_forest(X_all_combined, y_all_combined, 
                                       scaler, unified_label_map)
    
    # Save models (overwrite V1 location as unified)
    print("\n=== Saving Unified Models ===")
    
    torch.save(model.state_dict(), MODELS_DIR / "autoencoder.pth")
    print("  Saved autoencoder.pth")
    
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("  Saved scaler.pkl")
    
    np.save(MODELS_DIR / "threshold.npy", threshold)
    print("  Saved threshold.npy")
    
    with open(MODELS_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("  Saved random_forest.pkl")
    
    with open(MODELS_DIR / "label_mapping.json", "w") as f:
        json.dump(unified_label_map, f, indent=2)
    print("  Saved label_mapping.json")
    
    print("\n" + "=" * 60)
    print("UNIFIED MODEL TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

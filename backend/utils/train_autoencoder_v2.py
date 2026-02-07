"""
V2 Autoencoder Training Script

Trains autoencoder on healthy car audio data for anomaly detection.
Outputs: autoencoder.pth, scaler.pkl, threshold.npy
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Configuration
INPUT_DIM = 13
HIDDEN_DIM_1 = 16
HIDDEN_DIM_2 = 8
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
THRESHOLD_PERCENTILE = 95

DATA_DIR = Path(__file__).parent.parent / "data" / "v2"
MODELS_DIR = Path(__file__).parent.parent / "models" / "v2"


class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection."""
    def __init__(self, input_dim: int = 13):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_1, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder():
    """Train autoencoder on healthy data."""
    print("=" * 60)
    print("V2 AUTOENCODER TRAINING")
    print("=" * 60)
    
    # Load healthy data
    X_healthy = np.load(DATA_DIR / "X_healthy.npy")
    print(f"Loaded healthy data: {X_healthy.shape}")
    
    # Handle NaN/Inf values
    X_healthy = np.nan_to_num(X_healthy, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_healthy)
    print(f"Fitted scaler: mean={scaler.mean_[:3]}..., std={scaler.scale_[:3]}...")
    
    # Split train/val
    n_train = int(0.8 * len(X_scaled))
    X_train = X_scaled[:n_train]
    X_val = X_scaled[n_train:]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = Autoencoder(INPUT_DIM)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n--- Training ---")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            X = batch[0]
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                X = batch[0]
                output = model(X)
                loss = criterion(output, X)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    # Compute reconstruction errors on all healthy data
    print("\n--- Computing Threshold ---")
    model.eval()
    errors = []
    with torch.no_grad():
        X_all = torch.FloatTensor(X_scaled)
        X_pred = model(X_all).numpy()
        errors = np.mean(np.square(X_scaled - X_pred), axis=1)
    
    threshold = np.percentile(errors, THRESHOLD_PERCENTILE)
    print(f"Error stats: min={errors.min():.6f}, max={errors.max():.6f}, mean={errors.mean():.6f}")
    print(f"Threshold ({THRESHOLD_PERCENTILE}th percentile): {threshold:.6f}")
    
    # Save models
    print("\n--- Saving Models ---")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save autoencoder
    torch.save(model.state_dict(), MODELS_DIR / "autoencoder.pth")
    print(f"  Saved autoencoder.pth")
    
    # Save scaler
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler.pkl")
    
    # Save threshold
    np.save(MODELS_DIR / "threshold.npy", threshold)
    print(f"  Saved threshold.npy")
    
    print("\n" + "=" * 60)
    print("V2 AUTOENCODER TRAINING COMPLETE")
    print("=" * 60)
    
    return model, scaler, threshold


if __name__ == "__main__":
    train_autoencoder()

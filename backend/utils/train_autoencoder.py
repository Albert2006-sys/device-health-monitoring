"""
Autoencoder Training for Anomaly Detection (PyTorch)

Trains an autoencoder on normal bearing vibration features.
High reconstruction error indicates anomaly (fault).

Input: X_train.npy (normal samples)
Output: autoencoder.pth + scaler.pkl
"""

import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Configuration
EPOCHS = 40
BATCH_SIZE = 8
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
THRESHOLD_PERCENTILE = 95

# File paths (relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
X_FAULT_PATH = os.path.join(DATA_DIR, "X_fault.npy")
MODEL_PATH = os.path.join(MODELS_DIR, "autoencoder.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")


class Autoencoder(nn.Module):
    """
    Autoencoder for anomaly detection.
    Architecture: Input → 16 → 8 → 16 → Output
    """
    def __init__(self, input_dim: int):
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def compute_reconstruction_error(model, X: np.ndarray) -> np.ndarray:
    """
    Compute MSE between input and reconstruction.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        X_pred = model(X_tensor).numpy()
        mse = np.mean(np.square(X - X_pred), axis=1)
    return mse


def main():
    """Train autoencoder and evaluate on normal vs fault data."""
    
    # Load data
    print("Loading data...")
    X_train = np.load(X_TRAIN_PATH)
    X_fault = np.load(X_FAULT_PATH)
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_fault shape: {X_fault.shape}")
    
    # Normalize features (fit only on training data)
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_fault_scaled = scaler.transform(X_fault).astype(np.float32)
    
    # Split for validation
    X_tr, X_val = train_test_split(X_train_scaled, test_size=VALIDATION_SPLIT, random_state=42)
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(X_tr))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Build model
    print("\nBuilding autoencoder...")
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim)
    print(model)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\nTraining autoencoder...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} - Train Loss: {train_loss/len(train_loader):.6f} - Val Loss: {val_loss/len(val_loader):.6f}")
    
    # Compute reconstruction errors
    print("\nEvaluating reconstruction errors...")
    errors_normal = compute_reconstruction_error(model, X_train_scaled)
    errors_fault = compute_reconstruction_error(model, X_fault_scaled)
    
    # Calculate anomaly threshold
    threshold = np.percentile(errors_normal, THRESHOLD_PERCENTILE)
    
    # Print results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Mean reconstruction error (normal):  {np.mean(errors_normal):.6f}")
    print(f"Mean reconstruction error (faulty):  {np.mean(errors_fault):.6f}")
    print(f"Anomaly threshold (95th percentile): {threshold:.6f}")
    print("="*50)
    
    # Classify samples using threshold
    normal_detected = np.sum(errors_normal > threshold)
    fault_detected = np.sum(errors_fault > threshold)
    print(f"\nNormal samples flagged as anomaly: {normal_detected}/{len(errors_normal)}")
    print(f"Fault samples detected as anomaly: {fault_detected}/{len(errors_fault)}")
    
    # Save model and scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler: {SCALER_PATH}")
    
    # Save threshold for inference
    threshold_path = os.path.join(MODELS_DIR, "threshold.npy")
    np.save(threshold_path, threshold)
    print(f"Saved threshold: {threshold_path}")


if __name__ == "__main__":
    main()

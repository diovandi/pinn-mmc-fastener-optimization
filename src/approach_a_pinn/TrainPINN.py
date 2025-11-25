import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DATA_PATH = (ROOT_DIR / "data/results/pinn_training_data.csv").resolve()
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "pinn_model.pth"
STATS_PATH = ARTIFACT_DIR / "norm_stats.npz"
HIDDEN_SIZE = 64
LR = 0.001
EPOCHS = 5000

# --- 1. Load Data ---
if not DATA_PATH.exists():
    print(f"❌ Error: {DATA_PATH} not found. Run Julia generator first.")
    exit()

# Load CSV (No header, columns: x1, y1, x2, y2, Compliance)
raw_data = pd.read_csv(DATA_PATH, header=None).values
print(f"Loaded {len(raw_data)} samples.")

# Split Inputs (Screws) and Target (Compliance)
X_raw = raw_data[:, 0:4].astype(np.float32) # x1, y1, x2, y2
y_raw = raw_data[:, 4:5].astype(np.float32) # Compliance

# --- 2. Normalization (Crucial for NN) ---
# We standardize to mean=0, std=1
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
y_mean = y_raw.mean(axis=0)
y_std = y_raw.std(axis=0)

# Avoid division by zero if std is 0 (unlikely but safe)
X_std[X_std == 0] = 1.0
y_std[y_std == 0] = 1.0

X_norm = (X_raw - X_mean) / X_std
y_norm = (y_raw - y_mean) / y_std

# Convert to PyTorch Tensors
X_train = torch.tensor(X_norm)
y_train = torch.tensor(y_norm)

# --- 3. Define PINN (Surrogate Model) ---
class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN_SIZE),
            nn.Tanh(), # Tanh often works better for physics/smooth functions
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SurrogateModel()
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# --- 4. Training Loop ---
print("Starting Training...")
loss_history = []

for epoch in range(EPOCHS):
    # Forward pass
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# --- 5. Validation & Saving ---
print(f"Final Loss: {loss_history[-1]:.6f}")

# Save Model weights
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
# Save Normalization stats (Required for inference!)
np.savez(STATS_PATH, X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Stats saved to {STATS_PATH}")

# Quick Sanity Check
with torch.no_grad():
    test_pred_norm = model(X_train)
    # Denormalize
    test_pred = test_pred_norm * torch.tensor(y_std) + torch.tensor(y_mean)
    
    # Compare first 5 samples
    print("\n--- Sanity Check (First 5 Samples) ---")
    print(f"{'True Compl':<15} | {'Pred Compl':<15} | {'Error %':<10}")
    for i in range(5):
        truth = y_raw[i][0]
        pred = test_pred[i].item()
        err = abs(pred - truth) / truth * 100
        print(f"{truth:<15.2f} | {pred:<15.2f} | {err:<10.2f}")

# Plot Loss
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("PINN Training Convergence")
plt.yscale('log')
plt.savefig("training_curve.png")
print("Chart saved to training_curve.png")

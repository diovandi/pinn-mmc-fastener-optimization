import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DATA_PATH = (ROOT_DIR / "data/cantilever/iter_6_training_data.csv").resolve()
ARTIFACT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = ROOT_DIR / "data/cantilever"
FIGURES_DIR = ROOT_DIR / "figures/cantilever"
MODEL_PATH = ARTIFACT_DIR / "pinn_cantilever.pth"
STATS_PATH = ARTIFACT_DIR / "norm_stats_cantilever.npz"
HIDDEN_SIZES = [32, 64, 32]
LR = 0.001
EPOCHS = 5000

# Create directories
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Load Data ---
if not DATA_PATH.exists():
    print(f"❌ Error: {DATA_PATH} not found. Run Iteration6_TrainingData.jl first.")
    exit(1)

# Load CSV (header: support_pos,compliance,iteration)
raw_data = pd.read_csv(DATA_PATH)
print(f"Loaded {len(raw_data)} samples.")

# Split Inputs (support position) and Target (Compliance)
X_raw = raw_data['support_pos'].values.astype(np.float32).reshape(-1, 1)
y_raw = raw_data['compliance'].values.astype(np.float32).reshape(-1, 1)

# --- 2. Normalization (Crucial for NN) ---
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
y_mean = y_raw.mean(axis=0)
y_std = y_raw.std(axis=0)

# Avoid division by zero
X_std[X_std == 0] = 1.0
y_std[y_std == 0] = 1.0

X_norm = (X_raw - X_mean) / X_std
y_norm = (y_raw - y_mean) / y_std

# Convert to PyTorch Tensors
X_train = torch.tensor(X_norm, dtype=torch.float32)
y_train = torch.tensor(y_norm, dtype=torch.float32)

# --- 3. Define PINN (Surrogate Model) ---
class SurrogateModel(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers = []
        input_size = 1
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = SurrogateModel(HIDDEN_SIZES)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# --- 4. Training Loop ---
print("Starting Training...")
loss_history = []
val_loss_history = []

# Simple train/val split (80/20)
n_train = int(0.8 * len(X_train))
X_train_split = X_train[:n_train]
y_train_split = y_train[:n_train]
X_val = X_train[n_train:]
y_val = y_train[n_train:]

for epoch in range(EPOCHS):
    # Training
    model.train()
    y_pred = model(X_train_split)
    loss = loss_fn(y_pred, y_train_split)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    # Validation
    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_val)
        val_loss = loss_fn(y_pred_val, y_val)
        val_loss_history.append(val_loss.item())
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6e}, Val Loss = {val_loss.item():.6e}")

# --- 5. Save Results ---
print(f"\nFinal Training Loss: {loss_history[-1]:.6e}")
print(f"Final Validation Loss: {val_loss_history[-1]:.6e}")

# Save Model weights
torch.save(model.state_dict(), MODEL_PATH)
np.savez(STATS_PATH, X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Stats saved to {STATS_PATH}")

# Save training log
training_log = pd.DataFrame({
    'epoch': range(EPOCHS),
    'loss': loss_history,
    'val_loss': val_loss_history
})
training_log.to_csv(OUTPUT_DIR / "iter_7-9_pinn_training.csv", index=False)
print(f"✅ Training log saved to {OUTPUT_DIR / 'iter_7-9_pinn_training.csv'}")

# Plot Loss
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("PINN Training Convergence (Cantilever)")
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig(FIGURES_DIR / "pinn_training_curve.png", dpi=150)
print(f"✅ Training curve saved to {FIGURES_DIR / 'pinn_training_curve.png'}")

# Quick Sanity Check
model.eval()
with torch.no_grad():
    test_pred_norm = model(X_train)
    test_pred = test_pred_norm * torch.tensor(y_std) + torch.tensor(y_mean)
    
    print("\n--- Sanity Check (First 5 Samples) ---")
    print(f"{'True Compl':<15} | {'Pred Compl':<15} | {'Error %':<10}")
    for i in range(min(5, len(X_train))):
        truth = y_raw[i][0]
        pred = test_pred[i].item()
        err = abs(pred - truth) / truth * 100 if truth != 0 else 0
        print(f"{truth:<15.6e} | {pred:<15.6e} | {err:<10.2f}")


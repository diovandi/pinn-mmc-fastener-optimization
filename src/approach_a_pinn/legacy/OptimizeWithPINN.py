import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import os

# --- Configuration ---
MODEL_PATH = "pinn_model.pth"
STATS_PATH = "norm_stats.npz"
HIDDEN_SIZE = 64

# --- 1. Define Model Architecture (Must match Training) ---
class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, x):
        return self.net(x)

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(STATS_PATH):
        print("❌ Error: Model/Stats not found.")
        return None, None
    
    # Load Stats
    stats = np.load(STATS_PATH)
    X_mean = torch.tensor(stats['X_mean'], dtype=torch.float32)
    X_std = torch.tensor(stats['X_std'], dtype=torch.float32)
    y_mean = torch.tensor(stats['y_mean'], dtype=torch.float32)
    y_std = torch.tensor(stats['y_std'], dtype=torch.float32)
    
    # Load Model
    model = SurrogateModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # Set to evaluation mode
    
    return model, (X_mean, X_std, y_mean, y_std)

def enforce_lbracket_bounds(screws):
    # Screws tensor shape: [4]
    # Hardcoded constraints for L-Bracket (25mm thick)
    THICK = 25.0
    MAX_DIM = 100.0
    PADDING = 5.0
    
    new_screws = screws.clone()
    
    # Clamp outer bounds
    new_screws = torch.clamp(new_screws, PADDING, MAX_DIM - PADDING)
    
    # Handle Void (x > 25 and y > 25)
    # Since PyTorch graph must be differentiable, we use soft constraints or iterative projection.
    # For inference loop, hard projection is fine (just like in Julia).
    
    for i in range(2): # 2 screws
        x = new_screws[2*i]
        y = new_screws[2*i+1]
        
        if x > THICK and y > THICK:
            dist_v = x - THICK
            dist_h = y - THICK
            if dist_v < dist_h:
                new_screws[2*i] = THICK - PADDING
            else:
                new_screws[2*i+1] = THICK - PADDING
                
    return new_screws

def optimize_fasteners():
    print("=== AI-Accelerated Screw Placement ===")
    
    model, stats = load_model()
    if model is None: return
    
    X_mean, X_std, y_mean, y_std = stats
    
    # --- 1. Initialization ---
    # Start at a "bad" spot (e.g., center of void, then projected)
    # Or specific coords: (80, 10) and (10, 80)
    init_coords = torch.tensor([80.0, 10.0, 10.0, 80.0], dtype=torch.float32, requires_grad=True)
    
    print(f"Initial Screws: {init_coords.detach().numpy()}")
    
    # Optimizer (ADAM is standard for this)
    optimizer = torch.optim.Adam([init_coords], lr=1.0) # Fast learning rate
    
    start_time = time.time()
    
    # --- 2. Optimization Loop ---
    for i in range(100):
        optimizer.zero_grad()
        
        # Normalize Input
        # (Input - Mean) / Std
        norm_input = (init_coords - X_mean) / X_std
        
        # Forward Pass (Predict Compliance)
        norm_pred = model(norm_input)
        
        # Denormalize Output (for logging, though optimizer works on normalized loss fine)
        compliance = norm_pred * y_std + y_mean
        
        # We want to MINIMIZE compliance
        loss = compliance
        
        loss.backward()
        optimizer.step()
        
        # Project constraints (No-gradient step)
        with torch.no_grad():
            projected = enforce_lbracket_bounds(init_coords)
            init_coords.data = projected
        
        if i % 10 == 0:
            print(f"Iter {i}: Compliance = {compliance.item():.2f}")
            
    end_time = time.time()
    duration = end_time - start_time
    
    print("-" * 30)
    print(f"✅ Final Result in {duration:.4f} seconds")
    print(f"Final Compliance: {compliance.item():.2f}")
    print(f"Final Screws: {init_coords.detach().numpy()}")
    print("-" * 30)

if __name__ == "__main__":
    optimize_fasteners()

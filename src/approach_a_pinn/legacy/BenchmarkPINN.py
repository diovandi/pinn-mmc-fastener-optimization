import torch
import torch.nn as nn
import numpy as np
import time
import os

# --- Configuration ---
MODEL_PATH = "pinn_model.pth"
STATS_PATH = "norm_stats.npz"
HIDDEN_SIZE = 64
N_ITERATIONS = 1000  # Running 1000 iterations for stable timing

# --- Model Architecture (Must match) ---
class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def run_pinn_benchmark():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(STATS_PATH):
        print("❌ Error: Model/Stats not found. Cannot benchmark PINN.")
        return

    # Load Model and Stats
    model = SurrogateModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    stats = np.load(STATS_PATH)
    X_mean = torch.tensor(stats['X_mean'], dtype=torch.float32)
    X_std = torch.tensor(stats['X_std'], dtype=torch.float32)
    
    # Initialize a dummy variable for the optimization loop
    screws = torch.tensor([50.0, 50.0, 50.0, 50.0], dtype=torch.float32, requires_grad=True)

    print(f"--- Running PINN Benchmark ({N_ITERATIONS} steps) ---")
    
    start_time = time.time()

    # The benchmark loop: Forward pass + Backward pass + Optimizer step
    for i in range(N_ITERATIONS):
        # 1. Forward Pass
        norm_input = (screws - X_mean) / X_std
        loss = model(norm_input)
        
        # 2. Backward Pass (Gradient calculation)
        loss.backward()
        
        # 3. Dummy Optimizer Step (to simulate the full work cycle)
        with torch.no_grad():
            screws -= 0.001 * screws.grad
            screws.grad.zero_()
            
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_iteration = total_time / N_ITERATIONS
    
    print("-" * 40)
    print(f"✅ PINN Total Time: {total_time:.4f} seconds")
    print(f"✅ PINN Time per Iteration: {time_per_iteration * 1000:.4f} ms/iter")
    print("-" * 40)
    
if __name__ == "__main__":
    # Activate the PyTorch environment
    # Note: Running this command will not activate the environment inside a python script, 
    # but the environment is active from the terminal before execution.
    run_pinn_benchmark()

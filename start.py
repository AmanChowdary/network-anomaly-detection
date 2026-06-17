"""
Render.com startup script.
Generates data + trains model if artifacts are missing, then starts the API.
"""
import os, subprocess, sys

def run(cmd):
    print(f">> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)

# Step 1: Generate data if missing
if not os.path.exists("data/network_traffic.csv"):
    print("Generating dataset...")
    os.makedirs("data", exist_ok=True)
    run("python -c \"from data.generate_data import generate_network_data; generate_network_data(n=50000).to_csv('data/network_traffic.csv', index=False)\"")

# Step 2: Train model if artifacts missing
if not os.path.exists("model/artifacts/anomaly_model.joblib"):
    print("Training model (first deploy — this takes ~90s)...")
    run("python model/train.py")

# Step 3: Start API
port = os.getenv("PORT", "8000")
print(f"Starting API on port {port}...")
os.execvp("uvicorn", ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", port])

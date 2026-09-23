import time
import torch
from neurafin_lnn import NeuraFinLNN

def run_demo():
    model = NeuraFinLNN()
    model.eval()

    # Simulated transaction batch (1 batch, 10 time steps, 4 features)
    mock_stream = torch.randn(1, 10, 4)

    start_time = time.perf_counter()
    with torch.no_grad():
        anomaly_scores = model(mock_stream)
    latency_ms = (time.perf_counter() - start_time) * 1000

    
print("=== NeuraFin LNN Core Demo ===")
    print(f"Inference Latency: {latency_ms:.2f} ms")
    print("Continuous ODE Anomaly Scores:")
    print(anomaly_scores.squeeze().numpy())

if __name__ == "__main__":
    run_demo()

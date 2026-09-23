import torch
from neurafin_lnn import NeuraFinLNN
from losses import ContinuousAnomalyLoss

def train_model(epochs: int = 5):
    model = NeuraFinLNN(input_dim=4, hidden_units=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = ContinuousAnomalyLoss(smoothness_weight=0.01)

    model.train()
    print("--- Starting LNN Training Pipeline ---")
    
    for epoch in range(1, epochs + 1):
        # Simulated batch (32 samples, 10 time steps, 4 features)
        inputs = torch.randn(32, 10, 4)
        targets = torch.randint(0, 2, (32, 10, 1)).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Clip gradients for numerical ODE stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"Epoch [{epoch}/{epochs}] | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_model()

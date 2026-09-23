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

        

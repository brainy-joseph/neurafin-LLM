import torch
import torch.nn as nn

class ContinuousAnomalyLoss(nn.Module):
    """Combines binary cross-entropy with dynamic trajectory smoothness regularization."""
    
    def __init__(self, smoothness_weight: float = 0.01):
        super().__init__()
        self.bce = nn.BCELoss()
        self.smoothness_weight = smoothness_weight

def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(predictions, targets)
        
        # First-order temporal derivative penalty (dy/dt dynamic smoothness check)
        temporal_diff = predictions[:, 1:] - predictions[:, :-1]
        smoothness_penalty = torch.mean(temporal_diff ** 2)
        
        total_loss = bce_loss + (self.smoothness_weight * smoothness_penalty)
        return total_loss

    

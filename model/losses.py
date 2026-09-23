import torch
import torch.nn as nn

class ContinuousAnomalyLoss(nn.Module):
    """Combines binary cross-entropy with dynamic trajectory smoothness regularization."""
    
    def __init__(self, smoothness_weight: float = 0.01):
        super().__init__()
        self.bce = nn.BCELoss()
        self.smoothness_weight = smoothness_weight

    

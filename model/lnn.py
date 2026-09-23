import torch
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP

class NeuraFinLNN(nn.Module):
    """Continuous-time Liquid Neural Network engine for financial anomaly scoring."""
    def __init__(self, input_dim=4, hidden_units=16):
        super(NeuraFinLNN, self).__init__()
        # AutoNCP sets up biological neural wiring topologies
        wiring = AutoNCP(hidden_units, output_size=1)
        self.ltc = LTC(input_dim, wiring, batch_first=True)

    

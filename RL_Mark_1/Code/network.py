import torch
import torch.nn as nn
import torch.nn.functional as F

class neural_network(nn.Module):
    def __init__(self, input_size, output_size, requires_grad = True):
        super().__init__()
        # create a neural network
        self.network = nn.Sequential(
            nn.Linear(input_size, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, output_size),
            nn.Sigmoid()
        )
        self.loss_fn = F.mse_loss
        self.optimiser = torch.optim.Adam(self.network.parameters())

        if not requires_grad:
            for parameter in self.network.parameters():
                parameter.requires_grad = False

    def forward(self, X):
        out = self.network(X)
        return out
    
    def load_checkpoint(self):
        pass

    def save_checkpoint(self):
        pass

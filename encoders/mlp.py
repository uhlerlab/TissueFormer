import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout=0.):
        super(MLP, self).__init__()

        layers = [nn.Linear(in_channels, hidden_channels), nn.BatchNorm1d(hidden_channels), nn.ELU(), nn.Dropout(dropout)]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))

        # layers.append(nn.Linear(hidden_channels, hidden_channels))

        self.model = nn.Sequential(*layers)

    def reset_parameters(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index=None, edge_weight=None):
        return self.model(x)
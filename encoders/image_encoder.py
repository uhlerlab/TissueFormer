import torch
import torch.nn as nn

class Image_Encoder(nn.Module):
    def __init__(self):
        super(Image_Encoder, self).__init__()

    def forward(self, x, edge_index=None, edge_weight=None):
        return x
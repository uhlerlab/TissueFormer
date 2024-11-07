import torch
import torch.nn as nn

class Image_Pretrain(nn.Module):
    def __init__(self):
        super(Image_Pretrain, self).__init__()

    def forward(self, x, edge_index=None, edge_weight=None):
        return x
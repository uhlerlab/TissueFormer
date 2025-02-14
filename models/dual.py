import math
import torch
import torch.nn as nn

class Gene_Encoder(nn.Module):
    def __init__(self, gene_embeddings, is_trainable, use_pretrained):
        super(Gene_Encoder, self).__init__()
        self.gene_embeddings = nn.Parameter(gene_embeddings, requires_grad=is_trainable) # [total genes, H]
        if not use_pretrained:
            nn.init.xavier_uniform_(self.gene_embeddings)

    def forward(self, x, gene_idx):
        gene_embs = self.gene_embeddings[gene_idx] # [G, H]
        cell_embs = torch.matmul(x, gene_embs) # [C, G] * [G, H] -> [C, H]
        # norm = torch.matmul(x, torch.ones((x.shape[1], 1), device=x.device)) + 1e-4
        # cell_embs = cell_embs / norm # [C, H] / [C, 1] -> [C, H]
        return cell_embs

class MLP_Predict(nn.Module):
    def __init__(self, encoder1, encoder2, gene_embeddings, hidden_channels, out_channels, ge_trainable=False, ge_pretrained=False):
        super(MLP_Predict, self).__init__()

        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.gene_encoder = Gene_Encoder(gene_embeddings, ge_trainable, ge_pretrained)
        layers = []
        layers += [nn.Linear(hidden_channels, hidden_channels), nn.BatchNorm1d(hidden_channels), nn.ELU()]
        layers += [nn.Linear(hidden_channels, hidden_channels), nn.BatchNorm1d(hidden_channels), nn.ELU()]
        layers += [nn.Linear(hidden_channels, out_channels)]
        self.mlp_out = nn.Sequential(*layers)

    def reset_parameters(self):
        for layer in self.mlp_out.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x1, x2, gene_idx, edge_index=None, edge_weight=None):
        z1 = self.encoder1(x1, edge_index, edge_weight)
        x2 = self.gene_encoder(x2, gene_idx)
        z2 = self.encoder2(x2, edge_index, edge_weight)
        pred = self.mlp_out(torch.cat([z1, z2], dim=-1))
        return pred
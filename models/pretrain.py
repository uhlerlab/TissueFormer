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

class Model_Pretrain(nn.Module):
    def __init__(self, encoder1, encoder2, gene_embeddings, reg_w=1.0, ge_trainable=True, ge_pretrained=False):
        super(Model_Pretrain, self).__init__()

        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.gene_encoder = Gene_Encoder(gene_embeddings, ge_trainable, ge_pretrained)
        self.reg_w = reg_w

    def reset_parameters(self):
        self.encoder1.reset_parameters()
        self.encoder2.reset_parameters()

    def forward(self, x1, x2, gene_idx, edge_index, edge_weight=None):
        z1 = self.encoder1(x1, edge_index, edge_weight)

        x2 = self.gene_encoder(x2, gene_idx)
        z2 = self.encoder2(x2, edge_index, edge_weight)
        return z1, z2

    def loss(self, x1, x2, gene_idx, train_idx=None, edge_index=None, edge_weight=None):
        z1, z2 = self.forward(x1, x2, gene_idx, edge_index, edge_weight)

        if train_idx is not None:
            z1, z2 = z1[train_idx], z2[train_idx]

        z1 = z1 / (torch.norm(z1, p=2, dim=-1, keepdim=True) + 1e-4) # [N, D]
        z2 = z2 / (torch.norm(z2, p=2, dim=-1, keepdim=True) + 1e-4)  # [N, D]

        N = z1.shape[0]
        loss_align = - torch.multiply(z1, z2).sum(-1).mean()
        loss_entropy = (torch.log(torch.multiply(torch.sum(z2, 0, keepdim=True), z1).sum(-1) + N).mean()
                    + torch.log(torch.multiply(torch.sum(z1, 0, keepdim=True), z2).sum(-1) + N).mean()) / 2.
        return loss_align + self.reg_w * loss_entropy
import math
import torch
import torch.nn as nn

class MLP_Predict(nn.Module):
    def __init__(self, encoder1, hidden_channels, out_channels):
        super(MLP_Predict, self).__init__()

        self.encoder1 = encoder1
        # layers = [nn.Linear(hidden_channels, hidden_channels), nn.BatchNorm1d(hidden_channels), nn.ELU()]
        # layers += [nn.Linear(hidden_channels, hidden_channels), nn.BatchNorm1d(hidden_channels), nn.ELU()]
        # layers += [nn.Linear(hidden_channels, out_channels)]
        layers = [nn.Linear(hidden_channels, out_channels)]
        self.mlp_out = nn.Sequential(*layers)

    def reset_parameters(self):
        for layer in self.mlp_out.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, gene_idx=None, edge_index=None, edge_weight=None):
        z = self.encoder1(x, edge_index, edge_weight)
        pred_all = self.mlp_out(z) # [C, total G]
        if gene_idx is not None:
            return pred_all[:, gene_idx] # [C, total G] -> [C, G]
        else:
            return pred_all


class InContext_Predict(nn.Module):
    def __init__(self, encoder1, hidden_channels, out_channels, num_neighbors=1000, batch_size=50, device='cpu'):
        super(InContext_Predict, self).__init__()

        self.encoder1 = encoder1
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.ref_x = torch.empty((0, hidden_channels), dtype=torch.float32).to(device)
        self.ref_y = torch.empty((0, out_channels), dtype=torch.float32).to(device)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.device = device

    def reset_parameters(self):
        self.ref_x = torch.empty((0, self.hidden_channels), dtype=torch.float32).to(self.device)
        self.ref_y = torch.empty((0, self.out_channels), dtype=torch.float32).to(self.device)

    @ torch.no_grad()
    def update(self, x, y, gene_idx=None, train_idx=None, edge_index=None, edge_weight=None):

        x = self.encoder1(x, edge_index, edge_weight)
        if train_idx is not None:
            x, y = x[train_idx], y[train_idx]
        self.ref_x = torch.cat((self.ref_x, x), 0)
        if gene_idx is not None:
            y_ = torch.zeros((y.shape[0], self.ref_y.shape[1]), dtype=torch.float32, device=x.device)
            y_[:, gene_idx] = y
            self.ref_y = torch.cat((self.ref_y, y_), 0)
        else:
            self.ref_y = torch.cat((self.ref_y, y), 0)

    @torch.no_grad()
    def forward(self, x, gene_idx=None, edge_index=None, edge_weight=None):
        x = self.encoder1(x, edge_index, edge_weight)
        x /= torch.norm(x, p=2, dim=-1, keepdim=True)
        ref_x = self.ref_x / torch.norm(self.ref_x, p=2, dim=-1, keepdim=True)
        if gene_idx is not None:
            ref_y_ = self.ref_y[:, gene_idx]
        else:
            ref_y_ = self.ref_y

        # query: x, key: ref_x, value: ref_y_
        if x.shape[0] <= self.batch_size:
            sim = torch.matmul(x, ref_x.T) # [spot num, ref num]
            knn_values, knn_indices = torch.topk(sim, k=self.num_neighbors, dim=1, largest=True)
            weight = torch.nn.functional.softmax(knn_values, dim=1)
            pred_y = torch.multiply(ref_y_[knn_indices], weight.unsqueeze(2)).sum(dim=1) # [spot num, gene num]

        else:
            pred_y = torch.empty((0, ref_y_.shape[1])).to(x.device)
            for i in range(x.shape[0] // self.batch_size + 1):
                x_i = x[i * self.batch_size: (i + 1) * self.batch_size]
                sim_i = torch.matmul(x_i, ref_x.T)  # [spot num, ref num]
                knn_values_i, knn_indices_i = torch.topk(sim_i, k=self.num_neighbors, dim=1, largest=True)
                weight_i = torch.nn.functional.softmax(knn_values_i, dim=1)
                pred_y_i = torch.multiply(ref_y_[knn_indices_i], weight_i.unsqueeze(2)).sum(dim=1) # [spot num, gene num]
                pred_y = torch.cat([pred_y, pred_y_i], dim=0)

        return pred_y
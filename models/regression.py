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

class Mean_Pooling(nn.Module):
    def __init__(self, out_channels):
        super(Mean_Pooling, self).__init__()
        self.sum_pred = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.nums = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.out_channels = out_channels

    def update(self, x, y, gene_idx=None, train_idx=None, edge_index=None, edge_weight=None):
        if train_idx is not None:
            y = y[train_idx]
        new_y = torch.zeros((self.out_channels, )).to(x.device)
        if gene_idx is not None:
            new_y[gene_idx] = y.mean(dim=0)
        else:
            new_y = y.mean(dim=0)
        self.sum_pred += new_y
        self.nums += 1

    def forward(self, x, gene_idx=None, edge_index=None, edge_weight=None):
        nonzero_idx = self.nums.nonzero()[:, 0]
        self.nums += 1
        self.nums[nonzero_idx] -= 1
        mean_pred = self.sum_pred / self.nums
        if gene_idx is not None:
            return mean_pred[gene_idx].unsqueeze(0).repeat(x.shape[0], 1)
        else:
            return mean_pred.unsqueeze(0).repeat(x.shape[0], 1)

class KNN_Predict(nn.Module):
    def __init__(self, encoder1, hidden_channels, out_channels, num_neighbors=1000, batch_size=50, device='cpu'):
        super(KNN_Predict, self).__init__()

        self.encoder1 = encoder1
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.ref_x = torch.empty((0, hidden_channels), dtype=torch.float32).to(device)
        self.ref_y = torch.empty((0, out_channels), dtype=torch.float32).to(device)

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
        # ref_x_square = torch.sum(self.ref_x ** 2, dim=1, keepdim=True)  # Shape: (n_train, 1)
        x /= torch.norm(x, p=2, dim=-1, keepdim=True)
        ref_x = self.ref_x / torch.norm(self.ref_x, p=2, dim=-1, keepdim=True)
        if gene_idx is not None:
            ref_y_ = self.ref_y[:, gene_idx]
        else:
            ref_y_ = self.ref_y

        if x.shape[0] <= self.batch_size:
            sim = torch.matmul(x, ref_x.T)  # [spot num, ref num]
            knn_values, knn_indices = torch.topk(sim, k=self.num_neighbors, dim=1, largest=True)
            pred_y = ref_y_[knn_indices].mean(dim=1)
        else:
            pred_y = torch.empty((0, ref_y_.shape[1])).to(x.device)
            for i in range(x.shape[0] // self.batch_size + 1):
                x_i = x[i * self.batch_size: (i + 1) * self.batch_size]
                sim_i = torch.matmul(x_i, ref_x.T)  # [spot num, ref num]
                knn_values_i, knn_indices_i = torch.topk(sim_i, k=self.num_neighbors, dim=1, largest=True)
                pred_y_i = ref_y_[knn_indices_i].mean(dim=1)
                pred_y = torch.cat([pred_y, pred_y_i], dim=0)

        # if x.shape[0] <= self.batch_size:
        #     x_squared = torch.sum(x ** 2, dim=1, keepdim=True)  # Shape: (n_test, 1)
        #     distances = torch.sqrt(x_squared + ref_x_square.T - 2 * torch.matmul(x, self.ref_x.T))
        #     _, knn_indices = torch.topk(distances, k=self.num_neighbors, dim=1, largest=False)
        #     pred_y = ref_y_[knn_indices].mean(dim=1)
        # else:
        #     pred_y = torch.empty((0, ref_y_.shape[1])).to(x.device)
        #     for i in range(x.shape[0] // self.batch_size + 1):
        #         x_i = x[i * self.batch_size: (i + 1) * self.batch_size]
        #         x_squared_i = torch.sum(x_i ** 2, dim=1, keepdim=True)  # Shape: (n_test, 1)
        #         distances_i = torch.sqrt(x_squared_i + ref_x_square.T - 2 * torch.matmul(x_i, self.ref_x.T))
        #         _, knn_indices_i = torch.topk(distances_i, k=self.num_neighbors, dim=1, largest=False)
        #         pred_y_i = ref_y_[knn_indices_i].mean(dim=1)
        #         pred_y = torch.cat([pred_y, pred_y_i], dim=0)

        return pred_y

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
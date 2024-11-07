import torch
import torch.nn as nn

class KNN(nn.Module):
    def __init__(self, num_neighbors, batch_size=10000):
        super(KNN, self).__init__()

        self.num_neighbors = num_neighbors
        self.batch_size = batch_size

    def forward(self, train_x, train_y, test_x):
        train_squared = torch.sum(train_x ** 2, dim=1, keepdim=True)  # Shape: (n_train, 1)

        test_pred_y = torch.empty((0, train_y.shape[1]))

        for i in range(test_x.shape[0] // self.batch_size + 1):
            test_x_i = test_x[i * self.batch_size: (i + 1) * self.batch_size]
            test_squared_i = torch.sum(test_x_i ** 2, dim=1, keepdim=True)  # Shape: (n_test, 1)
            tr_te_distances = torch.sqrt(test_squared_i + train_squared.T - 2 * torch.matmul(test_x_i, train_x.T))
            _, test_knn_indices = torch.topk(tr_te_distances, k=self.num_neighbor, dim=1, largest=False)
            test_pred_y_i = train_y[test_knn_indices].mean(dim=1)
            test_pred_y = torch.cat([test_pred_y, test_pred_y_i], dim=0)

        return test_pred_y
from sklearn.metrics import mean_squared_error
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau, spearmanr
import torch

def mean_square_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def kendall_rank(y_true, y_pred):
    tau, p_value = kendalltau(y_true, y_pred)
    return tau

def spearman_score(y_true, y_pred):
    return spearmanr(y_true, y_pred)

def ndcg_k(y_true, y_pred, k):
    return ndcg_score(y_true, y_pred, k=k)

def calculate_metrics(y_true, y_pred, row_subset_id=None, column_subset_id=None):
    if column_subset_id is not None:
        y_true = y_true[:, column_subset_id]
        y_pred = y_pred[:, column_subset_id]
    if row_subset_id is not None:
        y_true = y_true[row_subset_id, :]
        y_pred = y_pred[row_subset_id, :]
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    mse = mean_square_error(y_true, y_pred)
    kendall = kendall_rank(y_true, y_pred)
    ndcg = ndcg_k(y_true, y_pred, k=20)
    ndcg_1 = ndcg_k(y_true, y_pred, k=1)
    return mse, kendall, ndcg, ndcg_1

def calculate_mse(y_true, y_pred, row_subset_id=None, column_subset_id=None):
    if column_subset_id is not None:
        y_true = y_true[:, column_subset_id]
        y_pred = y_pred[:, column_subset_id]
    if row_subset_id is not None:
        y_true = y_true[row_subset_id, :]
        y_pred = y_pred[row_subset_id, :]
    return torch.mean((y_pred - y_true) ** 2)
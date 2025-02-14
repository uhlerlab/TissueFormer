from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.metrics import ndcg_score
from sklearn.metrics import accuracy_score, roc_auc_score, top_k_accuracy_score, f1_score
from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata
from sklearn.preprocessing import label_binarize

import torch
import numpy as np

def mean_square_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def root_mean_square_error(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

def kendall_rank(y_true, y_pred):
    res = []
    for t, p in zip(y_true, y_pred):
        tau, p_value = kendalltau(t, p)
        res.append(tau)
    return np.mean(res)

def pearson_score(y_true, y_pred, rank_axis='by_gene'):

    axis = 1 if rank_axis == 'by_gene' else 0

    matrix1 = y_true - np.mean(y_true, axis=axis, keepdims=True)
    matrix2 = y_pred - np.mean(y_pred, axis=axis, keepdims=True)

    covariance = np.sum(matrix1 * matrix2, axis=axis, keepdims=True)

    std1 = np.sqrt(np.sum(matrix1 ** 2, axis=axis, keepdims=True))
    std2 = np.sqrt(np.sum(matrix2 ** 2, axis=axis, keepdims=True))

    std_prod = std1 * std2

    if rank_axis == 'by_gene':
        nonzero = std_prod.nonzero()[0]
        corrs = covariance[nonzero] / std_prod[nonzero]
    else:
        nonzero = std_prod.nonzero()[1]
        corrs = covariance[:, nonzero] / std_prod[:, nonzero]

    return np.mean(corrs)


def spearman_score(y_true, y_pred, rank_axis='by_gene'):

    axis = 1 if rank_axis == 'by_gene' else 0

    ranked_matrix1 = np.apply_along_axis(rankdata, axis=axis, arr=y_true)
    ranked_matrix2 = np.apply_along_axis(rankdata, axis=axis, arr=y_pred)

    ranked_matrix1 -= np.mean(ranked_matrix1, axis=axis, keepdims=True)
    ranked_matrix2 -= np.mean(ranked_matrix2, axis=axis, keepdims=True)

    covariance = np.sum(ranked_matrix1 * ranked_matrix2, axis=axis, keepdims=True)

    std1 = np.sqrt(np.sum(ranked_matrix1 ** 2, axis=axis, keepdims=True))
    std2 = np.sqrt(np.sum(ranked_matrix2 ** 2, axis=axis, keepdims=True))

    std_prod = std1 * std2

    if rank_axis == 'by_gene':
        nonzero = std_prod.nonzero()[0]
        corrs = covariance[nonzero] / std_prod[nonzero]
    else:
        nonzero = std_prod.nonzero()[1]
        corrs = covariance[:, nonzero] / std_prod[:, nonzero]

    return np.mean(corrs)

def ndcg_k(y_true, y_pred, k):
    return ndcg_score(y_true, y_pred, k=k)

def accuracy(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred)

def accuracy_topk(y_true, y_pred, class_num):
    return top_k_accuracy_score(y_true, y_pred, k=2, labels=[i for i in range(class_num)])

def f1(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average='macro')

def roc_auc(y_true, y_pred):
    # nonzero_mask = y_true.sum(0).nonzero()[0]
    # y_true, y_pred = y_true[:, nonzero_mask], y_pred[:, nonzero_mask]
    if y_true.sum() > 0:
        return roc_auc_score(y_true, y_pred, average='macro')
    else:
        return 1.0

def calculate_metrics(y_true, y_pred, metrics=['RMSE', 'Pearson', 'Spearman'], row_subset_id=None, column_subset_id=None, class_num=None):
    if column_subset_id is not None:
        y_true = y_true[:, column_subset_id]
        y_pred = y_pred[:, column_subset_id]
    if row_subset_id is not None:
        y_true = y_true[row_subset_id]
        y_pred = y_pred[row_subset_id]
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()

    output_metrics = []

    if 'MSE' in metrics:
        mse = torch.mean((y_true - y_pred) ** 2)
        output_metrics.append(mse)
    if 'RMSE' in metrics:
        rmse = root_mean_square_error(y_true, y_pred)
        output_metrics.append(rmse)
    if 'Pearson' in metrics:
        pearson = pearson_score(y_true, y_pred, rank_axis='by_gene')
        output_metrics.append(pearson)
    if 'Spearman' in metrics:
        spearman = spearman_score(y_true, y_pred, rank_axis='by_gene')
        output_metrics.append(spearman)
    if 'NDCG@5' in metrics:
        ndcg5 = ndcg_k(y_true, y_pred, k=5)
        output_metrics.append(ndcg5)
    if 'NDCG@20' in metrics:
        ndcg20 = ndcg_k(y_true, y_pred, k=20)
        output_metrics.append(ndcg20)
    if 'Accuracy' in metrics:
        acc = accuracy(y_true, y_pred)
        output_metrics.append(acc)
    if 'Accuracy_TopK' in metrics:
        acc = accuracy_topk(y_true, y_pred, class_num)
        output_metrics.append(acc)
    if 'AUC' in metrics:
        auc = roc_auc(y_true, y_pred)
        output_metrics.append(auc)
    if 'f1' in metrics:
        f = f1(y_true, y_pred)
        output_metrics.append(f)

    return output_metrics

def calculate_score(y_true, y_pred, metric='RMSE', row_subset_id=None, column_subset_id=None, class_num=None):
    if column_subset_id is not None:
        y_true = y_true[:, column_subset_id]
        y_pred = y_pred[:, column_subset_id]
    if row_subset_id is not None:
        y_true = y_true[row_subset_id]
        y_pred = y_pred[row_subset_id]

    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()

    if metric == 'MSE':
        return torch.mean((y_true - y_pred) ** 2)
    elif metric == 'RMSE':
        return root_mean_square_error(y_true, y_pred)
    elif metric == 'Pearson':
        return pearson_score(y_true, y_pred)
    elif metric == 'Spearman':
        return spearman_score(y_true, y_pred)
    elif metric == 'Accuracy':
        return accuracy(y_true, y_pred)
    elif metric == 'Accuracy_TopK':
        return accuracy_topk(y_true, y_pred, class_num)
    elif metric == 'AUC':
        return roc_auc(y_true, y_pred)
    elif metric == 'f1':
        return f1(y_true, y_pred)
    else:
        NotImplementedError
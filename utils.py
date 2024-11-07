import os
import numpy as np
import torch
from torch_cluster import knn_graph
from dataloader import CustomDataset
import scanpy as sc
import scipy.sparse as sp

def k_fold_split(samples):
    train_idx_list, test_idx_list = [], []
    n = len(samples)
    for i in range(n):
        train_idx_list.append([samples[i]])
        if i == 0:
            test_idx_list.append(samples[1:])
        elif i == n - 1:
            test_idx_list.append(samples[:-1])
        else:
            test_idx_list.append(samples[:i] + samples[i+1:])
    return train_idx_list, test_idx_list

def random_splits(idx, valid_prop, test_prop=None, labeled_idx=None):
    """ randomly splits label into train/valid/test splits """

    n = idx.shape[0] if labeled_idx is None else labeled_idx.shape[0]
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    valid_idx = perm[:valid_num]
    if test_prop is None:
        train_idx = perm[valid_num:]
        if labeled_idx is None:
            return train_idx, valid_idx
        else:
            return labeled_idx[train_idx], labeled_idx[valid_idx]
    else:
        test_num = int(n * test_prop)
        test_idx = perm[valid_num:valid_num + test_num]
        train_idx = perm[valid_num + test_num:]
        if labeled_idx is None:
            return train_idx, valid_idx, test_idx
        else:
            return labeled_idx[train_idx], labeled_idx[valid_idx], labeled_idx[test_idx]

def data_load(dir_path, sample):
    dir_path = os.path.join(dir_path, sample) + '.h5ad'
    adata = sc.read(dir_path)
    X_log1p = adata.layers['X_log1p']
    gene_mask = adata.var['gene_mask_scgpt_human']
    cell_by_gene = X_log1p[:, gene_mask]
    hvg_gene_mask = adata.var['highly_variable'][gene_mask]
    gene_index = adata.var['gene_filtered_idx'][gene_mask]
    cell_image_emb = adata.obsm['embeddings']
    cell_location = adata.obsm['centroids']

    dataset = {}
    dataset['x'] = torch.tensor(cell_image_emb, dtype=torch.float)
    if sp.issparse(cell_by_gene):
        rows, cols = cell_by_gene.nonzero()
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(cell_by_gene.data, dtype=torch.float32)
        dataset['y'] = torch.sparse_coo_tensor(indices, values, torch.Size(cell_by_gene.shape))
    else:
        dataset['y'] = torch.tensor(cell_by_gene, dtype=torch.float)
    dataset['gene_idx'] = torch.tensor(gene_index, dtype=torch.long)
    dataset['hvg_gene_mask'] = torch.tensor(hvg_gene_mask, dtype=torch.bool)
    dataset['edge_index'] = knn_graph(torch.tensor(cell_location, dtype=torch.float), k=5, loop=False)

    return dataset

def dataset_create_pretrain(dir_path, samples):
    datasets = []
    for s in samples:
        dataset = data_load(dir_path, s)
        datasets.append(dataset)
    return CustomDataset(datasets)

def dataset_create_supervise(dir_path, train_samples, test_samples=None, valid_prop=0.1, test_prop=0.8):
    train_datasets, valid_datasets, test_datasets = [], [], []
    if test_samples is not None:
        for s in train_samples:
            dataset = data_load(dir_path, s)
            idx = torch.arange(dataset['x'].shape[0])
            train_idx, valid_idx = random_splits(idx, valid_prop)
            dataset['split_idx'] = {'train_idx': train_idx, 'valid_idx': valid_idx, 'test_idx': None}
            train_datasets.append(dataset)
            valid_datasets.append(dataset)
        for s in test_samples:
            dataset = data_load(dir_path, s)
            idx = torch.arange(dataset['x'].shape[0])
            dataset['split_idx'] = {'train_idx': None, 'valid_idx': None, 'test_idx': idx}
            test_datasets.append(dataset)
    else:
        for s in train_samples:
            dataset = data_load(dir_path, s)
            idx = torch.arange(dataset['x'].shape[0])
            train_idx, valid_idx, test_idx = random_splits(idx, valid_prop, test_prop)
            dataset['split_idx'] = {'train_idx': train_idx, 'valid_idx': valid_idx, 'test_idx': test_idx}
            train_datasets.append(dataset)
            valid_datasets.append(dataset)
            test_datasets.append(dataset)
    return CustomDataset(train_datasets), CustomDataset(valid_datasets), CustomDataset(test_datasets)

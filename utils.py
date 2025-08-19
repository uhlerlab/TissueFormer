import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_cluster import knn_graph
from dataloader import CustomDataset
import scanpy as sc
import scipy.sparse as sp
from tqdm import tqdm

def k_shot_split(samples, num_splits, k=1):
    train_idx_list, test_idx_list = [], []
    n = len(samples)
    for i in range(num_splits):
        train_idx = torch.randint(0, n, (k, )).tolist()
        train_idx_i, test_idx_i = [], []
        for j in range(n):
            if j in train_idx:
                train_idx_i += [samples[j]]
            else:
                test_idx_i += [samples[j]]
        train_idx_list.append(train_idx_i)
        test_idx_list.append(test_idx_i)
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

def spatial_splits(locations, idx, valid_prop, test_prop=None):
    x, y = locations[:, 0], locations[:, 1]
    n = idx.shape[0]
    valid_num = int(n * valid_prop)
    test_num = int(n * test_prop)

    idx = torch.as_tensor(x.argsort())
    valid_idx, test_idx = idx[:valid_num], idx[valid_num:valid_num + test_num]
    train_idx = idx[valid_num + test_num:]
    return train_idx, valid_idx, test_idx


def data_load_hest(dir_path, sample, args, filter_genes=True):
    dir_path = os.path.join(dir_path, sample) + '.h5ad'
    adata = sc.read(dir_path)
    X_log1p = adata.layers['X_log1p']

    if filter_genes:
        gene_mask = adata.var['gene_filter_mask']
        cell_by_gene = X_log1p[:, gene_mask]
    else:
        gene_mask = np.ones_like(adata.var['gene_filter_mask'], dtype=bool)
        cell_by_gene = np.log1p(adata.X / (adata.X.sum(-1, keepdims=True) + 1e-5 ) * 1e4)
    gene_index = adata.var['gene_filtered_idx'][gene_mask]

    if args.image_model == 'hoptimus':
        cell_image_emb = adata.obsm['embeddings']
    elif args.image_model == 'gigapath':
        cell_image_emb = adata.obsm['patch_embs_gigapath']
    elif args.image_model == 'uni':
        cell_image_emb = adata.obsm['patch_embs_uni']
    elif args.image_model == 'pca':
        cell_image_emb = adata.obsm['patch_pca_100']
    else:
        raise NotImplementedError
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
    dataset['edge_index'] = knn_graph(torch.tensor(cell_location, dtype=torch.float), k=5, loop=False)
    dataset['cell_location'] = cell_location

    if 'gene_intersection_mask' in adata.var:
        gene_intersection_mask = adata.var['gene_intersection_mask']
        hvg_gene_rank = adata.var['highly_variable_rank'][gene_intersection_mask]
        gene_eval_mask = gene_intersection_mask[gene_mask]  # indicate which gene idx should be evaluated
        dataset['gene_eval_mask'] = torch.tensor(gene_eval_mask, dtype=torch.bool)
        dataset['hvg_gene_rank'] = torch.tensor(hvg_gene_rank, dtype=torch.long)
    else:
        hvg_gene_rank = adata.var['highly_variable_rank'][gene_mask]
        dataset['hvg_gene_rank'] = torch.tensor(hvg_gene_rank, dtype=torch.long)
    return dataset

def data_load_lung(dir_path, sample, args, use_pred_gene=False, split_with_region=False):
    cell_type = ['RASC', 'Secretory', 'Multiciliated', 'PNEC', 'Basal', 'Goblet', 'Proliferating Airway', 'AT2',
                 'Transitional AT2', 'AT1',
                 'KRT5-/KRT17+', 'Proliferating AT2', 'Langerhans cells', 'NK/NKT', 'Tregs', 'CD4+ T-cells',
                 'CD8+ T-cells',
                 'Proliferating T-cells', 'B cells', 'Plasma', 'pDCs', 'Proliferating NK/NKT', 'Proliferating B cells',
                 'cDCs', 'Mast',
                 'Interstitial Macrophages', 'Alveolar Macrophages', 'SPP1+ Macrophages', 'Neutrophils',
                 'Proliferating Myeloid',
                 'Migratory DCs', 'Macrophages - IFN-activated', 'Monocytes/MDMs', 'Basophils', 'Venous', 'Capillary',
                 'Lymphatic',
                 'Arteriole', 'SMCs/Pericytes', 'Alveolar FBs', 'Proliferating FBs', 'Inflammatory FBs',
                 'Activated Fibrotic FBs',
                 'Myofibroblasts', 'Subpleural FBs', 'Adventitial FBs', 'Mesothelial']
    ct_map = {c: i for i, c in enumerate(cell_type)}
    ct_map_inv = {i: c for i, c in enumerate(cell_type)}

    dir_path = os.path.join(dir_path, sample) + '.h5ad'
    adata = sc.read(dir_path)
    X_log1p = adata.layers['X_log1p']

    gene_mask = adata.var['gene_filter_mask']
    cell_by_gene = X_log1p[:, gene_mask]
    gene_index = adata.var['gene_filtered_idx'][gene_mask]

    cell_type_label = adata.obs['final_CT']

    cell_niche_T = adata.obs['TNiche']
    cell_niche_C = adata.obs['CNiche']

    if args.image_model == 'hoptimus':
        cell_image_emb = adata.obsm['embeddings']
    elif args.image_model == 'gigapath':
        cell_image_emb = adata.obsm['patch_embs_gigapath']
    elif args.image_model == 'uni':
        cell_image_emb = adata.obsm['patch_embs_uni']
    elif args.image_model == 'pca':
        cell_image_emb = adata.obsm['patch_pca_100']
    else:
        raise NotImplementedError
    cell_location = adata.obsm['centroids']

    dataset = {}
    dataset['x'] = torch.tensor(cell_image_emb, dtype=torch.float)
    if args.evaluate_task == 'gene_regression':
        dataset['y'] = torch.tensor(cell_by_gene, dtype=torch.float)
    elif args.evaluate_task == 'niche_classification':
        if use_pred_gene:
            file_path = '/data/wuqitian/analysis_pred_data/gene_expression_prediction/' + f'{sample}_ours_in.npy'
            y_pred = np.load(file_path)
            dataset['x2'] = torch.tensor(y_pred, dtype=torch.float)
        else:
            dataset['x2'] = torch.tensor(cell_by_gene, dtype=torch.float)

        if args.niche_type[0] == 'T':
            dataset['y'] = torch.tensor((cell_niche_T == args.niche_type), dtype=torch.long)
        else:
            dataset['y'] = torch.tensor((cell_niche_C == args.niche_type), dtype=torch.long)
        idx = torch.arange(0, dataset['y'].shape[0])
        neg_idx = idx[dataset['y'] == 0]
        pos_idx = idx[dataset['y'] == 1]
        sample_neg_idx_ = torch.as_tensor(np.random.permutation(neg_idx.shape[0]))[:5*pos_idx.shape[0]]
        sample_neg_idx = neg_idx[sample_neg_idx_]
        mask = torch.zeros_like(dataset['y'], dtype=torch.bool)
        mask[sample_neg_idx] = True
        mask[pos_idx] = True
        dataset['cell_mask'] = mask
    elif args.evaluate_task == 'cell_type_classification':
        if use_pred_gene:
            file_path = '/data/wuqitian/analysis_pred_data/gene_expression_prediction/' + f'{sample}_ours_region.npy'
            y_pred = np.load(file_path)
            dataset['x2'] = torch.tensor(y_pred, dtype=torch.float)
        else:
            dataset['x2'] = torch.tensor(cell_by_gene, dtype=torch.float)

        cell_type = ct_map[args.cell_type.replace('_', ' ')]
        dataset['y'] = torch.tensor((cell_type_label == cell_type), dtype=torch.long)
        idx = torch.arange(0, dataset['y'].shape[0])
        neg_idx = idx[dataset['y'] == 0]
        pos_idx = idx[dataset['y'] == 1]
        sample_neg_idx_ = torch.as_tensor(np.random.permutation(neg_idx.shape[0]))[:5 * pos_idx.shape[0]]
        sample_neg_idx = neg_idx[sample_neg_idx_]
        mask = torch.zeros_like(dataset['y'], dtype=torch.bool)
        mask[sample_neg_idx] = True
        mask[pos_idx] = True
        dataset['cell_mask'] = mask
    elif args.evaluate_task == 'region_time_prediction':
        if use_pred_gene:
            file_path = '/data/wuqitian/analysis_pred_data/gene_expression_prediction/' + f'{sample}_ours_region.npy'
            y_pred = np.load(file_path)
            dataset['x2'] = torch.tensor(y_pred, dtype=torch.float)
        else:
            dataset['x2'] = torch.tensor(cell_by_gene, dtype=torch.float)

        lumen_rank = adata.obs['lumen_rank'].to_numpy()
        lumen_rank = np.nan_to_num(lumen_rank, nan=-1)
        lumen_rank_unique = np.sort(np.unique(lumen_rank))
        lumen_rank_map = {k: i for i, k in enumerate(lumen_rank_unique)}
        vectorized_lookup = np.vectorize(lumen_rank_map.get)
        group_idx = vectorized_lookup(lumen_rank, None)
        dataset['group_idx'] = torch.tensor(group_idx, dtype=torch.long)

        dataset['y'] = torch.tensor(lumen_rank_unique, dtype=torch.float).reshape(-1, 1) / 100

    elif args.evaluate_task == 'he_annotation_classification':
        he_annotation = adata.obsm['he_annotation']
        dataset['y'] = torch.tensor(he_annotation, dtype=torch.long)[:, args.he_annotation_idx]
        idx = torch.arange(0, dataset['y'].shape[0])
        neg_idx = idx[dataset['y'] == 0]
        pos_idx = idx[dataset['y'] == 1]
        sample_neg_idx_ = torch.as_tensor(np.random.permutation(neg_idx.shape[0]))[:5*pos_idx.shape[0]]
        sample_neg_idx = neg_idx[sample_neg_idx_]
        mask = torch.zeros_like(dataset['y'], dtype=torch.bool)
        mask[sample_neg_idx] = True
        mask[pos_idx] = True
        dataset['cell_mask'] = mask
    else:
        raise NotImplementedError
    dataset['gene_idx'] = torch.tensor(gene_index, dtype=torch.long)
    dataset['edge_index'] = knn_graph(torch.tensor(cell_location, dtype=torch.float), k=5, loop=False)

    hvg_gene_rank = adata.var['highly_variable_rank'][gene_mask]
    dataset['hvg_gene_rank'] = torch.tensor(hvg_gene_rank, dtype=torch.long)

    if split_with_region:
        idx = torch.arange(0, dataset['y'].shape[0])
        lumen_rank = adata.obs['lumen_rank'].to_numpy()
        lumen_mask = torch.tensor(~np.isnan(lumen_rank), dtype=torch.bool)
        train_idx, test_idx = idx[~lumen_mask], idx[lumen_mask]
        dataset['split_idx'] = {'train_idx': train_idx, 'valid_idx': None, 'test_idx': test_idx}
    return dataset

def data_load_kidney(dir_path, sample, args):
    dir_path = os.path.join(dir_path, sample) + '.h5ad'
    adata = sc.read(dir_path)
    X_log1p = adata.layers['X_log1p']

    gene_mask = adata.var['gene_filter_mask']
    cell_by_gene = X_log1p[:, gene_mask]
    gene_index = adata.var['gene_filtered_idx'][gene_mask]

    microphage_type = adata.obs['macrophage_label']
    microphage_confidence = adata.obs['confidence_level']
    microphage_mask = adata.obs['macrophage_mask']

    cell_image_emb = adata.obsm['embeddings']
    cell_location = adata.obsm['centroids']

    dataset = {}
    dataset['x'] = torch.tensor(cell_image_emb, dtype=torch.float)
    if args.evaluate_task == 'gene_regression':
        dataset['y'] = torch.tensor(cell_by_gene, dtype=torch.float)
    elif args.evaluate_task == 'macrophage_identification':
        dataset['y'] = torch.tensor(microphage_mask, dtype=torch.long)
    elif args.evaluate_task == 'macrophage_classification':
        dataset['y'] = torch.tensor(microphage_type, dtype=torch.long)
    else:
        raise NotImplementedError
    dataset['microphage_mask'] = torch.tensor(microphage_mask, dtype=torch.bool)
    dataset['edge_index'] = knn_graph(torch.tensor(cell_location, dtype=torch.float), k=5, loop=False)

    hvg_gene_rank = adata.var['highly_variable_rank'][gene_mask]
    dataset['hvg_gene_rank'] = torch.tensor(hvg_gene_rank, dtype=torch.long)
    return dataset

def dataset_create(dir_path, samples, args, data_loader='hest', filter_genes=True, use_pred_gene=False):
    datasets = []
    if isinstance(samples, list):
        pbar = tqdm(samples, desc='loading dataset', ncols=100, ascii=True)
        for s in pbar:
            if data_loader == 'hest':
                dataset = data_load_hest(dir_path, s, args, filter_genes)
            elif data_loader == 'lung':
                dataset = data_load_lung(dir_path, s, args, use_pred_gene)
            elif data_loader == 'kidney':
                dataset = data_load_kidney(dir_path, s, args)
            idx = torch.arange(dataset['x'].shape[0])
            dataset['split_idx'] = {'train_idx': idx, 'valid_idx': idx, 'test_idx': idx}
            datasets.append(dataset)
            pbar.clear()
            pbar.refresh()
    else:
        if data_loader == 'hest':
            dataset = data_load_hest(dir_path, samples, args, filter_genes)
        elif data_loader == 'lung':
            dataset = data_load_lung(dir_path, samples, args, use_pred_gene)
        elif data_loader == 'kidney':
            dataset = data_load_kidney(dir_path, samples, args)
        idx = torch.arange(dataset['x'].shape[0])
        dataset['split_idx'] = {'train_idx': idx, 'valid_idx': idx, 'test_idx': idx}
        datasets.append(dataset)
    return CustomDataset(datasets)

def dataset_create_split(dir_path, samples, args, valid_prop=0.1, test_prop=0.8, split='random', data_loader='hest', filter_genes=True, split_with_region=False):
    datasets = []
    if isinstance(samples, list):
        pbar = tqdm(samples, desc='loading dataset', ncols=100, ascii=True)
        for s in pbar:
            if data_loader == 'hest':
                dataset = data_load_hest(dir_path, s, args, filter_genes)
            elif data_loader == 'lung':
                dataset = data_load_lung(dir_path, s, args, split_with_region=split_with_region)
            elif data_loader == 'kidney':
                dataset = data_load_kidney(dir_path, s, args)
            idx = torch.arange(dataset['x'].shape[0])
            if split == 'random':
                train_idx, valid_idx, test_idx = random_splits(idx, valid_prop, test_prop)
                dataset['split_idx'] = {'train_idx': train_idx, 'valid_idx': valid_idx, 'test_idx': test_idx}
            elif split == 'spatial':
                train_idx, valid_idx, test_idx = spatial_splits(dataset['cell_location'], idx, valid_prop, test_prop)
                dataset['split_idx'] = {'train_idx': train_idx, 'valid_idx': valid_idx, 'test_idx': test_idx}
            elif split == 'region':
                pass
            datasets.append(dataset)
            pbar.clear()
            pbar.refresh()
    else:
        if data_loader == 'hest':
            dataset = data_load_hest(dir_path, samples, args, filter_genes)
        elif data_loader == 'lung':
            dataset = data_load_lung(dir_path, samples, args, split_with_region=split_with_region)
        elif data_loader == 'kidney':
            dataset = data_load_kidney(dir_path, samples, args)
        idx = torch.arange(dataset['x'].shape[0])
        if split == 'random':
            train_idx, valid_idx, test_idx = random_splits(idx, valid_prop, test_prop)
            dataset['split_idx'] = {'train_idx': train_idx, 'valid_idx': valid_idx, 'test_idx': test_idx}
        elif split == 'spatial':
            train_idx, valid_idx, test_idx = spatial_splits(dataset['cell_location'], idx, valid_prop, test_prop)
            dataset['split_idx'] = {'train_idx': train_idx, 'valid_idx': valid_idx, 'test_idx': test_idx}
        elif split == 'region':
            pass
        datasets.append(dataset)
    return CustomDataset(datasets)


class FocalLoss(nn.Module):
    """
    binary focal loss
    """
    def __init__(self, alpha=0.25, gamma=2, device='cpu'):
        super(FocalLoss, self).__init__()
        self.weight = torch.Tensor([alpha, 1-alpha]).to(device)
        self.nllLoss = nn.NLLLoss(weight=self.weight)
        self.gamma = gamma

    def forward(self, input, target):
        softmax = F.softmax(input, dim=1)
        log_logits = torch.log(softmax)
        fix_weights = (1 - softmax) ** self.gamma
        logits = fix_weights * log_logits
        return self.nllLoss(logits, target)
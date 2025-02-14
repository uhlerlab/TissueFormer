import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import random
import argparse

from parse import parse_pretrain_method, parse_regression_method, parser_add_main_args
from run_finetune import run_update, run_train, run_test, evaluate
from utils import dataset_create, dataset_create_split, k_shot_split
import os
import warnings

from anndata import AnnData
from torch_cluster import knn_graph
import scanpy as sc
import scipy.sparse as sp
import anndata as ad

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='Pretraining Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()

fix_seed(args.seed)

dir_path = '/data/wuqitian/hest_data_xenium_protein_preprocess'
meta_info = pd.read_csv("../data/meta_info_xenium.csv")

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

args.gene_total_num = 372

# image embedding dim of foundation encoders
if args.image_model == 'hoptimus':
    args.image_emb_dim = 1536
elif args.image_model == 'gigapath':
    args.image_emb_dim = 1536
elif args.image_model == 'uni':
    args.image_emb_dim = 1024
elif args.image_model == 'pca':
    args.image_emb_dim = 100

samples = ['TENX126', 'TENX122', 'TENX121', 'TENX119', 'TENX118']
organs = ['Pancreas', 'Skin', 'Liver', 'Heart', 'Lung']
result_path = '/data/wuqitian/analysis_pred_data'

# load evaluate model
model_ours = parse_regression_method(args, device)
pretrained_state_dict = torch.load('../model_checkpoints/ours_pretrain_xenium_sample.pth')
encoder1_pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k.startswith("encoder1.")}
model_state_dict = model_ours.state_dict()
encoder1_model_dict = {k: v for k, v in model_state_dict.items() if k.startswith("encoder1.")}
for k, v in encoder1_pretrained_dict.items():
    assert (k in encoder1_model_dict)
    assert (v.size() == encoder1_model_dict[k].size())
model_state_dict.update(encoder1_pretrained_dict)
model_ours.load_state_dict(model_state_dict)

# our model pretrained on visium
model_ours_visium = parse_regression_method(args, device)
pretrained_state_dict = torch.load('../model_checkpoints/ours_pretrain_visium_all.pth')
encoder1_pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k.startswith("encoder1.")}
model_state_dict = model_ours_visium.state_dict()
encoder1_model_dict = {k: v for k, v in model_state_dict.items() if k.startswith("encoder1.")}
for k, v in encoder1_pretrained_dict.items():
    assert (k in encoder1_model_dict)
    assert (v.size() == encoder1_model_dict[k].size())
model_state_dict.update(encoder1_pretrained_dict)
model_ours_visium.load_state_dict(model_state_dict)

# hoptimus model
args.method = 'hoptimus-MLP'
args.image_emb_dim = 1536
model_hoptimus = parse_regression_method(args, device)
model_state_dict = torch.load('../model_checkpoints/hoptimus-MLP_evaluate_xenium_sample.pth')
model_hoptimus.load_state_dict(model_state_dict)

args.image_model = 'gigapath'
args.method = 'gigapath-MLP'
args.image_emb_dim = 1536
model_gigapath = parse_regression_method(args, device)
model_state_dict = torch.load('../model_checkpoints/gigapath-MLP_evaluate_xenium_sample.pth')
model_gigapath.load_state_dict(model_state_dict)

args.image_model = 'uni'
args.method = 'uni-MLP'
args.image_emb_dim = 1024
model_uni = parse_regression_method(args, device)
model_state_dict = torch.load('../model_checkpoints/uni-MLP_evaluate_xenium_sample.pth')
model_uni.load_state_dict(model_state_dict)

args.image_model = 'pca'
args.method = 'pca-MLP'
args.image_emb_dim = 100
model_pca = parse_regression_method(args, device)
model_state_dict = torch.load('../model_checkpoints/pca-MLP_evaluate_xenium_sample.pth')
model_pca.load_state_dict(model_state_dict)

args.image_model = 'hoptimus'
datasets = dataset_create_split(dir_path, samples, args, valid_prop=0., test_prop=0.5)
dataloader = DataLoader(datasets, batch_size=1, shuffle=True)

run_update(model_ours, dataloader, device, use_gene_idx=False)
run_update(model_ours_visium, dataloader, device, use_gene_idx=False)

param_K_keys = ['convs.0.Wk.weight', 'convs.0.Wk.bias', 'convs.1.Wk.weight', 'convs.1.Wk.bias']
param_Q_keys = ['convs.0.Wq.weight', 'convs.0.Wq.bias', 'convs.1.Wq.weight', 'convs.1.Wq.bias']

for i, sample in enumerate(samples):
    file_path = os.path.join(dir_path, sample) + '.h5ad'
    adata = sc.read(file_path)
    X_log1p = adata.layers['X_log1p']
    gene_mask = adata.var['gene_filter_mask']
    cell_by_gene = X_log1p[:, gene_mask]
    gene_index = adata.var['gene_filtered_idx'][gene_mask]
    gene_names = adata.var['gene_names'][gene_mask]
    hvg_gene_rank = adata.var['highly_variable_rank'][gene_mask]
    cell_image_emb = adata.obsm['embeddings']
    cell_location = adata.obsm['centroids']

    inputs = torch.tensor(cell_image_emb, dtype=torch.float).to(device)
    y = torch.tensor(cell_by_gene, dtype=torch.float).to(device)
    edge_index = knn_graph(torch.tensor(cell_location, dtype=torch.float), k=5, loop=False).to(device)

    model_ours.eval()
    with torch.no_grad():
        embs_ours = model_ours.encoder1.get_embeddings(inputs, edge_index) # [layer num, spot num, hidden size]
        pred_ours = model_ours(inputs, gene_idx=None, edge_index=edge_index) # [spot num, gene num] gene specific to sample
        embs_ours, pred_ours = embs_ours.cpu().numpy(), pred_ours.cpu().numpy()

        pred_ours_visium = model_ours_visium(inputs, gene_idx=None, edge_index=edge_index).cpu().numpy()
        pred_hoptimus = model_hoptimus(inputs, gene_idx=None, edge_index=edge_index).cpu().numpy()

        cell_embs_gigapath = adata.obsm['patch_embs_gigapath']
        inputs = torch.tensor(cell_embs_gigapath, dtype=torch.float).to(device)
        pred_gigapath = model_gigapath(inputs, gene_idx=None, edge_index=edge_index).cpu().numpy()

        cell_embs_uni = adata.obsm['patch_embs_uni']
        inputs = torch.tensor(cell_embs_uni, dtype=torch.float).to(device)
        pred_uni = model_uni(inputs, gene_idx=None, edge_index=edge_index).cpu().numpy()

        cell_embs_pca = adata.obsm['patch_pca_100']
        inputs = torch.tensor(cell_embs_pca, dtype=torch.float).to(device)
        pred_pca = model_pca(inputs, gene_idx=None, edge_index=edge_index).cpu().numpy()

    adata_new = adata[:, gene_mask].copy()
    adata_new.layers['X_pred_ours'] = pred_ours
    adata_new.layers['X_pred_ours_visium'] = pred_ours_visium
    adata_new.layers['X_pred_hoptimus'] = pred_hoptimus
    adata_new.layers['X_pred_gigapath'] = pred_gigapath
    adata_new.layers['X_pred_uni'] = pred_uni
    adata_new.layers['X_pred_pca'] = pred_pca

    for i in range(embs_ours.shape[0]):
        adata_new.obsm[f'embs_ours_layer{i}'] = embs_ours[i]
    # adata_new.uns['attn_ours'] = attn_ours

    print(i, sample)
    adata_new.write_h5ad(os.path.join(result_path, sample) + '_pred.h5ad')

adata_list = []
for i, s in enumerate(samples):
    file_path = os.path.join(dir_path, s) + '.h5ad'
    adata_i = sc.read(file_path)
    gene_mask = adata_i.var['gene_filter_mask']
    adata_i = adata_i[:, gene_mask]
    adata_i.var_names = adata_i.var['gene_names']
    adata_list += [adata_i]

adata_merged = ad.concat(adata_list, join='outer', label='organ', keys=organs)
# sc.pp.filter_cells(adata_merged, min_genes=5, inplace=False)
# sc.external.pp.scanorama_integrate(adata_merged, 'organ')
sc.pp.pca(adata_merged, n_comps=50)
sc.pp.neighbors(adata_merged, n_neighbors=15)
sc.tl.leiden(adata_merged, resolution=0.2, flavor="igraph", n_iterations=2, directed=False)
cluster_num = adata_merged.obs['leiden'].unique().shape[0]
print(f"Leiden cluster number: {cluster_num}")
sc.tl.umap(adata_merged)
adata_merged.write(os.path.join(result_path, 'sample_merge.h5ad'))



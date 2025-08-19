import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import random
import argparse

from run_finetune import run_update, run_train, run_test, evaluate
from parse import parse_pretrain_method, parse_regression_method, parse_classification_method, parser_add_main_args
from utils import dataset_create, dataset_create_split, k_shot_split
import os
import warnings

from torch_cluster import knn_graph
import scanpy as sc
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
from scipy.spatial.distance import pdist

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

dir_path = '/data/wuqitian/lung_preprocess'
meta_info = pd.read_csv("../../data/meta_info_lung.csv")

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

# image embedding dim of foundation encoders
if args.image_model == 'hoptimus':
    args.image_emb_dim = 1536

cell_lineage = ['Epithelial', 'Immune', 'Endothelial', 'Mesenchymal']
cl_map = {c: i for i, c in enumerate(cell_lineage)}
cell_type = ['RASC', 'Secretory', 'Multiciliated', 'PNEC', 'Basal', 'Goblet', 'Proliferating Airway', 'AT2', 'Transitional AT2', 'AT1', 'KRT5-/KRT17+', 'Proliferating AT2', 'Langerhans cells', 'NK/NKT', 'Tregs', 'CD4+ T-cells', 'CD8+ T-cells', 'Proliferating T-cells', 'B cells', 'Plasma', 'pDCs', 'Proliferating NK/NKT', 'Proliferating B cells', 'cDCs', 'Mast', 'Interstitial Macrophages', 'Alveolar Macrophages', 'SPP1+ Macrophages', 'Neutrophils', 'Proliferating Myeloid', 'Migratory DCs', 'Macrophages - IFN-activated', 'Monocytes/MDMs', 'Basophils', 'Venous', 'Capillary', 'Lymphatic', 'Arteriole', 'SMCs/Pericytes', 'Alveolar FBs', 'Proliferating FBs', 'Inflammatory FBs', 'Activated Fibrotic FBs', 'Myofibroblasts', 'Subpleural FBs', 'Adventitial FBs', 'Mesothelial']
ct_map = {c: i for i, c in enumerate(cell_type)}

args.gene_total_num = 340
args.cell_type_num = 47
args.cell_type_num = 4

result_path = '/data/wuqitian/analysis_pred_data'

train_samples = meta_info[meta_info['affect'] == 'Unaffected']['sample'].tolist()[:-2]
train_samples += meta_info[meta_info['affect'] == 'Less Affected']['sample'].tolist()[:-2]
train_samples += meta_info[meta_info['affect'] == 'More Affected']['sample'].tolist()[:-2]
test_samples = meta_info[~meta_info['sample'].isin(train_samples)]['sample'].tolist()

# load pretrain model
gene_embeddings = torch.zeros((23258, args.gene_emb_dim), dtype=torch.float).to(device)
gene_embeddings = gene_embeddings
model_pretrain = parse_pretrain_method(args, gene_embeddings, device)
pretrained_state_dict = torch.load('../model_checkpoints/ours_pretrain_xenium_lung.pth')
model_pretrain.load_state_dict(pretrained_state_dict)

# load evaluate model
model_ours = parse_regression_method(args, device)
pretrained_state_dict = torch.load('../model_checkpoints/ours_pretrain_xenium_lung.pth')
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


# model update
train_datasets = dataset_create(dir_path, train_samples, args, data_loader='lung')
train_dataloader = DataLoader(train_datasets, batch_size=1, shuffle=True)

test_datasets = dataset_create(dir_path, test_samples, args, data_loader='lung')
test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False)

# datasets = dataset_create_split(dir_path, test_samples, args, valid_prop=0., test_prop=0.99, data_loader='lung')
# train_dataloader = DataLoader(datasets, batch_size=1, shuffle=False)
# test_dataloader = DataLoader(datasets, batch_size=1, shuffle=False)

run_update(model_ours, train_dataloader, device, use_gene_idx=False)

print("Update Finished")

evaluate_samples = train_samples + test_samples
# model evaluate
for i, sample in enumerate(test_samples[-1:]):
    file_path = os.path.join(dir_path, sample) + '.h5ad'
    adata = sc.read(file_path)

    gene_mask = adata.var['gene_filter_mask']
    gene_index = adata.var['gene_filtered_idx'][gene_mask]

    cell_image_emb = adata.obsm['embeddings']
    inputs = torch.tensor(cell_image_emb, dtype=torch.float).to(device)
    cell_by_gene = adata.layers['X_log1p'][:, gene_mask]
    y_ge = torch.tensor(cell_by_gene, dtype=torch.float).to(device)

    cell_location = torch.tensor(adata.obsm['centroids'], dtype=torch.float)
    edge_index = knn_graph(cell_location, k=5, loop=False).to(device)

    with torch.no_grad():
        image_embs_ours, gene_embs_ours = model_pretrain(inputs, y_ge, gene_index, edge_index)
        image_embs_ours, gene_embs_ours = image_embs_ours.cpu().numpy(), gene_embs_ours.cpu().numpy()
        embs_ours = model_ours.encoder1.get_embeddings(inputs, edge_index).cpu().numpy() # [layer num, spot num, hidden size]
        attn_ours = model_ours.encoder1.get_attentions(inputs, edge_index).cpu().numpy() # [layer num, spot num, hidden size]

        pred_ours = model_ours(inputs, edge_index=edge_index).cpu().numpy() # [spot num, gene num] gene specific to sample

        # pred_hoptimus = model_hoptimus(inputs, edge_index=edge_index).cpu().numpy()

    adata_new = adata[:, gene_mask].copy()
    adata_new.layers['X_pred_ours'] = pred_ours
    # adata_new.layers['X_pred_hoptimus'] = pred_hoptimus

    for i in range(embs_ours.shape[0]):
        adata_new.obsm[f'embs_ours_layer{i}'] = embs_ours[i]
    adata_new.obsm['image_embs_ours'] = image_embs_ours
    adata_new.obsm['gene_embs_ours'] = gene_embs_ours
    adata_new.obsm['embs_hoptimus'] = cell_image_emb

    # adata_new.obs['cell_type_pred_ours'] = pred_ct_ours
    # adata_new.obs['cell_type_pred_hoptimus'] = pred_ct_hoptimus

    attn_map = (attn_ours[0] + 1) / 2.
    # attn_map = attn_map / attn_map.sum(-1, keepdims=True)
    attn_map = (attn_map + attn_map.T) / 2.
    attn_map = attn_map / attn_map.max()
    condensed_dist = pdist(attn_map, metric='euclidean')
    linkage_matrix = linkage(condensed_dist, method='ward')
    cluster_labels = fcluster(linkage_matrix, t=10, criterion='maxclust')
    adata_new.obs['attn_cluster_labels'] = cluster_labels
    adata_new.uns['attn_linkage'] = linkage_matrix

    print(i, sample)
    adata_new.write_h5ad(os.path.join(result_path, sample) + '_pred.h5ad')




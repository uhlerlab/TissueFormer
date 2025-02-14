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

dir_path = '/data/wuqitian/hest_data_visium_protein_preprocess'
meta_info = pd.read_csv("../data/meta_info_visium.csv")

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

# human protein-coding genes as shared gene idx
gene_protein_human_idx = {}
with open('../data/human protein coding gene-GRCh38.p14.txt', 'r') as f:
    i = 0
    for line in f.readlines()[1:]:
        l = line[:-1].split('\t')
        if len(l) == 2 and len(l[-1]) > 0:
            gene = l[-1]
            if gene in gene_protein_human_idx.values():
                continue
            gene_protein_human_idx[i] = gene
            i += 1
args.gene_total_num = 23258

# image embedding dim of foundation encoders
if args.image_model == 'hoptimus':
    args.image_emb_dim = 1536
elif args.image_model == 'gigapath':
    args.image_emb_dim = 1536
elif args.image_model == 'uni':
    args.image_emb_dim = 1024

if args.domain_protocol == 'organ':
    # from frequent organs to rare organs, evaluate on Bladder samples, reference from pretrain organs
    pretrain_organs = ['Spinal cord', 'Brain', 'Breast', 'Bowel', 'Skin']  # top five organs
    train_samples = meta_info[meta_info['organ'].isin(pretrain_organs)]['sample'].tolist()
    reference_samples = train_samples[:20]
    # evaluate_samples = ['NCBI603', 'NCBI602', 'NCBI601', 'NCBI600'] # Bladder
    # evaluate_samples += ['NCBI684', 'NCBI683', 'NCBI682', 'NCBI681'] # Lymph node
    # evaluate_samples += ['NCBI572', 'NCBI571', 'NCBI570', 'NCBI569'] # Pancreas
    # evaluate_samples += ['MISC65', 'TENX80', 'NCBI800', 'TENX52', 'MISC34', 'NCBI802', 'TENX18', 'MEND129', 'MISC43', 'NCBI524', 'NCBI668', 'MEND130', 'TENX55', 'MEND74', 'NCBI663', 'MISC48', 'TENX69', 'TENX19', 'NCBI471', 'MISC67']
    # evaluate_samples += ['MEND156', 'MEND161', 'MEND157', 'TENX50', 'MEND59', 'MEND60', 'MISC13', 'INT23', 'TENX40', 'MEND159', 'NCBI594', 'NCBI592', 'MISC15', 'MISC16', 'NCBI593', 'NCBI591', 'NCBI642', 'TENX51', 'MEND154', 'INT24']
    evaluate_samples = meta_info[~meta_info['organ'].isin(pretrain_organs)]['sample'].tolist()

    pretrain_model_path = f'../model_checkpoints/ours_pretrain_visium_organ_v2.pth'
    result_path = '/data/wuqitian/hest_data_visium_organ_pred'
elif args.domain_protocol == 'mouse2human':
    # from mouse to human, evaluate on human Liver samples, reference from mouse Liver samples
    samples = ['NCBI844', 'NCBI843', 'NCBI842', 'NCBI841', 'NCBI840', 'NCBI839', 'NCBI838', 'NCBI837', 'NCBI836',
     'NCBI835', 'NCBI834', 'NCBI833', 'NCBI832', 'NCBI831', 'NCBI830', 'NCBI829', 'NCBI828', 'NCBI827', 'NCBI826']
    meta_info_ = meta_info[meta_info['sample'].isin(samples)]
    reference_samples = meta_info_[meta_info_['species']=='Mus musculus']['sample'].tolist()
    evaluate_samples = meta_info_[meta_info_['species']=='Homo sapiens']['sample'].tolist()

    pretrain_model_path = f'../model_checkpoints/ours_pretrain_visium_mouse2human_v2.pth'
    result_path = '/data/wuqitian/hest_data_visium_mouse2human_pred'
else:
    raise NotImplementedError

# load pretrain model
gene_embeddings = torch.zeros((args.gene_total_num, args.gene_emb_dim), dtype=torch.float).to(device)
model_ours_pretrain = parse_pretrain_method(args, gene_embeddings, device)
checkpoint = torch.load(pretrain_model_path)
model_ours_pretrain.load_state_dict(checkpoint, strict=True)

# load evaluate model
model_ours = parse_regression_method(args, device)
pretrained_state_dict = torch.load(pretrain_model_path)
encoder1_pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k.startswith("encoder1.")}
model_state_dict = model_ours.state_dict()
encoder1_model_dict = {k: v for k, v in model_state_dict.items() if k.startswith("encoder1.")}
for k, v in encoder1_pretrained_dict.items():
    assert (k in encoder1_model_dict)
    assert (v.size() == encoder1_model_dict[k].size())
model_state_dict.update(encoder1_pretrained_dict)
model_ours.load_state_dict(model_state_dict)

reference_datasets = dataset_create(dir_path, reference_samples, args)
reference_dataloader = DataLoader(reference_datasets, batch_size=1, shuffle=True)
run_update(model_ours, reference_dataloader, device, args.hvg_gene_top)

# hoptimus model
# args.method = 'hoptimus-MLP'
# model_hoptimus = parse_supervise_method(args, device)

for i, sample in enumerate(evaluate_samples):
    file_path = os.path.join(dir_path, sample) + '.h5ad'
    adata = sc.read(file_path)
    X_log1p = adata.layers['X_log1p']
    gene_mask = adata.var['gene_filter_mask']
    cell_by_gene = X_log1p[:, gene_mask]
    gene_index = adata.var['gene_filtered_idx'][gene_mask]
    hvg_gene_rank = adata.var['highly_variable_rank'][gene_mask]
    cell_image_emb = adata.obsm['embeddings']
    cell_location = adata.obsm['centroids']

    inputs = torch.tensor(cell_image_emb, dtype=torch.float).to(device)
    if sp.issparse(cell_by_gene):
        rows, cols = cell_by_gene.nonzero()
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(cell_by_gene.data, dtype=torch.float32)
        y = torch.sparse_coo_tensor(indices, values, torch.Size(cell_by_gene.shape)).to(device)
    else:
        y = torch.tensor(cell_by_gene, dtype=torch.float).to(device)
    gene_index = torch.tensor(gene_index, dtype=torch.long).to(device)
    edge_index = knn_graph(torch.tensor(cell_location, dtype=torch.float), k=5, loop=False).to(device)

    model_ours.eval()
    with torch.no_grad():
        embs_ours = model_ours.encoder1(inputs, edge_index) # [spot num, hidden size]
        pred_ours = model_ours(inputs, gene_index, edge_index) # [spot num, gene num] gene specific to sample
        embs_ours, pred_ours = embs_ours.cpu().numpy(), pred_ours.cpu().numpy()

        attn_ours = model_ours.encoder1.get_attentions(inputs, edge_index) # [layer num, spot num, spot num]
        attn_ours = attn_ours.cpu().numpy()

        # pred_hoptimus = model_hoptimus(inputs, gene_index, edge_index)

    adata_new = AnnData(X=cell_by_gene)
    adata_new.layers['X_pred_ours'] = pred_ours
    adata_new.obsm['embs_ours'] = embs_ours
    adata_new.obsm['embs_hoptimus'] = cell_image_emb
    adata_new.uns['attn_ours'] = attn_ours

    adata_new.obsm['cell_loc'] = cell_location

    gene_names = []
    gene_index = gene_index.cpu().numpy()
    for g_idx in gene_index:
        gene_names += [gene_protein_human_idx[int(g_idx)]]
    adata_new.var['gene_names'] = gene_names
    adata_new.var['gene_index'] = gene_index

    if 'gene_intersection_mask' in adata.var:
        gene_intersection_mask = adata.var['gene_intersection_mask'].to_numpy(dtype=bool)
        gene_eval_mask = gene_intersection_mask[gene_mask]  # indicate which gene idx should be evaluated
        adata_new.var['gene_eval_mask'] = gene_eval_mask

    print(i, sample)
    adata_new.write_h5ad(os.path.join(result_path, sample) + '_pred.h5ad')




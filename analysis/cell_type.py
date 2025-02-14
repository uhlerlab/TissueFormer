import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import warnings
import os

from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import anndata as ad

result_path = '/data/wuqitian/analysis_pred_data'
data_path = '/data/wuqitian/hest_data_xenium_protein_preprocess'

samples = ['TENX126', 'TENX122', 'TENX121', 'TENX119', 'TENX118'] # six different organs
organs = ['Pancreas', 'Skin', 'Liver', 'Heart', 'Lung']

# for i, s in enumerate(samples):
#     file_path = os.path.join(data_path, s) + '_pred.h5ad'
#     adata_i = sc.read(file_path)
#
#     sc.pp.filter_cells(adata_i, min_genes=5)
#     idx = adata_i.obs.index.to_numpy().astype(int)
#     adata_i.uns['attn_ours'] = adata_i.uns['attn_ours'].mean(0)
#     adata_i.uns['attn_ours'] = adata_i.uns['attn_ours'][idx, :]
#     adata_i.uns['attn_ours'] = adata_i.uns['attn_ours'][:, idx]
#     adata_i.obs.index = [f"{i}" for i in range(adata_i.n_obs)]
#
#     sc.pp.pca(adata_i, n_comps=50)
#     sc.pp.neighbors(adata_i, n_neighbors=15)
#     sc.tl.leiden(adata_i, resolution=0.5, flavor="igraph", n_iterations=2, directed=False)
#     sc.tl.umap(adata_i)
#
#     cluster_num = adata_i.obs['leiden'].unique().shape[0]
#     print(f"Leiden cluster number: {cluster_num}")
#
#     attn_cluster_map = np.zeros((cluster_num, cluster_num), dtype=np.float32)
#     cluster_idx_ = [adata_i[adata_i.obs['leiden'] == c].obs.index.to_numpy().astype(int) for c in adata_i.obs['leiden'].unique().tolist()]
#     for i in range(cluster_num):
#         cluster_idx = cluster_idx_[i]
#         attn_cluster = adata_i.uns['attn_ours'][cluster_idx, :]
#         for j in range(cluster_num):
#             cluster_idx2 = cluster_idx_[j]
#             attn_cluster_map[i, j] = attn_cluster[:, cluster_idx2].mean()
#
#     adata_i.uns['attn_ours'] = None
#     adata_i.uns['attn_cluster_map'] = attn_cluster_map
#
#     adata_i.write(os.path.join(result_path, f'{s}_analysis.h5ad'))

adata_list = []

for i, s in enumerate(samples):
    file_path = os.path.join(data_path, s) + '.h5ad'
    adata_i = sc.read(file_path)
    gene_mask = adata_i.var['gene_filter_mask']
    adata_i = adata_i[:, gene_mask]
    adata_i.var_names = adata_i.var['gene_names']
    adata_list += [adata_i]

adata_merged = ad.concat(adata_list, join='outer', label='organ', keys=organs)
sc.pp.filter_cells(adata_merged, min_genes=5)
# sc.external.pp.scanorama_integrate(adata_merged, 'organ')
sc.pp.pca(adata_merged, n_comps=50)
sc.pp.neighbors(adata_merged, n_neighbors=15)
sc.tl.leiden(adata_merged, resolution=1.0, flavor="igraph", n_iterations=2, directed=False)
cluster_num = adata_merged.obs['leiden'].unique().shape[0]
print(f"Leiden cluster number: {cluster_num}")
sc.tl.umap(adata_merged)
adata_merged.write(os.path.join(result_path, 'sample_merge_data.h5ad'))
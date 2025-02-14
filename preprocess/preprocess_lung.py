import pickle
import os
import pandas as pd
import numpy as np
import warnings

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

import scanpy as sc
import anndata
import csv

from data_preprocess import Preprocessor

import torch

# gene synonym map of human genes
gene_synonym_human_map = {}
gene_synonym_human = {}
with open('../data/human-genename-synonym-GRCh38.p14.txt', 'r') as f:
    for i, line in enumerate(f.readlines()[1:]):
        l = line[:-1].split('\t')
        for gene in l:
            gene_synonym_human_map[gene] = i
        gene_synonym_human[i] = l

# human protein-coding genes
gene_protein_human_map = {}
with open('../data/human protein coding gene-GRCh38.p14.txt', 'r') as f:
    i = 0
    for line in f.readlines()[1:]:
        l = line[:-1].split('\t')
        if len(l) == 2 and len(l[-1]) > 0:
            gene = l[-1]
            if gene in gene_protein_human_map.keys():
                continue
            gene_protein_human_map[gene] = i
            i += 1

preprocessor = Preprocessor(
    use_key=None,  # the key in adata.layers to use as raw data
    filter_gene_by_counts=0,  # step 1
    filter_cell_by_counts=0,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=True,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=500,  # 5. whether to subset the raw data to highly variable genes
    hvg_use_key=None,
    hvg_flavor="seurat_v3",
    binning=None,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)

cell_lineage = ['Epithelial', 'Immune', 'Endothelial', 'Mesenchymal']
cl_map = {c: i for i, c in enumerate(cell_lineage)}
cell_type = ['RASC', 'Secretory', 'Multiciliated', 'PNEC', 'Basal', 'Goblet', 'Proliferating Airway', 'AT2', 'Transitional AT2', 'AT1', 'KRT5-/KRT17+', 'Proliferating AT2', 'Langerhans cells', 'NK/NKT', 'Tregs', 'CD4+ T-cells', 'CD8+ T-cells', 'Proliferating T-cells', 'B cells', 'Plasma', 'pDCs', 'Proliferating NK/NKT', 'Proliferating B cells', 'cDCs', 'Mast', 'Interstitial Macrophages', 'Alveolar Macrophages', 'SPP1+ Macrophages', 'Neutrophils', 'Proliferating Myeloid', 'Migratory DCs', 'Macrophages - IFN-activated', 'Monocytes/MDMs', 'Basophils', 'Venous', 'Capillary', 'Lymphatic', 'Arteriole', 'SMCs/Pericytes', 'Alveolar FBs', 'Proliferating FBs', 'Inflammatory FBs', 'Activated Fibrotic FBs', 'Myofibroblasts', 'Subpleural FBs', 'Adventitial FBs', 'Mesothelial']
ct_map = {c: i for i, c in enumerate(cell_type)}

dir_path = '/data/lizhiyi/lung'
preprocess_save_path = '/data/wuqitian/lung_preprocess'

files = os.listdir(dir_path)

meta_data = sc.read('/data/wuqitian/lung/GSE250346_cell_type_niche_centroid.h5ad')
he_annotate_data = pd.read_csv(open('/data/wuqitian/lung/HE_annotations/cells_partitioned_by_annotation.csv', 'r', encoding='utf-8'))
sample_id_map = {
 'VUHD095': 'VUHD095', 'THD0011': 'THD0011', 'VUHD116A': 'VUHD116A', 'VUHD116B': 'VUHD116B', 'VUHD069': 'VUHD069', 'TILD117LA': 'TILD117LF', 'VUILD110LA': 'VUILD110',
 'VUILD96LA': 'VUILD96LF', 'VUILD102LA': 'VUILD102LF', 'VUILD48LA1': 'VUILD48LF', 'VUILD78MA': 'VUILD78MF', 'VUILD115MA': 'VUILD115', 'VUILD104MA1': 'VUILD104LF',
 'VUILD105MA1': 'VUILD105LF', 'VUILD96MA': 'VUILD96MF', 'VUILD105MA2': 'VUILD105MF', 'VUILD107MA': 'VUILD107MF', 'TILD175MA': 'TILD175', 'VUILD104MA2': 'VUILD104MF',
 'VUILD106MA': 'VUILD106', 'VUILD48LA2': 'VUILD48MF', 'THD0008': 'THD0008', 'VUHD113': 'VUHD113', 'VUILD91MA': 'VUILD91MF', 'VUILD78LA': 'VUILD78LF', 'VUILD102MA': 'VUILD102MF'
}
he_annotate_type = ['normal_alveoli', 'small_airway', 'minimally_remodeled_alveoli', 'artery', 'venule', 'remodeled_epithelium', 'advanced_remodeling', 'fibrosis', 'multinucleated_cell', 'epithelial_detachment', 'mixed_inflammation', 'muscularized_artery', 'severe_fibrosis', 'large_airway', 'goblet_cell_metaplasia', 'hyperplastic_aec', 'fibroblastic_focus', 'granuloma', 'microscopic_honeycombing', 'TLS', 'airway_smooth_muscle', 'remnant_alveoli', 'giant_cell', 'interlobular_septum', 'emphysema']

samples, types, affects, status, percent = [], [], [], [], []

for i, f in enumerate(files):
    file_path = os.path.join(dir_path, f)
    adata = sc.read(file_path)
    sample = f.split('.')[0]
    print(i, sample)

    # modify gene names, filter genes
    gene_name_new = []
    gene_filter_mask, gene_filtered_idx = [], []
    for gi in adata.var.index:
        # handle special symbols

        # find all synonym genes
        gi_synonym = [gi]
        if gi in gene_synonym_human_map.keys():
            gi_synonym += gene_synonym_human[gene_synonym_human_map[gi]]

        # filter and map to protein-coding genes
        gi_ = gi
        gi_mask = False
        gi_idx = -1
        for si in gi_synonym:  # find protein-coding gene over all synonym
            if si in gene_protein_human_map.keys():  # find one human protein-coding gene
                gi_ = si
                gi_mask = True
                gi_idx = gene_protein_human_map[si]
                continue
        gene_filter_mask.append(gi_mask)
        gene_filtered_idx.append(gi_idx)
        gene_name_new.append(gi_)

    adata.var['gene_names'] = gene_name_new
    adata.var['gene_filter_mask'] = gene_filter_mask
    adata.var['gene_filtered_idx'] = gene_filtered_idx

    print(i, sample, adata.var['gene_filter_mask'].shape, adata.var['gene_filter_mask'].sum())

    # remove original normalize
    mean = adata.var['mean'].to_numpy().reshape(1, -1)
    std = adata.var['std'].to_numpy().reshape(1, -1)
    adata.X = adata.X * std + mean
    adata.X = np.exp(adata.X) - 1
    adata.X[adata.X < 0.5] = 0.
    adata.X = adata.X.astype(int)

    adata.obsm['embeddings'] = adata.obsm['patch_embs']
    del adata.obsm['patch_embs']
    x_centroid = adata.obs['x_centroid'].to_numpy()
    y_centroid = adata.obs['y_centroid'].to_numpy()
    adata.obsm['centroids'] = np.stack([x_centroid, y_centroid], axis=1)
    del adata.obs['x_centroid'], adata.obs['y_centroid']

    # add cell type annotation
    adata.obs['final_CT'] = adata.obs['final_CT'].map(ct_map)
    adata.obs['final_lineage'] = adata.obs['final_lineage'].map(cl_map)

    # add HE annotation as multi-label array
    he_annotate_array = np.zeros((adata.shape[0], len(he_annotate_type)), dtype=np.int32)
    he_annotate_data_s = he_annotate_data[he_annotate_data['sample']==sample_id_map[sample]]
    for i, an in enumerate(he_annotate_type):
        he_s_i = he_annotate_data_s[he_annotate_data_s['annotation_type']==an]
        array_i = adata.obs['cell_id'].isin(he_s_i['cell_id'].tolist()).to_numpy().astype(int)
        he_annotate_array[:, i] = array_i
    adata.obsm['he_annotation'] = he_annotate_array
    print(adata.shape[0], adata.obsm['he_annotation'].sum(), he_annotate_data_s.shape[0])

    preprocessor(adata)
    adata.write_h5ad(os.path.join(preprocess_save_path, sample) + '.h5ad')

    samples.append(sample)
    types.append(meta_data[meta_data.obs['sample']==sample].obs['sample_type'].unique().tolist()[0])
    affects.append(meta_data[meta_data.obs['sample']==sample].obs['sample_affect'].unique().tolist()[0])
    status.append(meta_data[meta_data.obs['sample']==sample].obs['disease_status'].unique().tolist()[0])
    percent.append(meta_data[meta_data.obs['sample']==sample].obs['percent_pathology'].unique().tolist()[0])

meta_info = {'sample': samples, 'type': types, 'affect': affects, 'status': status, 'percent_pathology': percent}
meta_df = pd.DataFrame(meta_info)
meta_df.to_csv('../data/meta_info_lung.csv', index=False)
print(meta_df.shape)


# dir_path2 = '/data/lizhiyi/lung_new'
# files = os.listdir(preprocess_save_path)
# for i, f in enumerate(files):
#     file_path = os.path.join(preprocess_save_path, f)
#     adata = sc.read(file_path)
#     print(adata)
#
#     sample = f.split('.')[0]
#     print(i, sample)
#     file_path2 = os.path.join(dir_path2, f)
#     adata2 = sc.read(file_path2)
#     print(adata2)
#     adata.obsm["patch_embs_gigapath"] = adata2.obsm["patch_embs_gigapath"]
#     adata.obsm["patch_embs_uni"] = adata2.obsm["patch_embs_uni"]
#     # adata.obsm["patch_pca_100"] = adata2.obsm["patch_pca_100"]
#     adata.write_h5ad(os.path.join(preprocess_save_path, sample) + '.h5ad')
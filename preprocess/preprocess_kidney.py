import pickle
import os
import pandas as pd
import numpy as np
import warnings

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

import scanpy as sc

from data_preprocess import Preprocessor

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
    subset_hvg=1000,  # 5. whether to subset the raw data to highly variable genes
    hvg_use_key=None,
    hvg_flavor="seurat_v3",
    binning=None,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)

macrophage_type = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14',
                   '15', '16', '17', '18', '19', '20', '21', '22']
mt_map = {c: i for i, c in enumerate(macrophage_type)}
confidence = ['0_to_0.49', '0.5_to_0.74', '0.75_to_1']
c_map = {c: i for i, c in enumerate(confidence)}

label_files = [
    'label_transfer_amp2_to_xo_ctrl_1_k200_thresh_categories.csv',
    'label_transfer_amp2_to_xo_ctrl_2_k200_thresh_categories.csv',
    'label_transfer_amp2_to_xo_A_k200_thresh_categories.csv',
    'label_transfer_amp2_to_xo_B_k200_thresh_categories.csv',
    'label_transfer_amp2_to_xo_C_k200_thresh_categories.csv',
    'label_transfer_amp2_to_xo_D_k200_thresh_categories.csv',
    'label_transfer_amp2_to_xo_E_k200_thresh_categories.csv',
    'label_transfer_amp2_to_xo_F_k200_thresh_categories.csv'
]

original_path = '/data/wuqitian/kidney'
dir_path = '/data/wuqitian/kidney_converted'
preprocess_save_path = '/data/wuqitian/kidney_preprocess'

samples = ['ctrl_01', 'ctrl_02', 'ptA', 'ptB', 'ptC', 'ptD', 'ptE', 'ptF']

for i, sample in enumerate(samples):
    file_path = os.path.join(dir_path, sample) + '.h5ad'
    adata = sc.read(file_path)
    print(i, sample)

    adata.var = adata.var.rename(columns={"gene_name": "gene_names"})
    # modify gene names, filter genes
    gene_name_new = []
    gene_filter_mask, gene_filtered_idx = [], []
    for gi in adata.var['gene_names'].tolist():
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

    annotation_df = pd.read_csv(original_path + f'/cell_annotations/{sample}/{label_files[i]}')
    mt_str = annotation_df['group'].tolist()
    annotation_df['macrophage_type'] = [s.split('__')[0] for s in mt_str]
    annotation_df['macrophage_type'] = annotation_df['macrophage_type'].map(mt_map)
    annotation_df['confidence'] = [s.split('__')[1] for s in mt_str]
    annotation_df['confidence'] = annotation_df['confidence'].map(c_map)

    adata.obs['macrophage_label'] = None
    mapping_m = annotation_df.set_index('cell_id')['macrophage_type'].to_dict()
    adata.obs['macrophage_label'] = adata.obs['cell_id'].map(mapping_m)
    print(adata[adata.obs['macrophage_label']>=0.].shape)
    adata.obs['confidence_level'] = None
    mapping_c = annotation_df.set_index('cell_id')['confidence'].to_dict()
    adata.obs['confidence_level'] = adata.obs['cell_id'].map(mapping_c)
    print(adata[adata.obs['confidence_level']>=0.].shape)
    adata.obs['macrophage_mask'] = (adata.obs['macrophage_label']>=0)
    print(adata.obs['macrophage_mask'].sum(), adata.obs['macrophage_mask'].shape)

    adata.obs['macrophage_label'] = adata.obs['macrophage_label'].fillna(0).astype(int)
    adata.obs['confidence_level'] = adata.obs['confidence_level'].fillna(0).astype(int)

    preprocessor(adata)
    adata.write_h5ad(os.path.join(preprocess_save_path, sample) + '.h5ad')



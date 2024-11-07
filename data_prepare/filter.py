import pickle
import os
import pandas as pd
import numpy as np
import warnings

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

import scanpy as sc
import anndata

from preprocess import Preprocessor

preprocessor = Preprocessor(
    use_key="X_scgpt_human",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=False,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=True,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=20,  # 5. whether to subset the raw data to highly variable genes
    hvg_use_key="X_scgpt_human",
    hvg_flavor="seurat_v3",
    binning=None,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)

gene_embeddings_dict = pickle.load(open('../../data/gene_embeddings_scgpt_human.pkl', 'rb'))
gene_filter_name = list(gene_embeddings_dict.keys())
gene_filter_map = {gene: i for i, gene in enumerate(gene_filter_name)}

data_csv = pd.read_csv("../../data/HEST_v1_1_0.csv")

gene_name_map_csv = pd.read_csv("../../data/gene_info.csv")
gene_name_1, gene_name_2 = gene_name_map_csv['feature_id'].tolist(), gene_name_map_csv['feature_name'].tolist()
gene_name_map = {gene_name_1[i]: gene_name_2[i] for i in range(len(gene_name_1))}

def get_new_data(dir_path, save_path):
    files = os.listdir(dir_path)
    total_slides = len(files)
    human_slides = 0
    valid_slides = 0
    for i, f in enumerate(files):
        file_path = os.path.join(dir_path, f)
        if os.path.isdir(file_path):
            total_slides -= 1
        else:
            adata = sc.read(file_path)
            sample = f.split('.')[0]
            species = data_csv[data_csv['id'] == sample]['species'].values[0]
            organ = data_csv[data_csv['id'] == sample]['organ'].values[0]
            tech = data_csv[data_csv['id'] == sample]['st_technology'].values[0]
            print(i, sample, species, organ, tech)
            # print(adata.X.shape, adata.obs.shape, adata.var.shape, adata.obsm['embeddings'].shape,
            #       adata.obsm['centroids'].shape)

            if 'gene_name' in adata.var.columns:
                adata.var = adata.var.rename(columns={"gene_name": "gene_names"})

            if species == 'Homo sapiens':
                human_slides += 1

                gene_name = adata.var['gene_names']

                # modify gene names
                gene_name_new = []
                for gene in gene_name:
                    g = gene.split('______')
                    gi = g[-1] if len(g) > 1 else g[0]
                    gi_ = gene_name_map[gi] if gi in list(gene_name_map.keys()) else gi
                    gene_name_new.append(gi_)
                adata.var['gene_names'] = gene_name_new

                # filter genes overlapped with scGPT
                gene_mask_scgpt_human, gene_filtered_idx = [], []
                for gene in gene_name_new:
                    if gene in gene_filter_name:
                        gene_mask_scgpt_human += [True]
                        gene_filtered_idx += [gene_filter_map[gene]]
                    else:
                        gene_mask_scgpt_human += [False]
                        gene_filtered_idx += [-1]
                adata.var['gene_mask_scgpt_human'] = gene_mask_scgpt_human
                adata.var['gene_filtered_idx'] = gene_filtered_idx
                X_copy = adata.X.copy()
                X_copy[:, ~np.array(gene_mask_scgpt_human, dtype=np.bool)] = 0.
                adata.layers['X_scgpt_human'] = X_copy

                adata.uns['species'] = species
                adata.uns['organ'] = organ
                adata.uns['tech'] = tech

                # print(adata)
                print(adata.X.shape, adata.obs.shape, adata.var.shape, adata.obsm['embeddings'].shape,
                      adata.obsm['centroids'].shape)

                if sum(gene_mask_scgpt_human) >= 20:
                    try:
                        preprocessor(adata)
                        valid_slides += 1
                        # preprocess gene expressions
                        adata.write_h5ad(os.path.join(save_path, sample)+'.h5ad')
                    except:
                        pass
    print(total_slides, human_slides, valid_slides)


dir_path1 = '/data/lizhiyi/hest_data_converted_visium'
dir_path2 = '/data/lizhiyi/hest_data_converted_new'

save_path1 = '/data/wuqitian/hest_data_visium'
save_path2 = '/data/wuqitian/hest_data_xenium'

# get_new_data(dir_path2, save_path2)
get_new_data(dir_path1, save_path1)

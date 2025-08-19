import pickle
import os
import pandas as pd
import numpy as np
import warnings

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

import scanpy as sc
import anndata

from data_preprocess import Preprocessor

import torch

# data meta info of HEST-1K
data_csv = pd.read_csv("../data/HEST_v1_1_0.csv")

# gene name and embs from scGPT Human, in total 60k
gene_embeddings_dict = pickle.load(open('../data/gene_embeddings_scgpt_human.pkl', 'rb'))
gene_filter_name = list(gene_embeddings_dict.keys())
gene_filter_map_scgpt = {gene: i for i, gene in enumerate(gene_filter_name)}

# gene map from ENS to name of scGPT Human, in total 60k
gene_name_map_csv = pd.read_csv("../data/gene_info.csv")
gene_name_1, gene_name_2 = gene_name_map_csv['feature_id'].tolist(), gene_name_map_csv['feature_name'].tolist()
gene_name_map_scgpt = {gene_name_1[i]: gene_name_2[i] for i in range(len(gene_name_1))}

# gene synonym map of human genes
gene_synonym_human_map = {}
gene_synonym_human = {}
with open('../data/human-genename-synonym-GRCh38.p14.txt', 'r') as f:
    for i, line in enumerate(f.readlines()[1:]):
        l = line[:-1].split('\t')
        for gene in l:
            gene_synonym_human_map[gene] = i
        gene_synonym_human[i] = l

# gene synonym map of mouse genes
gene_synonym_mouse_map = {}
gene_synonym_mouse = {}
with open('../data/mouse-genename-synonym-GRCm39.txt', 'r') as f:
    for i, line in enumerate(f.readlines()[1:]):
        l = line[:-1].split('\t')
        for gene in l:
            gene_synonym_mouse_map[gene] = i
        gene_synonym_mouse[i] = l

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

# mouse protein-coding genes
gene_protein_mouse_map = {}
with open('../data/mouse protein coding gene-GRCm39.txt', 'r') as f:
    i = 0
    for line in f.readlines()[1:]:
        l = line[:-1].split('\t')
        if len(l) == 2 and len(l[-1]) > 0:
            gene = l[-1]
            if gene in gene_protein_mouse_map.keys():
                continue
            gene_protein_mouse_map[gene] = i
            i += 1

# gene map from mouse protein-coding genes to human protein-coding genes
gene_mouse_to_human = {}
gene_protein_filter_map = {}
with open('../data/Human Orthology Mouse_ProteinCoding-MGI.rpt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        l = line[:-1].split('\t')
        gene_mouse_to_human[l[1]] = l[4]
        gene_protein_filter_map[l[4]] = i

def convert_filter_gene_names(dir_path, save_path):
    files = os.listdir(dir_path)
    for i, f in enumerate(files):
        file_path = os.path.join(dir_path, f)
        if os.path.isdir(file_path):
            continue
        adata = sc.read(file_path)
        sample = f.split('.')[0]
        species = data_csv[data_csv['id'] == sample]['species'].values[0]
        organ = data_csv[data_csv['id'] == sample]['organ'].values[0]
        tech = data_csv[data_csv['id'] == sample]['st_technology'].values[0]
        adata.uns['species'] = species
        adata.uns['organ'] = organ
        adata.uns['tech'] = tech

        print(i, sample, species, organ, tech)

        if 'gene_name' in adata.var.columns:
            adata.var = adata.var.rename(columns={"gene_name": "gene_names"})

        gene_name = adata.var['gene_names']

        # modify gene names, filter genes
        gene_name_new = []
        gene_filter_mask, gene_filtered_idx = [], []
        for gene in gene_name:
            # handle special symbols
            g = gene.split('______')[-1]
            g = g.split('___________')[-1]
            g = g.split('_')[-1]
            gi = g.split('.')[0]

            # find all synonym genes
            gi_synonym = [gi]
            if species == 'Homo sapiens':
                if gi in gene_synonym_human_map.keys():
                    gi_synonym += gene_synonym_human[gene_synonym_human_map[gi]]
            else: # mouse
                if gi in gene_synonym_mouse_map.keys():
                    gi_synonym += gene_synonym_mouse[gene_synonym_mouse_map[gi]]

            # filter and map to protein-coding genes
            gi_ = gi
            gi_mask = False
            gi_idx = -1
            if species == 'Homo sapiens':
                for si in gi_synonym: # find protein-coding gene over all synonym
                    if si in gene_protein_human_map.keys(): # find one human protein-coding gene
                        gi_ = si
                        gi_mask = True
                        gi_idx = gene_protein_human_map[si]
                        continue
            else:
                for si in gi_synonym: # find protein-coding gene over all synonym
                    if si in gene_mouse_to_human.keys(): # find correspondence with a human gene
                        si_human = gene_mouse_to_human[si]  # convert to human gene name for obtaining map idx
                        if si_human in gene_protein_human_map.keys(): # is human protein-coding gene
                            gi_ = si # record mouse gene name in data
                            gi_mask = True
                            gi_idx = gene_protein_human_map[si_human]
                            continue
            gene_filter_mask.append(gi_mask)
            gene_filtered_idx.append(gi_idx)
            gene_name_new.append(gi_)

        adata.var['gene_names'] = gene_name_new
        adata.var['gene_filter_mask'] = gene_filter_mask
        adata.var['gene_filtered_idx'] = gene_filtered_idx

        X_copy = adata.X.copy()
        X_copy[:, ~np.array(gene_filter_mask, dtype=np.bool)] = 0.
        adata.layers['X_filter_protein'] = X_copy

        # print(adata)
        adata.write_h5ad(os.path.join(save_path, sample) + '.h5ad')
        print(adata.X.shape[1], sum(gene_filter_mask))

def gene_preprocess(dir_path, save_path, tech='visium'):
    files = os.listdir(dir_path)
    total_slide, valid_slide = len(files), 0

    if tech == 'visium':
        for i, f in enumerate(files):
            file_path = os.path.join(dir_path, f)
            adata = sc.read(file_path)
            sample = f.split('.')[0]
            if adata.uns['tech'] == 'Xenium' or sample in ['TENX56', 'TENX28', 'TENX23', 'TENX30']:
                continue
            gene_mask = adata.var['gene_filter_mask']
            gene_index = adata.var['gene_filtered_idx'][gene_mask].tolist()
            if i == 0:
                gene_index_share = gene_index
            else:
                gene_index_share = list(set(gene_index_share).intersection(gene_index))
        print(f"Intersection gene panel size: {len(gene_index_share)}")

        preprocessor = Preprocessor(
            use_key="X_filter_protein",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=0,  # step 1
            filter_cell_by_counts=0,  # step 2
            normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=True,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=len(gene_index_share),  # 5. whether to subset the raw data to highly variable genes
            hvg_use_key="X_filter_protein_intersection",
            hvg_flavor="seurat_v3",
            binning=None,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )
        for i, f in enumerate(files):
            file_path = os.path.join(dir_path, f)
            adata = sc.read(file_path)
            sample = f.split('.')[0]
            print(i, sample, adata.uns['species'], adata.uns['organ'], adata.uns['tech'])
            if adata.uns['tech'] == 'Xenium' or sample in ['TENX56', 'TENX28', 'TENX23', 'TENX30']:
                continue

            gene_intersection_mask = adata.var['gene_filtered_idx'].isin(gene_index_share).tolist()
            adata.var['gene_intersection_mask'] = gene_intersection_mask
            X_copy = adata.X.copy()
            X_copy[:, ~np.array(gene_intersection_mask, dtype=np.bool)] = 0.
            adata.layers['X_filter_protein_intersection'] = X_copy

            # preprocess gene expressions
            try:
                preprocessor(adata)
                adata.var['highly_variable_rank'] = np.nan_to_num(adata.var['highly_variable_rank'], nan=gene_intersection_mask.sum())
                valid_slide += 1
                adata.write_h5ad(os.path.join(save_path, sample) + '.h5ad')
                print(adata.layers['X_filter_protein_intersection'].shape, adata.var['gene_intersection_mask'].sum())
            except:
                print(f"Failed to compute hvg of {sample}")
                pass

    else: # for xenium samples
        preprocessor = Preprocessor(
            use_key="X_filter_protein",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=0,  # step 1
            filter_cell_by_counts=0,  # step 2
            normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=True,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=20000,  # 5. whether to subset the raw data to highly variable genes
            hvg_use_key="X_filter_protein",
            hvg_flavor="seurat_v3",
            binning=None,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )

        for i, f in enumerate(files):
            file_path = os.path.join(dir_path, f)
            adata = sc.read(file_path)
            sample = f.split('.')[0]
            if sample not in ['TENX157', 'TENX158', 'TENX159', 'TENX143']:
                continue
            print(i, sample, adata.uns['species'], adata.uns['organ'], adata.uns['tech'])

            # preprocess gene expressions
            try:
                preprocessor(adata)
                valid_slide += 1
                adata.write_h5ad(os.path.join(save_path, sample) + '.h5ad')
                print(adata.layers['X_filter_protein'].shape, adata.X.shape)
            except:
                print(f"Failed to compute hvg of {sample}")
                pass
    print(valid_slide, total_slide)

def meta_data_info(dir_path, save_path):
    files = os.listdir(dir_path)
    organs, species, techs, states, samples = [], [], [], [], []
    for i, f in enumerate(files):
        file_path = os.path.join(dir_path, f)
        if os.path.isdir(file_path):
            continue
        adata = sc.read(file_path)
        sample = f.split('.')[0]
        print(i, sample, adata.uns['species'], adata.uns['organ'], adata.uns['tech'])
        samples.append(sample)
        organs.append(adata.uns['organ'])
        species.append(adata.uns['species'])
        techs.append(adata.uns['tech'])

        state = data_csv[data_csv['id'] == sample]['disease_state'].values[0]
        states.append(state)

    meta_info = {'sample': samples, 'organ': organs, 'species': species, 'tech': techs, 'state': states}
    meta_df = pd.DataFrame(meta_info)
    meta_df.to_csv(save_path, index=False)
    print(meta_df.shape)

def gene_pretrain_embs_process():
    avg = torch.tensor(list(gene_embeddings_dict.values())).mean(0)
    gene_embs_scgpt_human_protein = torch.zeros((len(gene_protein_human_map), 512), dtype=torch.float)
    gene_scgpt = list(gene_filter_map_scgpt.keys())
    count = 0
    for i, g in enumerate(gene_protein_human_map.keys()):
        if g in gene_scgpt:
            gene_embs_scgpt_human_protein[i] = torch.tensor(gene_embeddings_dict[g])
            count += 1
            continue
        g_synonyms = gene_synonym_human[gene_synonym_human_map[g]]
        for gi in g_synonyms:
            if gi in gene_scgpt:
                gene_embs_scgpt_human_protein[i] = torch.tensor(gene_embeddings_dict[gi])
                count += 1
                continue
        gene_embs_scgpt_human_protein[i] = avg

    print(count, len(gene_protein_human_map))
    with open('../data/gene_embeddings_scgpt_human_protein.pkl', 'wb') as f:
        pickle.dump(gene_embs_scgpt_human_protein, f)

if __name__ == '__main__':
    dir_path1 = '/data/lizhiyi/hest_converted'
    save_path1 = '/data/wuqitian/hest_data_visium_protein'
    preprocess_save_path1 = '/data/wuqitian/hest_data_visium_protein_preprocess'
    meta_info_save_path1 = '../data/meta_info_visium.csv'
    convert_filter_gene_names(dir_path1, save_path1)
    gene_preprocess(save_path1, preprocess_save_path1, tech='visium')
    meta_data_info(preprocess_save_path1, meta_info_save_path1)

    dir_path2 = '/data/lizhiyi/hest_converted/xenium_cell'
    save_path2 = '/data/wuqitian/hest_data_xenium_protein'
    preprocess_save_path2 = '/data/wuqitian/hest_data_xenium_protein_preprocess'
    meta_info_save_path2 = '../data/meta_info_xenium.csv'
    convert_filter_gene_names(dir_path2, save_path2)
    gene_preprocess(save_path2, preprocess_save_path2, tech='xenium')
    meta_data_info(preprocess_save_path2, meta_info_save_path2)

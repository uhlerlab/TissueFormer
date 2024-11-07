import pickle
import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import warnings

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

import scanpy as sc
import anndata
import csv

gene_embeddings_dict = pickle.load(open('../data/gene_embeddings_scgpt_human.pkl', 'rb'))
gene_filter_name = list(gene_embeddings_dict.keys())

# print(gene_filter_name)

data_csv = pd.read_csv("../data/HEST_v1_1_0.csv")

gene_name_map_csv = pd.read_csv("../data/gene_info.csv")
gene_name_1, gene_name_2 = gene_name_map_csv['feature_id'].tolist(), gene_name_map_csv['feature_name'].tolist()
gene_name_map = {gene_name_1[i]: gene_name_2[i] for i in range(len(gene_name_1))}

dir_path = '/data/wuqitian/hest_data_visium'
files = os.listdir(dir_path)
total_slides = 0
datainfo = {}
sample, organ, species, state, technology = [], [], [], [], []
filter = []

for f in files[:1]:
    file_path = os.path.join(dir_path, f)
    if os.path.isdir(file_path):
        total_slides -= 1
        filter_slides = total_slides
    else:
        adata = sc.read(file_path)
        # print(f, np.sum(adata.var['gene_mask_scgpt_human']))
        print(adata.X[0, 21:53])
        print(adata.layers['X_scgpt_human'][0, 21:53])
        print(adata.layers['X_log1p'][0, 21:53])
        # s = f.split('.')[0]
        # sample += [s]
        # p = data_csv[data_csv['id'] == s]['species'].values[0]
        # species += [p]
        # o = data_csv[data_csv['id'] == s]['organ'].values[0]
        # organ += [o]
        # t = data_csv[data_csv['id'] == s]['st_technology'].values[0]
        # technology += [t]
        # d = data_csv[data_csv['id'] == s]['disease_state'].values[0]
        # state += [d]
        # print(adata.layers['X_log1p'])
        # print(adata.var['highly_variable'].sum())
        # print(adata.var['highly_variable_rank'].sort_values(ascending=False))

# data_meta_info = {'sample': sample,
#                   'organ': organ,
#                   'species': species,
#                   'state': state,
#                   'technology': technology}
#
# data_meta_info = pd.DataFrame(data_meta_info)
# data_meta_info.to_csv('../data/meta_info_xenium.csv', index=True)
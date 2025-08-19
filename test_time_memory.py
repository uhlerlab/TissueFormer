import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import random
import argparse
import pickle

from run_pretrain import run_pretrain
from run_finetune import run_update
from utils import dataset_create, dataset_create_split
from parse import parse_pretrain_method, parse_regression_method, parser_add_main_args
import os
import warnings
import time

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

import subprocess
def get_gpu_memory_map(device):
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    return gpu_memory[device]

### Parse args ###
parser = argparse.ArgumentParser(description='Pretraining Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()

fix_seed(args.seed)

dir_path = '/data/wuqitian/hest_data_xenium_protein_preprocess'
meta_info = pd.read_csv("../data/meta_info_xenium.csv")
meta_info = meta_info[meta_info['tech'] == 'Xenium']
pretrain_samples = ['TENX126', 'TENX125', 'TENX124', 'TENX123', 'TENX122', 'TENX121', 'TENX119', 'TENX118',
                    'TENX110', 'TENX111', 'TENX139', 'TENX96', 'TENX99', 'TENX138', 'TENX95', 'TENX140', 'TENX98', 'TENX97']
sample = pretrain_samples

pretrain_dataset = dataset_create_split(dir_path, sample, args, valid_prop=0., test_prop=0.1)

print(len(pretrain_samples))

pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=1, shuffle=True)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

# human protein-coding genes as shared gene idx
args.gene_total_num = 23258

# image emb dim of foundation encoders
if args.image_model == 'hoptimus':
    args.image_emb_dim = 1536
elif args.image_model == 'gigapath':
    args.image_emb_dim = 1536
elif args.image_model == 'uni':
    args.image_emb_dim = 1024

if args.gene_encoder_pretrained:
    gene_embeddings = pickle.load(open('../data/gene_embeddings_scgpt_human_protein.pkl', 'rb'))
else:
    gene_embeddings = torch.zeros((args.gene_total_num, args.gene_emb_dim), dtype=torch.float)
gene_embeddings = gene_embeddings.to(device)

model = parse_pretrain_method(args, gene_embeddings, device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_pretrain, weight_decay=args.wd_pretrain)

sizes, times, mems = [], [], []

model.train()
end_time = time.time()
for i, dataset in enumerate(pretrain_dataloader):
    data = dataset[0]
    x1, x2 = data.x.to(device), data.y.to(device)
    edge_index = data.edge_index.to(device)
    gene_idx = data.gene_idx.to(device)
    train_idx = data.split_idx['train_idx'].to(device)

    start_time = end_time
    loss = model.loss(x1, x2, gene_idx, train_idx, edge_index)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    end_time = time.time()
    memory_cost = get_gpu_memory_map(args.device)
    training_time = end_time - start_time

    print(f'{i}, sample size: {x1.shape[0]}, train time {training_time:.4f}, train memory {memory_cost:.1f}')
    sizes += [x1.shape[0]]
    times += [training_time]
    mems += [memory_cost]
    torch.cuda.empty_cache()

# model.eval()
# end_time = time.time()
# for i, dataset in enumerate(pretrain_dataloader):
#     data = dataset[0]
#     x1, x2 = data.x.to(device), data.y.to(device)
#     edge_index = data.edge_index.to(device)
#     gene_idx = data.gene_idx.to(device)
#     train_idx = data.split_idx['train_idx'].to(device)
#
#     start_time = end_time
#     outputs = model.encoder1(x1, edge_index)
#     end_time = time.time()
#     memory_cost = get_gpu_memory_map(args.device)
#     inference_time = end_time - start_time
#
#     print(f'{i}, sample size: {x1.shape[0]}, prediction time {inference_time:.4f}, prediction memory {memory_cost:.1f}')
#     sizes += [x1.shape[0]]
#     times += [inference_time]
#     mems += [memory_cost]
#     torch.cuda.empty_cache()

print(sizes)
print(times)
print(mems)
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
from utils import dataset_create
from parse import parse_pretrain_method, parser_add_main_args
import os
import warnings
from tqdm import tqdm

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
meta_info = meta_info[meta_info['tech']!='Xenium']

if args.domain_protocol == 'organ':
    # from frequent organs to rare organs
    pretrain_organs = ['Spinal cord', 'Brain', 'Breast', 'Bowel', 'Skin']
    pretrain_samples = meta_info[meta_info['organ'].isin(pretrain_organs)]['sample'].tolist()
    pretrain_samples = pretrain_samples[:-100] # hold out samples for testing
elif args.domain_protocol == 'mouse2human':
    # from mouse to human
    pretrain_samples = meta_info[meta_info['species']=='Mus musculus']['sample'].tolist()
elif args.domain_protocol == 'human2mouse':
    # from mouse to human
    pretrain_samples = meta_info[meta_info['species']=='Homo sapiens']['sample'].tolist()
elif args.domain_protocol == 'all':
    # all slides
    pretrain_samples = meta_info['sample'].tolist()
elif args.domain_protocol == 'all_small':
    # all slides
    pretrain_samples = meta_info['sample'].tolist()
else:
    raise NotImplementedError
print(len(pretrain_samples), len(meta_info))

pretrain_model_path = f'../model_checkpoints/{args.method}_pretrain_visium_{args.domain_protocol}.pth'

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

pretrain_dataset = dataset_create(dir_path, pretrain_samples, args)
pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=1, shuffle=True)

model_pretrain = parse_pretrain_method(args, gene_embeddings, device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_pretrain.parameters()), lr=args.lr_pretrain, weight_decay=args.wd_pretrain)

for epoch in range(args.pretrain_epochs):
    train_loss = run_pretrain(model_pretrain, pretrain_dataloader, optimizer, device, args.accumulate_steps)

    print(f'Epoch [{epoch + 1}/{args.pretrain_epochs}], Pretrain Loss: {train_loss:.4f}')

if args.save_model:
    torch.save(model_pretrain.state_dict(), pretrain_model_path)
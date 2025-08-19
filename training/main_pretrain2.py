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
from utils import dataset_create, dataset_create_split
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

if args.domain_protocol == 'sample':
    dir_path = '/data/wuqitian/hest_data_xenium_protein_preprocess'
    meta_info = pd.read_csv("../../data/meta_info_xenium.csv")
    meta_info = meta_info[meta_info['tech'] == 'Xenium']
    pretrain_samples = ['TENX126', 'TENX122', 'TENX121', 'TENX119', 'TENX118']
    pretrain_dataset = dataset_create_split(dir_path, pretrain_samples, args, valid_prop=0., test_prop=0.1)
    pretrain_model_path1 = f'../model_checkpoints/{args.method}_pretrain_visium_all.pth'

elif args.domain_protocol == 'sample+':
    dir_path = '/data/wuqitian/lung_preprocess'
    meta_info = pd.read_csv("../../data/meta_info_lung.csv")
    pretrain_samples = meta_info['sample'].tolist()
    pretrain_dataset = dataset_create(dir_path, pretrain_samples, args)

    dir_path2 = '/data/wuqitian/hest_data_xenium_protein_preprocess'
    meta_info2 = pd.read_csv("../../data/meta_info_xenium.csv")
    meta_info2 = meta_info2[meta_info2['organ'] != 'Lung']
    meta_info2 = meta_info2[~meta_info2['sample'].isin(
        ['TENX125', 'TENX124', 'TENX110', 'TENX111', 'TENX139', 'TENX96', 'TENX99', 'TENX138', 'TENX95', 'TENX140',
         'TENX98', 'TENX97', 'TENX126', 'TENX122', 'TENX121', 'TENX119', 'TENX118'])]
    pretrain_samples2 = meta_info2['sample'].tolist()
    pretrain_dataset2 = dataset_create(dir_path2, pretrain_samples2, args)
    pretrain_dataset.merge(pretrain_dataset2)
    pretrain_model_path1 = f'../model_checkpoints/{args.method}_pretrain_visium_all.pth'

elif args.domain_protocol == 'sample+_small':
    dir_path = '/data/wuqitian/lung_preprocess'
    meta_info = pd.read_csv("../../data/meta_info_lung.csv")
    pretrain_samples = meta_info[meta_info['affect'] == 'Unaffected']['sample'].tolist()[:-2]
    pretrain_samples += meta_info[meta_info['affect'] == 'Less Affected']['sample'].tolist()[:-2]
    pretrain_samples += meta_info[meta_info['affect'] == 'More Affected']['sample'].tolist()[:-2]
    pretrain_dataset = dataset_create(dir_path, pretrain_samples, args)

    dir_path2 = '/data/wuqitian/hest_data_xenium_protein_preprocess'
    meta_info2 = pd.read_csv("../../data/meta_info_xenium.csv")
    meta_info2 = meta_info2[meta_info2['organ'] != 'Lung']
    meta_info2 = meta_info2[~meta_info2['sample'].isin(
        ['TENX126', 'TENX125', 'TENX124', 'TENX123', 'TENX122', 'TENX121', 'TENX119', 'TENX118'])]
    pretrain_samples2 = meta_info2['sample'].tolist()
    pretrain_dataset2 = dataset_create(dir_path2, pretrain_samples2, args)
    pretrain_dataset.merge(pretrain_dataset2)
    pretrain_model_path1 = f'../model_checkpoints/{args.method}_pretrain_visium_all_small.pth'

elif args.domain_protocol == 'lung':
    dir_path = '/data/wuqitian/lung_preprocess'
    meta_info = pd.read_csv("../../data/meta_info_lung.csv")
    pretrain_samples = meta_info[meta_info['affect']=='Unaffected']['sample'].tolist()[:-2]
    pretrain_samples += meta_info[meta_info['affect'] == 'Less Affected']['sample'].tolist()[:-2]
    pretrain_samples += meta_info[meta_info['affect'] == 'More Affected']['sample'].tolist()[:-2]
    pretrain_dataset = dataset_create(dir_path, pretrain_samples, args)
    pretrain_model_path1 = f'../model_checkpoints/{args.method}_pretrain_visium_all.pth'

elif args.domain_protocol == 'lung+':
    dir_path = '/data/wuqitian/lung_preprocess'
    meta_info = pd.read_csv("../../data/meta_info_lung.csv")
    pretrain_samples = meta_info[meta_info['affect']=='Unaffected']['sample'].tolist()[:-2]
    pretrain_samples += meta_info[meta_info['affect'] == 'Less Affected']['sample'].tolist()[:-2]
    pretrain_samples += meta_info[meta_info['affect'] == 'More Affected']['sample'].tolist()[:-2]
    pretrain_dataset = dataset_create(dir_path, pretrain_samples, args)

    dir_path2 = '/data/wuqitian/hest_data_xenium_protein_preprocess'
    meta_info2 = pd.read_csv("../../data/meta_info_xenium.csv")
    meta_info2 = meta_info2[meta_info2['organ'] != 'Lung']
    meta_info2 = meta_info2[~meta_info2['sample'].isin(
        ['TENX125', 'TENX124', 'TENX110', 'TENX111', 'TENX139', 'TENX96', 'TENX99', 'TENX138', 'TENX95', 'TENX140', 'TENX98', 'TENX97'])]
    pretrain_samples2 = meta_info2['sample'].tolist()
    pretrain_dataset2 = dataset_create(dir_path2, pretrain_samples2, args)
    pretrain_dataset.merge(pretrain_dataset2)
    pretrain_model_path1 = f'../model_checkpoints/{args.method}_pretrain_visium_all.pth'

else:
    raise NotImplementedError
print(len(pretrain_samples))

pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=1, shuffle=True)

pretrain_model_path2 = f'../model_checkpoints/{args.method}_pretrain_xenium_{args.domain_protocol}.pth'

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

model_pretrain = parse_pretrain_method(args, gene_embeddings, device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_pretrain.parameters()), lr=args.lr_pretrain, weight_decay=args.wd_pretrain)

# pretrained_state_dict = torch.load(pretrain_model_path1)
# pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k.startswith("gene_encoder.") or k.startswith("encoder1.")}
# model_state_dict = model_pretrain.state_dict()
# model_dict = {k: v for k, v in model_state_dict.items() if k.startswith("gene_encoder.") or k.startswith("encoder1.")}
# for k, v in pretrained_dict.items():
#     assert (k in model_dict)
#     assert (v.size() == model_dict[k].size())
# model_state_dict.update(pretrained_dict)
# model_pretrain.load_state_dict(model_state_dict)

pretrained_state_dict = torch.load(pretrain_model_path1)
model_pretrain.load_state_dict(pretrained_state_dict)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_pretrain.parameters()), lr=args.lr_evaluation,
                       weight_decay=args.wd_evaluation)

for epoch in range(args.pretrain_epochs):
    train_loss = run_pretrain(model_pretrain, pretrain_dataloader, optimizer, device, args.accumulate_steps)

    print(f'Epoch [{epoch + 1}/{args.pretrain_epochs}], Pretrain Loss: {train_loss:.4f}')

    if args.save_model:
        torch.save(model_pretrain.state_dict(), pretrain_model_path2)
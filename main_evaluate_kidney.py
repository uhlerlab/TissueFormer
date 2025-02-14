import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import random
import argparse

from run_finetune import run_update, run_train, run_test, evaluate
from utils import dataset_create, dataset_create_split, k_shot_split
from parse import parse_regression_method, parser_add_main_args, parse_classification_method
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

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

dir_path = '/data/wuqitian/kidney_preprocess'

if args.domain_protocol == 'kidney':
    pretrain_model_path = f'../model_checkpoints/ours_pretrain_xenium_kidney.pth'
    # pretrain_model_path = f'../model_checkpoints/ours_pretrain_xenium_wo_kidney.pth'
    evaluation_model_path = f'../model_checkpoints/{args.method}_evaluate_xenium_kidney.pth'
    train_samples = ['ctrl_02', 'ptD', 'ptE', 'ptF']

    test_samples = ['ctrl_01', 'ptA', 'ptB', 'ptC']
else:
    raise NotImplementedError

args.gene_total_num = 470
if args.evaluate_task == 'macrophage_classification':
    args.cell_type_num = 23
elif args.evaluate_task == 'macrophage_identification':
    args.cell_type_num = 1

# image emb dim of foundation encoders
if args.image_model == 'hoptimus':
    args.image_emb_dim = 1536

if args.evaluate_task == 'gene_regression':
    model_eval = parse_regression_method(args, device)
elif args.evaluate_task in ['macrophage_classification', 'macrophage_identification']:
    model_eval = parse_classification_method(args, device)
else:
    raise NotImplementedError

# if args.method in ['ours', 'ours-MLP', 'ours-KNN']:
#     pretrained_state_dict = torch.load(pretrain_model_path)
#     encoder1_pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k.startswith("encoder1.")}
#     model_state_dict = model_eval.state_dict()
#     encoder1_model_dict = {k: v for k, v in model_state_dict.items() if k.startswith("encoder1.")}
#     for k, v in encoder1_pretrained_dict.items():
#         assert (k in encoder1_model_dict)
#         assert (v.size() == encoder1_model_dict[k].size())
#     model_state_dict.update(encoder1_pretrained_dict)
#     model_eval.load_state_dict(model_state_dict)
# if args.method == 'ours-MLP':
#     for param in model_eval.encoder1.parameters():
#         param.requires_grad = False
if args.method not in ['ours', 'ours-KNN', 'hoptimus-KNN', 'mean-pooling']:
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_eval.parameters()), lr=args.lr_evaluation,
                       weight_decay=args.wd_evaluation)

if args.evaluate_task == 'gene_regression':
    criterion = nn.MSELoss()
elif args.evaluate_task == 'macrophage_classification':
    criterion = nn.NLLLoss()
elif args.evaluate_task == 'macrophage_identification':
    criterion = nn.BCEWithLogitsLoss()

train_datasets = dataset_create(dir_path, train_samples, args, data_loader='kidney')
train_dataloader = DataLoader(train_datasets, batch_size=1, shuffle=True)

test_datasets = dataset_create(dir_path, test_samples, args, data_loader='kidney')
test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False)

if args.method in ['ours', 'ours-KNN', 'hoptimus-KNN', 'mean-pooling']:
    run_update(model_eval, train_dataloader, device, use_gene_idx=False)
else:
    for epoch in range(args.evaluation_epochs):
        train_loss = run_train(model_eval, train_dataloader, criterion, optimizer, device, args, use_gene_idx=False)
        test_score = run_test(model_eval, test_dataloader, device, args, use_gene_idx=False)

        print(f'Epoch [{epoch + 1}/{args.evaluation_epochs}], Train Loss: {train_loss:.4f}')
        print(f'Test Score: {test_score:.4f}')

        torch.save(model_eval.state_dict(), evaluation_model_path)

test_scores = evaluate(model_eval, test_dataloader, device, args, use_gene_idx=False)
formatted_scores = ', '.join([f'{score:.4f}' for score in test_scores])
print(f'Test Score: {formatted_scores}')

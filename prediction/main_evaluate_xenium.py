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
from parse import parse_regression_method, parser_add_main_args
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

if args.domain_protocol == 'sample':
    dir_path = '/data/wuqitian/hest_data_xenium_protein_preprocess'
    meta_info = pd.read_csv("../../data/meta_info_xenium.csv")
    samples = ['TENX126', 'TENX123', 'TENX124', 'TENX121', 'TENX119', 'TENX118']
    filter_genes = True
elif args.domain_protocol == 'bone':
    dir_path = '/data/wuqitian/hest_data_xenium_protein_preprocess'
    meta_info = pd.read_csv("../../data/meta_info_xenium.csv")
    samples = ['TENX137', 'TENX136', 'TENX135']
    filter_genes = False
elif args.domain_protocol == 'lung':
    dir_path = '/data/wuqitian/lung_preprocess'
    meta_info = pd.read_csv("../../data/meta_info_lung.csv")
    train_samples = ['VUHD095', 'THD0011', 'VUHD116A', 'VUHD116B', 'VUHD069', 'TILD117LA', 'VUILD110LA', 'VUILD96LA',
                     'VUILD102LA',
                     'VUILD48LA1', 'VUILD78MA', 'VUILD115MA', 'VUILD104MA1', 'VUILD105MA1', 'VUILD96MA', 'VUILD105MA2',
                     'VUILD107MA',
                     'TILD175MA', 'VUILD104MA2', 'VUILD106MA']
    test_samples = ['VUILD48LA2', 'THD0008', 'VUHD113', 'VUILD91MA', 'VUILD78LA', 'VUILD102MA']
    test_samples += ['TILD315MA', 'TILD299MA', 'VUILD58MA',
                    'TILD028LA', 'TILD111LA', 'TILD080LA', 'TILD130LA', 'VUILD49LA',
                    'VUHD090', 'VUHD038']
    filter_genes = True
else:
    raise NotImplementedError

model_version = 'xenium' # or visium or large
if model_version == 'xenium':
    pretrain_model_path = f'../model_checkpoints/ours_pretrain_xenium_sample+_small.pth'
elif model_version == 'visium':
    pretrain_model_path = f'../model_checkpoints/ours_pretrain_visium_all_small.pth'
elif model_version == 'large':
    pretrain_model_path = f'../model_checkpoints/ours_pretrain_xenium_sample+.pth'


# image emb dim of foundation encoders
if args.image_model == 'hoptimus':
    args.image_emb_dim = 1536
elif args.image_model == 'gigapath':
    args.image_emb_dim = 1536
elif args.image_model == 'uni':
    args.image_emb_dim = 1024
elif args.image_model == 'pca':
    args.image_emb_dim = 100

criterion = nn.MSELoss()

if args.domain_protocol in ['sample', 'bone']:
    for i, sample in enumerate(samples):
        test_sample = sample
        train_samples = samples[:i] + samples[i+1:]

        train_datasets = dataset_create(dir_path, train_samples, args, filter_genes=filter_genes)
        train_dataloader = DataLoader(train_datasets, batch_size=1, shuffle=True)

        test_datasets = dataset_create(dir_path, test_sample, args, filter_genes=filter_genes)
        test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=True)

        # datasets = dataset_create_split(dir_path, sample, args, valid_prop=0., test_prop=0.5, split='spatial',
        #                                 filter_genes=False)
        # dataloader = DataLoader(datasets, batch_size=1, shuffle=False)
        args.gene_total_num = train_datasets[0]['y'].shape[1]
        print(args.gene_total_num)
        model_eval = parse_regression_method(args, device)
        if args.method in ['ours', 'ours-MLP', 'ours-KNN']:
            pretrained_state_dict = torch.load(pretrain_model_path)
            encoder1_pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k.startswith("encoder1.")}
            model_state_dict = model_eval.state_dict()
            encoder1_model_dict = {k: v for k, v in model_state_dict.items() if k.startswith("encoder1.")}
            for k, v in encoder1_pretrained_dict.items():
                assert (k in encoder1_model_dict)
                assert (v.size() == encoder1_model_dict[k].size())
            model_state_dict.update(encoder1_pretrained_dict)
            model_eval.load_state_dict(model_state_dict)
        if args.method == 'ours-MLP':
            for param in model_eval.encoder1.parameters():
                param.requires_grad = False
        if args.method not in ['ours', 'ours-KNN', 'hoptimus-KNN', 'mean-pooling']:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_eval.parameters()), lr=args.lr_evaluation,
                                   weight_decay=args.wd_evaluation)

        if args.method in ['ours', 'ours-KNN', 'hoptimus-KNN', 'mean-pooling']:
            run_update(model_eval, train_dataloader, device, use_gene_idx=False)
        else:
            for epoch in range(args.evaluation_epochs):
                train_loss = run_train(model_eval, train_dataloader, criterion, optimizer, device, args, use_gene_idx=False)
                test_score = run_test(model_eval, test_dataloader, device, args, use_gene_idx=False)

                print(f'Epoch [{epoch + 1}/{args.evaluation_epochs}], Train Loss: {train_loss:.4f}')
                print(f'Test Score: {test_score:.4f}')

        test_scores = evaluate(model_eval, test_dataloader, device, args, use_gene_idx=False)
        formatted_scores = ', '.join([f'{score:.4f}' for score in test_scores])
        print(f'Test Score: {formatted_scores}')

        result_path = f'/data/wuqitian/analysis_pred_data/gene_expression_prediction/{test_sample}'
        y_pred, y_true = evaluate(model_eval, test_dataloader, device, args, use_gene_idx=False, output_result=True)
        y_pred, y_true = y_pred.cpu().numpy(), y_true.cpu().numpy()
        if model_version == 'large':
            np.save(result_path + f'_{args.method}_large', y_pred)
        elif model_version == 'visium':
            np.save(result_path + f'_{args.method}_visium', y_pred)
        else:
            np.save(result_path + f'_{args.method}', y_pred)
        np.save(result_path + '_true', y_true)

else: # lung
    # train_datasets = dataset_create(dir_path, train_samples, args, filter_genes=filter_genes)
    # train_dataloader = DataLoader(train_datasets, batch_size=1, shuffle=True)

    for i, test_sample in enumerate(test_samples):

        # test_datasets = dataset_create(dir_path, test_sample, args, filter_genes=filter_genes)
        # test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False)

        # train_datasets = dataset_create_split(dir_path, test_sample, args, split='random', valid_prop=0., test_prop=0.5)
        # train_dataloader = DataLoader(train_datasets, batch_size=1, shuffle=False)
        # test_dataloader = train_dataloader

        train_datasets = dataset_create_split(dir_path, test_sample, args, split='region', data_loader='lung', split_with_region=True)
        train_dataloader = DataLoader(train_datasets, batch_size=1, shuffle=False)
        test_dataloader = train_dataloader

        args.gene_total_num = train_datasets[0]['y'].shape[1]
        print(args.gene_total_num)
        model_eval = parse_regression_method(args, device)
        if args.method in ['ours', 'ours-MLP', 'ours-KNN']:
            pretrained_state_dict = torch.load(pretrain_model_path)
            encoder1_pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k.startswith("encoder1.")}
            model_state_dict = model_eval.state_dict()
            encoder1_model_dict = {k: v for k, v in model_state_dict.items() if k.startswith("encoder1.")}
            for k, v in encoder1_pretrained_dict.items():
                assert (k in encoder1_model_dict)
                assert (v.size() == encoder1_model_dict[k].size())
            model_state_dict.update(encoder1_pretrained_dict)
            model_eval.load_state_dict(model_state_dict)
        if args.method == 'ours-MLP':
            for param in model_eval.encoder1.parameters():
                param.requires_grad = False
        if args.method not in ['ours', 'ours-KNN', 'hoptimus-KNN', 'mean-pooling']:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_eval.parameters()), lr=args.lr_evaluation,
                                   weight_decay=args.wd_evaluation)

        if args.method in ['ours', 'ours-KNN', 'hoptimus-KNN', 'mean-pooling']:
            run_update(model_eval, train_dataloader, device, use_gene_idx=False)
        else:
            for epoch in range(args.evaluation_epochs):
                train_loss = run_train(model_eval, train_dataloader, criterion, optimizer, device, args, use_gene_idx=False)
                test_score = run_test(model_eval, test_dataloader, device, args, use_gene_idx=False)

                print(f'Epoch [{epoch + 1}/{args.evaluation_epochs}], Train Loss: {train_loss:.4f}')
                print(f'Test Score: {test_score:.4f}')

        test_scores = evaluate(model_eval, test_dataloader, device, args, use_gene_idx=False)
        formatted_scores = ', '.join([f'{score:.4f}' for score in test_scores])
        print(f'Test Score: {formatted_scores}')

        result_path = f'/data/wuqitian/analysis_pred_data/gene_expression_prediction/{test_sample}'
        y_pred, y_true = evaluate(model_eval, test_dataloader, device, args, use_gene_idx=False, output_result=True)
        y_pred, y_true = y_pred.cpu().numpy(), y_true.cpu().numpy()
        np.save(result_path + f'_{args.method}_region', y_pred)
        np.save(result_path + '_true', y_true)
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import random
import argparse

from run_pretrain import run_pretrain
from run_supervise import run_train, run_valid, run_test, evaluate
from utils import dataset_create_pretrain, dataset_create_supervise, k_fold_split
from parse import parse_method, parser_add_main_args
from logger import Logger

import os
import warnings

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

dir_path = '/data/wuqitian/hest_data_xenium'
# samples = ['NCBI859', 'TENX119', 'TENX143', 'TENX121', 'NCBI879', 'TENX96', 'TENX147', 'TENX125', 'TENX158',
#            'NCBI864', 'TENX95', 'TENX123', 'TENX98', 'TENX94', 'TENX115', 'NCBI858', 'NCBI873', 'TENX111',
#            'TENX141', 'NCBI783', 'TENX126', 'TENX139', 'NCBI876', 'TENX122', 'TENX120', 'TENX114', 'TENX138',
#            'TENX149', 'TENX118', 'TENX105', 'TENX134', 'NCBI883', 'TENX99', 'NCBI867', 'TENX132', 'TENX116',
#            'NCBI884', 'NCBI881', 'NCBI875', 'NCBI860', 'NCBI865', 'TENX133', 'NCBI857', 'NCBI870', 'TENX106',
#            'TENX148', 'TENX157', 'NCBI856', 'NCBI784', 'NCBI880', 'TENX124', 'TENX140', 'NCBI861', 'NCBI785',
#            'NCBI866', 'TENX117', 'TENX142', 'TENX97', 'NCBI882']
#
# pretrain_samples = ['NCBI859', 'TENX119', 'TENX143', 'TENX121', 'NCBI879', 'TENX96', 'TENX147', 'TENX125', 'TENX158',
#            'NCBI864', 'TENX95', 'TENX123', 'TENX98', 'TENX94', 'TENX115', 'NCBI858', 'NCBI873', 'TENX111',
#            'TENX141', 'NCBI783', 'TENX126', 'TENX139', 'NCBI876', 'TENX122', 'TENX120', 'TENX114', 'TENX138',
#            'TENX149', 'TENX118', 'TENX134', 'NCBI883', 'TENX99', 'NCBI867', 'TENX132', 'TENX116',
#            'NCBI884', 'NCBI881', 'NCBI875', 'NCBI860', 'NCBI865', 'TENX133', 'NCBI857', 'NCBI870',
#            'TENX148', 'TENX157', 'NCBI856', 'NCBI784', 'NCBI880', 'TENX124', 'TENX140', 'NCBI861', 'NCBI785',
#            'NCBI866', 'TENX117', 'TENX142', 'TENX97', 'NCBI882']

meta_info = pd.read_csv("../data/meta_info_xenium.csv")

pretrain_samples = meta_info[meta_info['organ']!='Kidney']['sample'].tolist()

evaluation_samples = meta_info[meta_info['organ']=='Kidney']['sample'].tolist()
train_samples = [evaluation_samples[0]]
test_samples = [evaluation_samples[1]]

# train_samples = ['TENX105', 'TENX106']
# test_samples = None

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

pretrain_model_path = f'../model_checkpoints/transformer_pretrain_xenium_-kidney.pth'
tmp_model_path = f'../tmp_checkpoints/{args.method}_{args.mode}_xenium_-kidney_{args.split_protocol}.pth'
result_path = f'../result/{args.method}_{args.mode}_xenium_-kidney_{args.split_protocol}.csv'

if args.mode == 'pretrain':
    pretrain_dataset = dataset_create_pretrain(dir_path, pretrain_samples)
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=1, shuffle=True)

    model_pretrain = parse_method(args, device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_pretrain.parameters()), lr=args.lr_pretrain, weight_decay=args.wd_pretrain)

    for epoch in range(args.pretrain_epochs):
        train_loss = run_pretrain(model_pretrain, pretrain_dataloader, optimizer, device)

        print(f'Epoch [{epoch + 1}/{args.pretrain_epochs}], Pretrain Loss: {train_loss:.4f}')

    if args.save_model:
        torch.save(model_pretrain.state_dict(), pretrain_model_path)

else: # evaluation with linear predictor
    train_samples_list, test_samples_list = k_fold_split(evaluation_samples)
    num_splits = args.num_splits if args.split_protocol == 'in_sample' else len(train_samples_list)
    test_metrics_all = []
    for s in range(num_splits):
        if args.split_protocol == 'in_sample':
            train_samples, test_samples = evaluation_samples, None
        else:  # out-sample splitting
            train_samples, test_samples = train_samples_list[s], test_samples_list[s]
        print(train_samples, test_samples)

        train_dataset, valid_dataset, test_dataset = dataset_create_supervise(dir_path, train_samples, test_samples, args.valid_prop, args.test_prop)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        args.gene_panel_num = train_dataset[0].y.shape[1]
        model_eval = parse_method(args, device)

        if args.mode == 'evaluation': # load pretrained model parameters
            checkpoint = torch.load(pretrain_model_path)
            model_eval.encoder1.load_state_dict(checkpoint, strict=False)
            for param in model_eval.encoder1.parameters():
                param.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_eval.parameters()), lr=args.lr_evaluation,
                                   weight_decay=args.wd_evaluation)
        else:
            optimizer = optim.Adam(model_eval.parameters(), lr=args.lr_supervise,
                                   weight_decay=args.wd_supervise)
        criterion = nn.MSELoss()
        epoch_nums = args.evaluation_epochs if args.mode == 'evaluation' else args.supervise_epochs

        valid_min_mse = 100.
        for run in range(args.runs):
            if args.mode == 'evaluation':
                model_eval.fc_out.reset_parameters()
            else:
                model_eval.reset_parameters()
            valid_min_mse = 100.
            for epoch in range(epoch_nums):
                train_loss = run_train(model_eval, train_dataloader, criterion, optimizer, device)
                valid_loss, valid_mse = run_valid(model_eval, valid_dataloader, criterion, device)
                test_loss, test_mse = run_test(model_eval, test_dataloader, criterion, device)

                print(f'Epoch [{epoch + 1}/{epoch_nums}], Train Loss: {train_loss:.4f}, ' +
                      f'Valid Loss: {valid_loss:.4f}, Test Loss: {test_loss:.4f}, ' +
                      f'Valid MSE: {valid_mse:.4f}, Test MSE: {test_mse:.4f}')

                # logger.add_result(run, valid_metrics, test_metrics)
                if valid_mse <= valid_min_mse:
                    valid_min_mse = valid_mse
                    test_best_mse = test_mse
                    torch.save(model_eval.state_dict(), tmp_model_path)

            print(f'Best Valid MSE: {valid_min_mse:.4f}, Final Test MSE: {test_best_mse:.4f}')
            checkpoint = torch.load(tmp_model_path)
            model_eval.load_state_dict(checkpoint)
            test_metrics = evaluate(model_eval, test_dataloader, device)
            test_metrics_all += [test_metrics]

            print_info = f'Split {s+1} Run {run+1} Final Test Results \n'
            for i in range(len(args.metrics)):
                print_info += f'{args.metrics[i]}: {test_metrics[i]:.4} \n'
            print(print_info)

    test_metrics_all = np.array(test_metrics_all)
    print_info = 'All Final Test \n'
    for i in range(len(args.metrics)):
        print_info += f'{args.metrics[i]}: {test_metrics_all[:, i].mean():.4f} ± {test_metrics_all[:, i].std():.4f} \n'
    print(print_info)

    results = {f'test {args.metrics[i]}': test_metrics_all[:, i].tolist() for i in range(len(args.metrics))}
    results = pd.DataFrame(results)
    results.to_csv(result_path, index=False)
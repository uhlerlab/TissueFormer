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

dir_path = '/data/wuqitian/hest_data_visium_protein_preprocess'
meta_info = pd.read_csv("../../data/meta_info_visium.csv")

pretrain_model_path = f'../model_checkpoints/ours_pretrain_visium_{args.domain_protocol}.pth'
evaluation_model_path = f'../model_checkpoints/{args.method}_evaluate_visium_{args.domain_protocol}.pth'
result_path = f'../result/{args.method}_visium_{args.domain_protocol}.csv'

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
elif args.image_model == 'pca':
    args.image_emb_dim = 100

model_eval = parse_regression_method(args, device)
if args.method in ['ours', 'ours-MLP']:
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
if args.method not in ['ours']:
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_eval.parameters()), lr=args.lr_evaluation,
                       weight_decay=args.wd_evaluation)
criterion = nn.MSELoss()


if args.domain_protocol == 'organ':
    # from frequent organs to rare organs, zero-shot prediction for evaluation
    pretrain_organs = ['Spinal cord', 'Brain', 'Breast', 'Bowel', 'Skin']  # top five organs
    pretrain_samples = meta_info[meta_info['organ'].isin(pretrain_organs)]['sample'].tolist()
    test_samples = meta_info[~meta_info['organ'].isin(pretrain_organs)]['sample'].tolist()
    valid_samples = pretrain_samples[-100:]
    train_samples = pretrain_samples[:20] # reference data for in-context learning
elif args.domain_protocol == 'mouse2human':
    # from mouse to human, zero-shot prediction for evaluation
    pretrain_samples = meta_info[meta_info['species'] == 'Mus musculus']['sample'].tolist()
    test_samples = meta_info[meta_info['species'] == 'Homo sapiens']['sample'].tolist()
    valid_samples = pretrain_samples[-100:]
    train_samples = pretrain_samples[:20] # reference data for in-context learning
else:
    raise NotImplementedError
print(len(meta_info), len(pretrain_samples), len(train_samples), len(valid_samples), len(test_samples))

# if not os.path.exists(evaluation_model_path): # need to train linear model
print("train linear model on all pretrained samples")
train_datasets = dataset_create(dir_path, train_samples, args)
train_dataloader = DataLoader(train_datasets, batch_size=1, shuffle=True)

valid_datasets = dataset_create(dir_path, valid_samples[:20], args)
valid_dataloader = DataLoader(valid_datasets, batch_size=1, shuffle=False)

test_datasets = dataset_create(dir_path, test_samples[:20], args)
test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False)

if args.method in ['ours']:
    run_update(model_eval, train_dataloader, device)
else:
    for epoch in range(args.evaluation_epochs):
        train_loss = run_train(model_eval, train_dataloader, criterion, optimizer, args, device)
        valid_score = run_test(model_eval, valid_dataloader, device, args)
        test_score = run_test(model_eval, test_dataloader, device, args)

        print(f'Epoch [{epoch + 1}/{args.evaluation_epochs}], Train Loss: {train_loss:.4f}')
        print(f'Valid Score: {valid_score:.4f}')
        print(f'Test Score: {test_score:.4f}')

    torch.save(model_eval.state_dict(), evaluation_model_path)

valid_scores = evaluate(model_eval, valid_dataloader, device, args)
formatted_scores = ', '.join([f'{score:.4f}' for score in valid_scores])
print(f'Valid Score: {formatted_scores}')
test_scores = evaluate(model_eval, test_dataloader, device, args)
formatted_scores = ', '.join([f'{score:.4f}' for score in test_scores])
print(f'Test Score: {formatted_scores}')


# test on new domains
test_metrics_all = []
test_info = []
num_metrics = len(args.metrics)

print("testing on evaluate samples one by one")
evaluate_samples = valid_samples + test_samples
for s, evaluate_sample in enumerate(evaluate_samples):
    test_dataset = dataset_create(dir_path, evaluate_sample, args)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_metrics = evaluate(model_eval, test_dataloader, device, args)
    test_metrics_all += [test_metrics]
    print_info = f'Sample {s} '
    print_info += f'{args.metrics[-1]} of top {args.hvg_gene_tops[-1]}: {test_metrics[-1]:.4f}'
    if s % 100 == 0:
        print(print_info)
    test_info += [evaluate_sample]

test_metrics_all = np.array(test_metrics_all)
# print_info = 'All Final Test \n'
# for t in range(len(args.hvg_gene_tops)):
#     for i in range(num_metrics):
#         print_info += f'{args.metrics[i]} of top {args.hvg_gene_tops[t]} genes: {test_metrics_all[:, num_metrics * t + i].mean():.4f} ± {test_metrics_all[:, num_metrics * t + i].std():.4f} \n'
# print(print_info)

results = {}
for t in range(len(args.hvg_gene_tops)):
    for i in range(len(args.metrics)):
        results[f'{args.metrics[i]} of top {args.hvg_gene_tops[t]} genes'] = \
            test_metrics_all[:, num_metrics * t + i].tolist()
results['sample'] = test_info
results['organ'] = [meta_info[meta_info['sample']==s]['organ'].tolist()[0] for s in test_info]
results['species'] = [meta_info[meta_info['sample']==s]['species'].tolist()[0] for s in test_info]
results['tech'] = [meta_info[meta_info['sample']==s]['tech'].tolist()[0] for s in test_info]

results = pd.DataFrame(results)
results.to_csv(result_path, index=False)

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import random
import argparse

from metrics import calculate_score
from run_finetune import run_update, run_train, run_test, evaluate
from utils import dataset_create, dataset_create_split, k_shot_split, FocalLoss
from parse import parse_regression_method, parser_add_main_args, parse_classification_method, parse_dual_method
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

dir_path = '/data/wuqitian/lung_preprocess'
meta_info = pd.read_csv("../../data/meta_info_lung.csv")
he_annotate_data = pd.read_csv(open('/data/wuqitian/lung/HE_annotations/cells_partitioned_by_annotation.csv', 'r', encoding='utf-8'))
sample_id_map = {
 'VUHD095': 'VUHD095',
 'THD0011': 'THD0011',
 'VUHD116A': 'VUHD116A',
 'VUHD116B': 'VUHD116B',
 'VUHD069': 'VUHD069',
 'TILD117LA': 'TILD117LF',
 'VUILD110LA': 'VUILD110',
 'VUILD96LA': 'VUILD96LF',
 'VUILD102LA': 'VUILD102LF',
 'VUILD48LA1': 'VUILD48LF',
 'VUILD78MA': 'VUILD78MF',
 'VUILD115MA': 'VUILD115',
 'VUILD104MA1': 'VUILD104LF',
 'VUILD105MA1': 'VUILD105LF',
 'VUILD96MA': 'VUILD96MF',
 'VUILD105MA2': 'VUILD105MF',
 'VUILD107MA': 'VUILD107MF',
 'TILD175MA': 'TILD175',
 'VUILD104MA2': 'VUILD104MF',
 'VUILD106MA': 'VUILD106',
 'VUILD48LA2': 'VUILD48MF',
 'THD0008': 'THD0008',
 'VUHD113': 'VUHD113',
 'VUILD91MA': 'VUILD91MF',
 'VUILD78LA': 'VUILD78LF',
 'VUILD102MA': 'VUILD102MF'
}
sample_id_map_inv = {k:v for v, k in sample_id_map.items()}
he_maps = {i: an for i, an in enumerate(he_annotate_data['annotation_type'].unique().tolist())}

if args.domain_protocol == 'lung':
    pretrain_model_path = f'../model_checkpoints/ours_pretrain_xenium_lung+.pth'
    # pretrain_model_path = f'../model_checkpoints/ours_pretrain_visium_all.pth'
    evaluation_model_path = f'../model_checkpoints/{args.method}_evaluate_xenium_lung.pth'
    train_samples = meta_info[meta_info['affect'] == 'Unaffected']['sample'].tolist()[:-2]
    train_samples += meta_info[meta_info['affect'] == 'Less Affected']['sample'].tolist()[:-2]
    train_samples += meta_info[meta_info['affect'] == 'More Affected']['sample'].tolist()[:-2]

    test_samples = meta_info[~meta_info['sample'].isin(train_samples)]['sample'].tolist()
else:
    raise NotImplementedError

args.gene_total_num = 340
if args.evaluate_task == 'cell_type_classification':
    args.cell_type_num = 2
    args.gene_total_num = 23258
    train_samples = ['VUHD095', 'THD0011', 'VUHD116A', 'VUHD116B', 'VUHD069', 'TILD117LA', 'VUILD110LA', 'VUILD96LA', 'VUILD102LA',
     'VUILD48LA1', 'VUILD78MA', 'VUILD115MA', 'VUILD104MA1', 'VUILD105MA1', 'VUILD96MA', 'VUILD105MA2', 'VUILD107MA',
     'TILD175MA', 'VUILD104MA2', 'VUILD106MA']
    test_samples = ['VUILD48LA2', 'THD0008', 'VUHD113', 'VUILD91MA', 'VUILD78LA', 'VUILD102MA']
    test_samples += ['TILD315MA', 'TILD299MA', 'VUILD58MA',
                     'TILD028LA', 'TILD111LA', 'TILD080LA', 'TILD130LA', 'VUILD49LA',
                     'VUHD090', 'VUHD038']
elif args.evaluate_task == 'niche_classification':
    args.cell_type_num = 2
    args.gene_total_num = 23258
    train_samples = ['VUHD095', 'THD0011', 'VUHD116A', 'VUHD116B', 'VUHD069', 'TILD117LA', 'VUILD110LA', 'VUILD96LA',
                     'VUILD102LA',
                     'VUILD48LA1', 'VUILD78MA', 'VUILD115MA', 'VUILD104MA1', 'VUILD105MA1', 'VUILD96MA', 'VUILD105MA2',
                     'VUILD107MA',
                     'TILD175MA', 'VUILD104MA2', 'VUILD106MA']
    test_samples = ['VUILD48LA2', 'THD0008', 'VUHD113', 'VUILD91MA', 'VUILD78LA', 'VUILD102MA']
elif args.evaluate_task == 'region_time_prediction':
    args.cell_type_num = 1
    args.gene_total_num = 23258
    train_samples = ['VUHD095', 'THD0011', 'VUHD116A', 'VUHD116B', 'VUHD069', 'TILD117LA', 'VUILD110LA', 'VUILD96LA',
                     'VUILD102LA',
                     'VUILD48LA1', 'VUILD78MA', 'VUILD115MA', 'VUILD104MA1', 'VUILD105MA1', 'VUILD96MA', 'VUILD105MA2',
                     'VUILD107MA',
                     'TILD175MA', 'VUILD104MA2', 'VUILD106MA']
    test_samples = ['VUILD48LA2', 'THD0008', 'VUHD113', 'VUILD91MA', 'VUILD78LA', 'VUILD102MA']
    test_samples += ['TILD315MA', 'TILD299MA', 'VUILD58MA',
                     'TILD028LA', 'TILD111LA', 'TILD080LA', 'TILD130LA', 'VUILD49LA',
                     'VUHD090', 'VUHD038']
elif args.evaluate_task == 'he_annotation_classification':
    args.cell_type_num = 2
    if args.he_annotation_type == 'severe_fibrosis':
        args.he_annotation_idx = 12
        train_samples = ['VUILD106MA', 'VUILD107MA', 'VUILD104MA1']
        test_samples = ['VUILD48LA2']
    elif args.he_annotation_type == 'epithelial_detachment':
        args.he_annotation_idx = 9
        train_samples = ['VUILD104MA1', 'VUILD106MA', 'VUILD91MA', 'VUILD102MA']
        test_samples = ['TILD117LA']
    elif args.he_annotation_type == 'fibroblastic_focus':
        args.he_annotation_idx = 16
        train_samples = ['VUILD106MA', 'VUILD104MA1', 'VUILD115MA', 'VUILD105MA1', 'VUILD104MA2', 'VUILD107MA']
        test_samples = ['VUILD48LA2']
    elif args.he_annotation_type == 'giant_cell':
        args.he_annotation_idx = 22
        train_samples = ['VUILD115MA']
        test_samples = ['VUILD110LA']
    elif args.he_annotation_type == 'hyperplastic_aec':
        args.he_annotation_idx = 15
        train_samples = ['VUILD105MA1', 'VUILD107MA', 'VUILD106MA', 'VUILD115MA']
        test_samples = ['VUILD48LA1']
    elif args.he_annotation_type == 'large_airway':
        args.he_annotation_idx = 13
        train_samples = ['VUILD107MA', 'VUILD96MA', 'TILD175MA']
        test_samples = ['TILD117LA']
    elif args.he_annotation_type == 'granuloma':
        args.he_annotation_idx = 17
        train_samples = ['VUILD96MA', 'VUILD104MA1']
        test_samples = ['VUILD96LA']
    elif args.he_annotation_type == 'advanced_remodeling':
        args.he_annotation_idx = 6
        train_samples = ['VUILD102MA']
        test_samples = ['VUILD78LA']
    elif args.he_annotation_type == 'small_airway':
        args.he_annotation_idx = 1
        train_samples = ['TILD175MA', 'VUILD115MA', 'VUILD78MA']
        test_samples = ['VUHD116A', 'VUHD069', 'VUHD095']
    elif args.he_annotation_type == 'venule':
        args.he_annotation_idx = 4
        train_samples = ['VUILD105MA1', 'VUILD106MA', 'VUILD96MA', 'TILD175MA']
        test_samples = ['VUILD110LA']

# image emb dim of foundation encoders
if args.image_model == 'hoptimus':
    args.image_emb_dim = 1536
elif args.image_model == 'gigapath':
    args.image_emb_dim = 1536
elif args.image_model == 'uni':
    args.image_emb_dim  = 1024
elif args.image_model == 'pca':
    args.image_emb_dim = 100

if args.evaluate_task == 'gene_regression':
    model_eval = parse_regression_method(args, device)
elif args.evaluate_task in ['he_annotation_classification'] or args.method != 'ours-MLP':
    model_eval = parse_classification_method(args, device)
elif args.evaluate_task in ['niche_classification', 'cell_type_classification', 'region_time_prediction']:
    gene_embeddings = torch.zeros((args.gene_total_num, args.gene_emb_dim), dtype=torch.float)
    model_eval = parse_dual_method(args, gene_embeddings, device)

if args.evaluate_task in ['niche_classification', 'cell_type_classification', 'region_time_prediction']:
    if args.method in ['ours', 'ours-MLP']:
        pretrained_state_dict = torch.load(pretrain_model_path)
        pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k.startswith("encoder1.") or k.startswith("encoder2.") or k.startswith("gene_encoder.")}
        model_state_dict = model_eval.state_dict()
        model_dict = {k: v for k, v in model_state_dict.items() if k.startswith("encoder1.") or k.startswith("encoder2.") or k.startswith("gene_encoder.")}
        for k, v in pretrained_dict.items():
            assert (k in model_dict)
            assert (v.size() == model_dict[k].size())
        model_state_dict.update(pretrained_dict)
        model_eval.load_state_dict(model_state_dict)
    if args.method == 'ours-MLP':
        for param in model_eval.encoder1.parameters():
            param.requires_grad = False
        for param in model_eval.encoder2.parameters():
            param.requires_grad = False
        for param in model_eval.gene_encoder.parameters():
            param.requires_grad = False
else:
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

if args.evaluate_task == 'gene_regression':
    criterion = nn.MSELoss()
elif args.evaluate_task == 'region_time_prediction':
    criterion = nn.MSELoss()
elif args.evaluate_task in ['cell_type_classification']:
    # criterion = nn.NLLLoss()
    criterion = FocalLoss(alpha=0.25, gamma=2, device=device)
elif args.evaluate_task in ['niche_classification']:
    # criterion = nn.NLLLoss()
    criterion = FocalLoss(alpha=0.25, gamma=2, device=device)
elif args.evaluate_task == 'he_annotation_classification':
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    criterion = FocalLoss(alpha=0.25, gamma=2, device=device)

train_datasets = dataset_create(dir_path, train_samples, args, data_loader='lung')
train_dataloader = DataLoader(train_datasets, batch_size=1, shuffle=True)

test_datasets = dataset_create(dir_path, test_samples, args, data_loader='lung', use_pred_gene=args.use_pred_gene)
test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False)

if args.method in ['ours']:
    run_update(model_eval, train_dataloader, device, use_gene_idx=True)
else:
    for epoch in range(args.evaluation_epochs):
        train_loss = run_train(model_eval, train_dataloader, criterion, optimizer, device, args, use_gene_idx=True)
        test_score = run_test(model_eval, test_dataloader, device, args, use_gene_idx=True)

        print(f'Epoch [{epoch + 1}/{args.evaluation_epochs}], Train Loss: {train_loss:.4f}')
        print(f'Test Score: {test_score:.4f}')

        torch.save(model_eval.state_dict(), evaluation_model_path)

test_scores = evaluate(model_eval, test_dataloader, device, args, use_gene_idx=True)
formatted_scores = ', '.join([f'{score:.4f}' for score in test_scores])
print(f'Test Score: {formatted_scores}')

if args.evaluate_task == 'he_annotation_classification':
    result_path = f'/data/wuqitian/analysis_pred_data/he_annotation_classification/{args.he_annotation_type}'
    y_pred, y_true = evaluate(model_eval, test_dataloader, device, args, output_result=True, use_gene_idx=True)
    y_pred, y_true = y_pred.cpu().numpy(), y_true.cpu().numpy()
    np.save(result_path+f'_{args.method}', y_pred)
    np.save(result_path+'_true', y_true)

    filename = f'/data/wuqitian/analysis_pred_data/he_annotation_classification/hyper_search/{args.he_annotation_type}.csv'
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f'{args.method} {args.lr_evaluation} {args.evaluation_epochs} {formatted_scores} \n')

elif args.evaluate_task == 'niche_classification':
    method = args.method
    method += '-pred-gene' if args.use_pred_gene else ''
    method += '-no-image' if args.no_image_encoder else ''
    result_path = f'/data/wuqitian/analysis_pred_data/niche_classification/{args.niche_type}'
    y_pred, y_true = evaluate(model_eval, test_dataloader, device, args, output_result=True, use_gene_idx=True)
    y_pred, y_true = y_pred.cpu().numpy(), y_true.cpu().numpy()
    np.save(result_path+f'_{method}', y_pred)
    np.save(result_path+'_true', y_true)

    filename = f'/data/wuqitian/analysis_pred_data/niche_classification/hyper_search/{args.niche_type}.csv'
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f'{method} {args.lr_evaluation} {args.evaluation_epochs} {formatted_scores} \n')

elif args.evaluate_task == 'cell_type_classification':
    method = args.method
    method += '-pred-gene' if args.use_pred_gene else ''
    method += '-no-image' if args.no_image_encoder else ''
    result_path = f'/data/wuqitian/analysis_pred_data/cell_type_classification/{args.cell_type.replace('/', '')}2'
    y_pred, y_true = evaluate(model_eval, test_dataloader, device, args, output_result=True, use_gene_idx=True)
    y_pred, y_true = y_pred.cpu().numpy(), y_true.cpu().numpy()
    np.save(result_path+f'_{method}', y_pred)
    np.save(result_path+'_true', y_true)

    filename = f'/data/wuqitian/analysis_pred_data/cell_type_classification/hyper_search/{args.cell_type.replace('/', '')}.csv'
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f'{method} {args.lr_evaluation} {args.evaluation_epochs} {formatted_scores} \n')

elif args.evaluate_task == 'region_time_prediction':
    method = args.method
    method += '-pred-gene' if args.use_pred_gene else ''
    method += '-no-image' if args.no_image_encoder else ''
    result_path = f'/data/wuqitian/analysis_pred_data/region_time_prediction/lumen_rank2'
    y_pred, y_true = evaluate(model_eval, test_dataloader, device, args, output_result=True, use_gene_idx=True)
    y_pred, y_true = y_pred.cpu().numpy(), y_true.cpu().numpy()
    np.save(result_path+f'_{method}', y_pred)
    np.save(result_path+'_true', y_true)

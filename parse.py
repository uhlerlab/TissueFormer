from model import Model_Pretrain, Model_Predict
import models
import torch
import pickle

def parse_method(args, device):
    gene_embeddings_dict = pickle.load(open('../data/gene_embeddings_scgpt_human.pkl', 'rb'))
    gene_embeddings = torch.tensor(list(gene_embeddings_dict.values()), dtype=torch.float)
    gene_emb_dim = gene_embeddings.shape[1]
    gene_total_num = gene_embeddings.shape[0]
    image_emb_dim = 1536
    if args.method == 'transformer':
        encoder1 = models.Transformer(in_channels=image_emb_dim, hidden_channels=args.enc1_hidden_channels, num_layers_prop=args.enc1_num_layers_prop,
                                  num_layers_mlp=args.enc1_num_layers_mlp, num_attn_heads=args.enc1_num_attn_heads,
                                  dropout=args.enc1_dropout, use_bn=-args.enc1_no_bn, use_graph=-args.enc1_no_graph, use_residual=-args.enc1_no_residual).to(device)
        if args.mode == 'pretrain':
            # encoder2 = models.Transformer(in_channels=gene_emb_dim, hidden_channels=args.enc2_hidden_channels, num_layers_prop=args.enc2_num_layers_prop,
            #                           num_layers_mlp=args.enc2_num_layers_mlp, num_attn_heads=args.enc2_num_attn_heads,
            #                           dropout=args.enc2_dropout, use_bn=-args.enc2_no_bn, use_graph=-args.enc2_no_graph, use_residual=-args.enc2_no_residual).to(device)
            encoder2 = models.MLP(in_channels=gene_emb_dim, hidden_channels=args.enc2_hidden_channels, num_layers=args.enc2_num_layers_mlp,
                                  dropout=args.enc2_dropout).to(device)
            model = Model_Pretrain(encoder1, encoder2, gene_embeddings, reg_w=args.reg_weight, ge_trainable=args.gene_encoder_trainable, ge_pretrained=args.gene_encoder_pretrained).to(device)
        else: # evaluation or supervise
            model = Model_Predict(encoder1, hidden_channels=args.enc1_hidden_channels, out_channels=args.gene_panel_num).to(device)
    elif args.method == 'linear':
        encoder1 = models.Image_Pretrain()
        if args.mode in ['pretrain', 'supervise']:
            raise ValueError('Not Implemented')
        else: # evaluation
            model = Model_Predict(encoder1, hidden_channels=image_emb_dim, out_channels=args.gene_panel_num).to(device)
    else:
        raise ValueError('Invalid method')
    return model

def parser_add_main_args(parser):
    # method, dataset, protocol
    parser.add_argument('--method', type=str, default='transformer')
    parser.add_argument('--mode', type=str, default='pretrain', choices=['pretrain', 'evaluation', 'supervise'])
    parser.add_argument('--data_dir', type=str, default='/data/wuqitian/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--save_model_dir', type=str, default='../model_checkpoints/')
    parser.add_argument('--valid_prop', type=float, default=0.2)
    parser.add_argument('--test_prop', type=float, default=0.6) # only used for in-sample splitting
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--metrics', type=list, default=['MSE', 'Kendall', 'NDCG@20', 'NDCG@1'])
    parser.add_argument('--split_protocol', type=str, default='in_sample', choices=['in_sample', 'out_sample'])
    parser.add_argument('--num_splits', type=int, default=5) # only used for in-sample splitting

    # model architecture encoder 1
    parser.add_argument('--enc1_hidden_channels', type=int, default=128)
    parser.add_argument('--enc1_num_layers_prop', type=int, default=2)
    parser.add_argument('--enc1_num_layers_mlp', type=int, default=2)
    parser.add_argument('--enc1_num_attn_heads', type=int, default=1)
    parser.add_argument('--enc1_dropout', type=float, default=0.0)
    parser.add_argument('--enc1_no_bn', action='store_true')
    parser.add_argument('--enc1_no_graph', action='store_true')
    parser.add_argument('--enc1_no_residual', action='store_true')

    # model architecture encoder 2
    parser.add_argument('--enc2_hidden_channels', type=int, default=128)
    parser.add_argument('--enc2_num_layers_prop', type=int, default=0)
    parser.add_argument('--enc2_num_layers_mlp', type=int, default=1)
    parser.add_argument('--enc2_num_attn_heads', type=int, default=1)
    parser.add_argument('--enc2_dropout', type=float, default=0.0)
    parser.add_argument('--enc2_no_bn', action='store_true')
    parser.add_argument('--enc2_no_graph', action='store_true')
    parser.add_argument('--enc2_no_residual', action='store_true')

    # model pretraining
    parser.add_argument('--lr_pretrain', type=float, default=1e-4)
    parser.add_argument('--wd_pretrain', type=float, default=0.)
    parser.add_argument('--reg_weight', type=float, default=0.1)
    parser.add_argument('--gene_encoder_trainable', action='store_true')
    parser.add_argument('--gene_encoder_pretrained', action='store_true')
    parser.add_argument('--pretrain_epochs', type=int, default=200)

    # evaluation
    parser.add_argument('--lr_evaluation', type=float, default=1e-2)
    parser.add_argument('--wd_evaluation', type=float, default=1e-4)
    parser.add_argument('--evaluation_epochs', type=int, default=100)

    # supervise
    parser.add_argument('--lr_supervise', type=float, default=1e-3)
    parser.add_argument('--wd_supervise', type=float, default=0.)
    parser.add_argument('--supervise_epochs', type=int, default=200)

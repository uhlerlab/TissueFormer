from models.pretrain import Model_Pretrain
import models.regression as regression
import models.classification as classification
import models.dual as dual
import encoders

def parse_pretrain_method(args, gene_embeddings, device):
    if args.method == 'ours':
        encoder1 = encoders.Transformer(in_channels=args.image_emb_dim, hidden_channels=args.enc1_hidden_channels, num_layers_prop=args.enc1_num_layers_prop,
                                        num_layers_mlp=args.enc1_num_layers_mlp, num_attn_heads=args.enc1_num_attn_heads,
                                        dropout=args.enc1_dropout, use_bn=-args.enc1_no_bn, use_graph=-args.enc1_no_graph, use_residual=-args.enc1_no_residual).to(device)
        encoder2 = encoders.MLP(in_channels=args.gene_emb_dim, hidden_channels=args.enc2_hidden_channels, num_layers=args.enc2_num_layers_mlp,
                                dropout=args.enc2_dropout).to(device)
        model = Model_Pretrain(encoder1, encoder2, gene_embeddings, reg_w=args.reg_weight, ge_trainable=args.gene_encoder_trainable, ge_pretrained=args.gene_encoder_pretrained).to(device)
    else:
        raise ValueError('Invalid method')
    return model

def parse_regression_method(args, device):
    if args.method in 'ours':
        encoder1 = encoders.Transformer(in_channels=args.image_emb_dim, hidden_channels=args.enc1_hidden_channels, num_layers_prop=args.enc1_num_layers_prop,
                                        num_layers_mlp=args.enc1_num_layers_mlp, num_attn_heads=args.enc1_num_attn_heads,
                                        dropout=args.enc1_dropout, use_bn=-args.enc1_no_bn, use_graph=-args.enc1_no_graph, use_residual=-args.enc1_no_residual).to(device)
        model = regression.InContext_Predict(encoder1, hidden_channels=args.enc1_hidden_channels,
                                             out_channels=args.gene_total_num, batch_size=args.batch_size,
                                            num_neighbors=args.neighbor_num, device=device).to(device)
    elif args.method in 'ours-MLP':
        encoder1 = encoders.Transformer(in_channels=args.image_emb_dim, hidden_channels=args.enc1_hidden_channels,
                                        num_layers_prop=args.enc1_num_layers_prop,
                                        num_layers_mlp=args.enc1_num_layers_mlp, num_attn_heads=args.enc1_num_attn_heads,
                                        dropout=args.enc1_dropout, use_bn=-args.enc1_no_bn, use_graph=-args.enc1_no_graph,
                                        use_residual=-args.enc1_no_residual).to(device)
        model = regression.MLP_Predict(encoder1, hidden_channels=args.enc1_hidden_channels,
                                       out_channels=args.gene_total_num).to(device)
    elif args.method == 'hoptimus-MLP':
        encoder1 = encoders.Image_Encoder()
        model = regression.MLP_Predict(encoder1, hidden_channels=args.image_emb_dim, out_channels=args.gene_total_num).to(device)
    elif args.method == 'gigapath-MLP':
        assert (args.image_model=='gigapath')
        encoder1 = encoders.Image_Encoder()
        model = regression.MLP_Predict(encoder1, hidden_channels=args.image_emb_dim, out_channels=args.gene_total_num).to(device)
    elif args.method == 'uni-MLP':
        assert (args.image_model=='uni')
        encoder1 = encoders.Image_Encoder()
        model = regression.MLP_Predict(encoder1, hidden_channels=args.image_emb_dim, out_channels=args.gene_total_num).to(device)
    elif args.method == 'pca-MLP':
        assert (args.image_model == 'pca')
        encoder1 = encoders.Image_Encoder()
        model = regression.MLP_Predict(encoder1, hidden_channels=args.image_emb_dim, out_channels=args.gene_total_num).to(device)
    else:
        raise ValueError('Invalid method')
    return model

def parse_classification_method(args, device):
    if args.method in 'ours':
        encoder1 = encoders.Transformer(in_channels=args.image_emb_dim, hidden_channels=args.enc1_hidden_channels,
                                        num_layers_prop=args.enc1_num_layers_prop,
                                        num_layers_mlp=args.enc1_num_layers_mlp,
                                        num_attn_heads=args.enc1_num_attn_heads,
                                        dropout=args.enc1_dropout, use_bn=-args.enc1_no_bn,
                                        use_graph=-args.enc1_no_graph, use_residual=-args.enc1_no_residual).to(device)
        model = classification.InContext_Predict(encoder1, hidden_channels=args.enc1_hidden_channels,
                                             out_channels=args.cell_type_num, batch_size=args.batch_size,
                                            num_neighbors=args.neighbor_num, device=device).to(device)
    elif args.method in 'ours-MLP':
        encoder1 = encoders.Transformer(in_channels=args.image_emb_dim, hidden_channels=args.enc1_hidden_channels,
                                        num_layers_prop=args.enc1_num_layers_prop,
                                        num_layers_mlp=args.enc1_num_layers_mlp,
                                        num_attn_heads=args.enc1_num_attn_heads,
                                        dropout=args.enc1_dropout, use_bn=-args.enc1_no_bn,
                                        use_graph=-args.enc1_no_graph,
                                        use_residual=-args.enc1_no_residual).to(device)
        model = classification.MLP_Predict(encoder1, hidden_channels=args.enc1_hidden_channels,
                                       out_channels=args.cell_type_num).to(device)
    elif args.method == 'hoptimus-MLP':
        encoder1 = encoders.Image_Encoder()
        model = classification.MLP_Predict(encoder1, hidden_channels=args.image_emb_dim, out_channels=args.cell_type_num).to(device)
    elif args.method == 'gigapath-MLP':
        assert (args.image_model == 'gigapath')
        encoder1 = encoders.Image_Encoder()
        model = classification.MLP_Predict(encoder1, hidden_channels=args.image_emb_dim, out_channels=args.cell_type_num).to(device)
    elif args.method == 'uni-MLP':
        assert (args.image_model == 'uni')
        encoder1 = encoders.Image_Encoder()
        model = classification.MLP_Predict(encoder1, hidden_channels=args.image_emb_dim, out_channels=args.cell_type_num).to(device)
    else:
        raise ValueError('Invalid method')
    return model

def parse_dual_method(args, gene_embeddings, device):
    if args.method in 'ours-MLP':
        encoder1 = encoders.Transformer(in_channels=args.image_emb_dim, hidden_channels=args.enc1_hidden_channels,
                                        num_layers_prop=args.enc1_num_layers_prop,
                                        num_layers_mlp=args.enc1_num_layers_mlp,
                                        num_attn_heads=args.enc1_num_attn_heads,
                                        dropout=args.enc1_dropout, use_bn=-args.enc1_no_bn,
                                        use_graph=-args.enc1_no_graph, use_residual=-args.enc1_no_residual).to(device)
        encoder2 = encoders.MLP(in_channels=args.gene_emb_dim, hidden_channels=args.enc2_hidden_channels,
                                num_layers=args.enc2_num_layers_mlp,
                                dropout=args.enc2_dropout).to(device)
        model = dual.MLP_Predict(encoder1, encoder2, gene_embeddings, hidden_channels=args.enc1_hidden_channels, out_channels=args.cell_type_num,
                               ge_trainable=args.gene_encoder_trainable, ge_pretrained=args.gene_encoder_pretrained,
                                 no_gene_encoder=args.no_gene_encoder, no_image_encoder=args.no_image_encoder).to(device)
    else:
        raise ValueError('Invalid method')
    return model

def parser_add_main_args(parser):
    # method, dataset, protocol
    parser.add_argument('--method', type=str, default='ours')
    # parser.add_argument('--mode', type=str, default='pretrain', choices=['pretrain', 'evaluation', 'supervise'])
    parser.add_argument('--data_dir', type=str, default='/data/wuqitian/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--valid_prop', type=float, default=0.)
    parser.add_argument('--test_prop', type=float, default=0.) # only used for sample in Xenium
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--accumulate_steps', type=int, default=20)
    parser.add_argument('--metrics', nargs="+", type=str, default=['RMSE', 'Pearson'])
    parser.add_argument('--hvg_gene_tops', nargs="+", type=int, default=[50, 100, 200, 500, 1000, 2000, 4000])
    parser.add_argument('--hvg_gene_top', type=list, default=2000)
    parser.add_argument('--domain_protocol', type=str, default='organ')
    parser.add_argument('--evaluate_task', type=str, default='gene_regression',
                        choices=['gene_regression', 'cell_type_classification',
                                 'he_annotation_classification', 'niche_classification',
                                 'region_time_prediction'])
    parser.add_argument('--he_annotation_type', type=str, default='multinucleated_cell')
    parser.add_argument('--niche_type', type=str, default='T1')
    parser.add_argument('--cell_type', type=str, default='B-cell')
    parser.add_argument('--use_pred_gene', action='store_true')
    parser.add_argument('--no_gene_encoder', action='store_true')
    parser.add_argument('--no_image_encoder', action='store_true')

    # model architecture encoder 1
    parser.add_argument('--enc1_hidden_channels', type=int, default=1024)
    parser.add_argument('--enc1_num_layers_prop', type=int, default=2)
    parser.add_argument('--enc1_num_layers_mlp', type=int, default=2)
    parser.add_argument('--enc1_num_attn_heads', type=int, default=1)
    parser.add_argument('--enc1_dropout', type=float, default=0.0)
    parser.add_argument('--enc1_no_bn', action='store_true')
    parser.add_argument('--enc1_no_graph', action='store_true')
    parser.add_argument('--enc1_no_residual', action='store_true')

    # model architecture encoder 2
    parser.add_argument('--enc2_hidden_channels', type=int, default=1024)
    parser.add_argument('--enc2_num_layers_prop', type=int, default=0)
    parser.add_argument('--enc2_num_layers_mlp', type=int, default=1)
    parser.add_argument('--enc2_num_attn_heads', type=int, default=1)
    parser.add_argument('--enc2_dropout', type=float, default=0.0)
    parser.add_argument('--enc2_no_bn', action='store_true')
    parser.add_argument('--enc2_no_graph', action='store_true')
    parser.add_argument('--enc2_no_residual', action='store_true')

    # model training
    parser.add_argument('--lr_pretrain', type=float, default=1e-5)
    parser.add_argument('--wd_pretrain', type=float, default=0.)
    parser.add_argument('--reg_weight', type=float, default=0.5)
    parser.add_argument('--gene_emb_dim', type=int, default=256)
    parser.add_argument('--image_model', type=str, default='hoptimus', choices=['hoptimus', 'gigapath', 'uni', 'pca'])
    parser.add_argument('--gene_encoder_trainable', action='store_true')
    parser.add_argument('--gene_encoder_pretrained', action='store_true')
    parser.add_argument('--pretrain_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--neighbor_num', type=int, default=500)

    # evaluation
    parser.add_argument('--lr_evaluation', type=float, default=1e-4)
    parser.add_argument('--wd_evaluation', type=float, default=0.)
    parser.add_argument('--evaluation_epochs', type=int, default=100)

    # supervise
    parser.add_argument('--lr_supervise', type=float, default=1e-4)
    parser.add_argument('--wd_supervise', type=float, default=0.)
    parser.add_argument('--supervise_epochs', type=int, default=100)

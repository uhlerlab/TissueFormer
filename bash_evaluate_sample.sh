#
## xenium samples

python main_evaluate_xenium.py --domain_protocol sample --hvg_gene_tops 400 --method ours --gene_emb_dim 128 \
--enc1_hidden_channels 128 --enc2_hidden_channels 128 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--neighbor_num 1000 --batch_size 1000 --device 7

python main_evaluate_xenium.py --domain_protocol sample --hvg_gene_tops 400 --method hoptimus-MLP --evaluation_epochs 100 --lr_evaluation 1e-4 --device 7
python main_evaluate_xenium.py --domain_protocol sample --hvg_gene_tops 400 --method gigapath-MLP --image_model gigapath --evaluation_epochs 100 --lr_evaluation 1e-4 --device 7
python main_evaluate_xenium.py --domain_protocol sample --hvg_gene_tops 400 --method uni-MLP --image_model uni --evaluation_epochs 100 --lr_evaluation 1e-4 --device 7
python main_evaluate_xenium.py --domain_protocol sample --hvg_gene_tops 400 --method pca-MLP --image_model pca --evaluation_epochs 100 --lr_evaluation 1e-4 --device 7


python main_evaluate_xenium.py --domain_protocol bone --hvg_gene_tops 400 --method ours --gene_emb_dim 128 \
--enc1_hidden_channels 128 --enc2_hidden_channels 128 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--neighbor_num 1000 --batch_size 200 --device 7

python main_evaluate_xenium.py --domain_protocol lung --hvg_gene_tops 400 --method ours --gene_emb_dim 128 \
--enc1_hidden_channels 128 --enc2_hidden_channels 128 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--neighbor_num 1000 --batch_size 200 --device 7


# kidney
#python main_evaluate_kidney.py --domain_protocol kidney --evaluate_task macrophage_classification --method ours-MLP --metrics Accuracy Accuracy_TopK --evaluation_epochs 1000 --lr_evaluation 1e-4 --device 3
#python main_evaluate_kidney.py --domain_protocol kidney --evaluate_task macrophage_classification --method hoptimus-MLP --metrics Accuracy Accuracy_TopK --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
#
#python main_evaluate_kidney.py --domain_protocol kidney --evaluate_task macrophage_identification --method ours-MLP --metrics AUC --evaluation_epochs 1000 --lr_evaluation 1e-4 --device 2
#python main_evaluate_kidney.py --domain_protocol kidney --evaluate_task macrophage_identification --method hoptimus-MLP --metrics AUC --evaluation_epochs 100 --lr_evaluation 1e-4 --device 2

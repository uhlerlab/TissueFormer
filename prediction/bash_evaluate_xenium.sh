#
## xenium samples

python main_evaluate_xenium.py --domain_protocol sample --hvg_gene_tops 400 --method ours --gene_emb_dim 128 \
--enc1_hidden_channels 128 --enc2_hidden_channels 128 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--neighbor_num 1000 --batch_size 1000 --device 7

python main_evaluate_xenium.py --domain_protocol bone --hvg_gene_tops 400 --method ours --gene_emb_dim 128 \
--enc1_hidden_channels 128 --enc2_hidden_channels 128 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--neighbor_num 1000 --batch_size 200 --device 7

# visium
#python get_analysis_visium.py --domain_protocol organ --method ours --gene_emb_dim 256 \
#--enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
#--device 3
#
#python get_analysis_visium.py --domain_protocol mouse2human --method ours --gene_emb_dim 256 \
#--enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
#--device 3


# xenium
python get_analysis_sample.py --method ours --enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 3

python get_analysis_lung.py --method ours --enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 3


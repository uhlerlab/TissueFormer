################ visium

# organ
python main_pretrain.py --domain_protocol organ --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --gene_emb_dim 256 \
--enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--gene_encoder_trainable --save_model --device 3

# mouse2human
python main_pretrain.py --domain_protocol mouse2human --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --gene_emb_dim 256 \
--enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--gene_encoder_trainable --save_model --device 5

# human2mouse
python main_pretrain.py --domain_protocol human2mouse --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --gene_emb_dim 256 \
--enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--gene_encoder_trainable --save_model --device 5

# all
python main_pretrain.py --domain_protocol all --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --accumulate_steps 20 --gene_emb_dim 256 \
--enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--gene_encoder_trainable --save_model --device 3

python main_pretrain.py --domain_protocol all_small --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --accumulate_steps 20 --gene_emb_dim 128 \
--enc1_hidden_channels 128 --enc2_hidden_channels 128 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--gene_encoder_trainable --save_model --device 3

################ visium + xenium (lung, lung+)

python main_pretrain2.py --domain_protocol lung --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --accumulate_steps 1 --gene_emb_dim 256 \
--enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--pretrain_epochs 1000 --save_model --device 3

python main_pretrain2.py --domain_protocol lung+ --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --accumulate_steps 1 --gene_emb_dim 256 \
--enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--pretrain_epochs 100 --save_model --device 3

################ visium + xenium (sample, sample+)

python main_pretrain2.py --domain_protocol sample --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --accumulate_steps 1 --gene_emb_dim 256 \
--enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--pretrain_epochs 1000 --save_model --device 3

python main_pretrain2.py --domain_protocol sample+ --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --accumulate_steps 1 --gene_emb_dim 256 \
--enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--pretrain_epochs 200 --save_model --device 3

python main_pretrain2.py --domain_protocol sample+_small --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --accumulate_steps 1 --gene_emb_dim 128 \
--enc1_hidden_channels 128 --enc2_hidden_channels 128 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--pretrain_epochs 100 --save_model --device 3



################ visium + xenium (kidney, wo_kidney)

python main_pretrain2.py --domain_protocol kidney --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --accumulate_steps 1 --gene_emb_dim 256 \
--enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--pretrain_epochs 1000 --gene_encoder_trainable --save_model --device 3

python main_pretrain2.py --domain_protocol wo_kidney --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --accumulate_steps 1 --gene_emb_dim 256 \
--enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
--pretrain_epochs 1000 --gene_encoder_trainable --save_model --device 3
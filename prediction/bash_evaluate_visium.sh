
# organ

python main_evaluate_visium.py --domain_protocol organ --method ours --neighbor_num 5000 --batch_size 10 \
--enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 1

python main_evaluate_visium.py --domain_protocol organ --method ours-MLP --evaluation_epochs 100 --lr_evaluation 1e-5 \
--enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 1

# mouse2human

python main_evaluate_visium.py --domain_protocol mouse2human --method ours --neighbor_num 5000 --batch_size 10 \
--enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 2

python main_evaluate_visium.py --domain_protocol mouse2human --method ours-MLP --evaluation_epochs 50 --lr_evaluation 1e-5 \
--enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 2


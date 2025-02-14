
# visium

python main_evaluate_visium.py --domain_protocol organ --method ours --neighbor_num 5000 --batch_size 10 \
--enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 1

python main_evaluate_visium.py --domain_protocol organ --method ours-MLP --evaluation_epochs 100 --lr_evaluation 1e-5 \
--enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 1

python main_evaluate_visium.py --domain_protocol organ --method mean-pooling --device 1
python main_evaluate_visium.py --domain_protocol organ --method hoptimus-MLP --evaluation_epochs 100 --lr_evaluation 1e-5 --device 1
python main_evaluate_visium.py --domain_protocol organ --method gigapath-MLP --image_model gigapath --evaluation_epochs 100 --lr_evaluation 1e-5 --device 1
python main_evaluate_visium.py --domain_protocol organ --method uni-MLP --image_model uni --evaluation_epochs 100 --lr_evaluation 1e-5 --device 1
python main_evaluate_visium.py --domain_protocol organ --method pca-MLP --image_model pca --evaluation_epochs 300 --lr_evaluation 1e-5 --device 1


# mouse2human

python main_evaluate_visium.py --domain_protocol mouse2human --method ours --neighbor_num 5000 --batch_size 10 \
--enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 2

python main_evaluate_visium.py --domain_protocol mouse2human --method ours-MLP --evaluation_epochs 50 --lr_evaluation 1e-5 \
--enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 2

python main_evaluate_visium.py --domain_protocol mouse2human --method mean-pooling --device 2
python main_evaluate_visium.py --domain_protocol mouse2human --method hoptimus-MLP --evaluation_epochs 100 --lr_evaluation 1e-5 --device 2
python main_evaluate_visium.py --domain_protocol mouse2human --method gigapath-MLP --image_model gigapath --evaluation_epochs 100 --lr_evaluation 1e-5 --device 2
python main_evaluate_visium.py --domain_protocol mouse2human --method uni-MLP --image_model uni --evaluation_epochs 100 --lr_evaluation 1e-5 --device 2
python main_evaluate_visium.py --domain_protocol mouse2human --method pca-MLP --image_model pca --evaluation_epochs 300 --lr_evaluation 1e-5 --device 2



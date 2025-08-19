# Instruction for Pretraining

This instruction provides step-by-step guide for running our pipeline for model pretraining and finetuning.

TissueFormer was pretrained via two-stage curriculum learning. 
In the first stage, the model was pretrained from scratch with spot-resolution data (e.g., Visium slides).
Then in the second stage, the model was further pretrained with cell-resolution data (e.g., Xenium slides).

Before running the pretraining pipeline, one needs to create a directory path (e.g., `../model_checkpoints/`) for storing the model checkpoints.

## Pretraining: First Stage

The file `main_pretrain.py` implements the pretraining pipeline with spot-resolution data from HEST-1K.
To run this pipeline, one needs to modify the data / model directory paths:

```python
    dir_path = '/data/wuqitian/hest_data_visium_protein_preprocess' # modify the path storing the preprocessed visium slides
    meta_info = pd.read_csv("../../data/meta_info_visium.csv") # modify the path for the meta info file of visium slides

    pretrain_model_path = f'../model_checkpoints/{args.method}_pretrain_visium_{args.domain_protocol}.pth' # modify the path for storing the model checkpoints
```

We provide multiple versions of pretrained models in this study. 
Please refer to the following commands to run the pretraining pipeline for different model versions (train / test splits and model sizes).
```bash
    # for generalization across organs, pretrain with visium tissue slides from top five frequent organs
    python main_pretrain.py --domain_protocol organ --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --gene_emb_dim 256 \
    --enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
    --gene_encoder_trainable --save_model --device 3
    
    # for generalization from mouse to human, pretrain with visium tissue slides from mouse
    python main_pretrain.py --domain_protocol mouse2human --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --gene_emb_dim 256 \
    --enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
    --gene_encoder_trainable --save_model --device 3
    
    # pretrain with all visium tissue slides from HEST-1K, large model size
    python main_pretrain.py --domain_protocol all --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --accumulate_steps 20 --gene_emb_dim 256 \
    --enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
    --gene_encoder_trainable --save_model --device 3
    
    # pretrain with all visium tissue slides from HEST-1K, small model size
    python main_pretrain.py --domain_protocol all_small --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --accumulate_steps 20 --gene_emb_dim 128 \
    --enc1_hidden_channels 128 --enc2_hidden_channels 128 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
    --gene_encoder_trainable --save_model --device 3
```

## Pretraining: Second Stage

The file `main_pretrain2.py` implements the pretraining pipeline with cell-resolution data from HEST-1K.
Similarly, one needs to modify the data / model directory paths to run this pipeline:

```python
    dir_path = '/data/wuqitian/hest_data_xenium_protein_preprocess' # modify the path storing the preprocessed xenium slides
    meta_info = meta_info[meta_info['tech'] == 'Xenium'] # modify the path for the meta info file of xenium slides

    pretrain_model_path1 = f'../model_checkpoints/{args.method}_pretrain_visium_all.pth' # modify the path storing the model checkpoint after first-stage pretraining
    pretrain_model_path2 = f'../model_checkpoints/{args.method}_pretrain_xenium_{args.domain_protocol}.pth' # modify the path for storing the model checkpoint for second-stage pretraining
```

The following commands execute the pretraining with xenium slides from HEST-1K. Furthermore, one can specify the train / test splits and model sizes according to specific needs.
```bash
    # pretrain the model (large model size) with xenium slides from HEST-1K
    python main_pretrain2.py --domain_protocol sample+ --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --accumulate_steps 1 --gene_emb_dim 256 \
    --enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
    --pretrain_epochs 200 --save_model --device 3
    
    # pretrain the model (large model size) with xenium slides from HEST-1K
    python main_pretrain2.py --domain_protocol sample+_small --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --accumulate_steps 1 --gene_emb_dim 128 \
    --enc1_hidden_channels 128 --enc2_hidden_channels 128 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
    --pretrain_epochs 100 --save_model --device 3
```

## Finetuning with New Data

As an optional step before applying the pretrained model to new datasets, one can perform the finetuning with the samples from new datasets. 
For doing so, the file `main_pretrain2.py` implements the finetuning pipeline with lung fibrosis data. 
Before running it, please modify the data / model directory paths:

```python
    dir_path = '/data/wuqitian/lung_preprocess' # modify the path storing the preprocessed slides from lung fibrosis data

    pretrain_model_path1 = f'../model_checkpoints/{args.method}_pretrain_visium_all.pth' # modify the path storing the model checkpoint after pretraining
    pretrain_model_path2 = f'../model_checkpoints/{args.method}_pretrain_xenium_{args.domain_protocol}.pth' # modify the path for storing the model checkpoint for finetuning
```

The following commands execute the pretraining with xenium slides from HEST-1K. Furthermore, one can specify the train / test splits and model sizes according to specific needs.
```bash
    # finetune the model (large model size) with new xenium slides from lung fibrosis data
    python main_pretrain2.py --domain_protocol lung --method ours --lr_pretrain 1e-5 --reg_weight 0.5 --accumulate_steps 1 --gene_emb_dim 256 \
    --enc1_hidden_channels 1024 --enc2_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
    --pretrain_epochs 1000 --save_model --device 3
```


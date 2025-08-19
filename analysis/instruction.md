# Instruction for Analysis

This guide describes how to apply TissueFormer for analysis.

In our study, we demonstrated the model on two applications, one based on the lung fibrosis data and another based on the breast tumor data.

## Get Analysis Result

For the lung fibrosis case, one first needs to run `get_analysis_lung.py` to obtain the results including the embeddings and attention maps for each sample.
Before running this script, please modify the directory path for data, model checkpoints and results.

```python
    dir_path = '/data/wuqitian/lung_preprocess'
    meta_info = pd.read_csv("../../data/meta_info_lung.csv")
    result_path = '/data/wuqitian/analysis_pred_data'
    pretrained_state_dict = torch.load('../model_checkpoints/ours_pretrain_xenium_lung.pth')
```

Then one can run the following command:
```bash
    python get_analysis_lung.py --method ours --enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 3
```

## Analysis using Notebook

We provide the jupyter notebook used by our analysis for reproducibility and as the reference for further analysis based on our demonstration.

- `./analysis_lung_fibrosis.ipynb`: analysis for the lung fibrosis case
- `./analysis_breast_tumor.ipynb`: analysis for the breast tumor case


# TissueFormer: A Multi-Modal Foundation Model for Spatial Biology

TissueFormer is pretrained over 1.2K paired tissue slides each of which includes a haematoxylin and eosin (H&E)-stained whole-slide image and its corresponding spatial transcriptomic profile. 
From these tissue slides, we derive 17 million image-expression pairs and a unified gene panel that contains over 20K protein-coding genes for pretraining. 
At inference time, the model can be applied to cross-modality generation (e.g., predict gene expression at cellular resolutions from histology images), predictive tasks at cell / region / slide levels, as well as analysis of intercellular communication and cell subtype identification.

## Model Overview

<img width="1021" height="662" alt="image" src="https://github.com/user-attachments/assets/8edfa4e5-1e53-4bc6-8602-6d806723e3b1" />

## Datasets

All datasets used for the training and evaluation of our model are publicly available. The HEST-1K can be accessed [HuggingFace](https://huggingface.co/datasets/MahmoodLab/hest).
The Xenium human breast tissue slides were included in HEST-1K. We used the data version released by the original paper in [10XGenomics](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast).
The dataset of human lung tissues with pulmonary fibrosis is deposited in the GEO database under accession number [GSE250346](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250346).

## Reproducing the Results

Please follow the steps below to reproduce the results and analysis performed in this study. 

1. Prerequisite

First of all, Anaconda or Miniconda is needed to run the commands in this guide. You can check if it is installed via running 
```bash
    conda
```
If it is not installed, then run the following commands:
```bash
    wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
    chmod +x Anaconda3-2023.03-1-Linux-x86_64.sh
    ./Anaconda3-2023.03-1-Linux-x86_64.sh
    vim ~/.bashrc
    export PATH=/root/anaconda3/bin:$PATH
    source ~/.bashrc
```
Then clone this repository by running the following command in a new terminal:
```bash
    git clone https://github.com/uhlerlab/TissueFormer
```
Make sure you are in the root directory (i.e., TissueFormer/) by typing 
```bash
    cd TissueFormer
```
Create a new environment with the packages provided by `environment.yml`:
```bash
    conda env create -f environment.yml -n tissueformer
    conda activate tissueformer
```

2. Data preprocessing

We provide the [instruction](https://github.com/qitianwu/TissueFormer/blob/main/preprocess/instruction.md) for using our data preprocessing pipeline.

3. Training (pretraining and finetuning)

TissueFormer adopts two-stage curriculum learning for pretraining: the model was first trained with spot-resolution data (e.g., Visium slides) and further trained with cell-resolution data (e.g., Xenium slides).
For applications, the pretrained TissueFormer can be directly applied to zero-shot predictions on new datasets or finetuned with new data. 
We provide the [instruction](https://github.com/qitianwu/TissueFormer/blob/main/training/instruction.md) for using our pipeline of pretraining and finetuning.

4. Inference for prediction

After pretraining, TissueFormer is capable of handling various predictive tasks only using histology images of test samples. 
One typical task is cross-modality generation, i.e., predicting spatial gene expression from histology images. 
Furthermore, the pretrained model can be applied to predictions at different biological scales (cell-level, region-level, slide-level) from histology images.
We provide the [instruction](https://github.com/qitianwu/TissueFormer/blob/main/prediction/instruction.md) for applying TissueFormer to predictions.

5. Inference for analysis

Apart from predictive tasks, TissueFormer supports analysis of intercellular communication and cell subtying from histology images. 
For these, the pretrained model provides whole-slide cell-cell attention maps and cell-level embeddings for interpreting and analyzing the mechanism. 
We provide the [instruction](https://github.com/qitianwu/TissueFormer/blob/main/analysis/instruction.md) for applying TissueFormer to analysis.

6. Visualization

All illustrative figures (Fig. 1 and Supplementary Fig. 1-2) in this study were made using Draw.io, PowerPoint and Adobe Illustrator.

Pointers for nonillustrative figures:

- `./analysis/gene_exp_pred_visium.ipynb`: Fig. 2, Supplementary Fig. 3-4
- `./analysis/gene_exp_pred_xenium.ipynb`: Fig. 3, Supplementary Fig. 5-7
- `./analysis/diagnosis_pred_lung.ipynb`: Fig. 4, Supplementary Fig. 8
- `./analysis/analysis_lung_fibrosis.ipynb`: Fig. 5, Supplementary Fig. 9-10
- `./analysis/analysis_breast_tumor.ipynb`: Supplementary Fig. 11

## Apply TissueFormer to User-provided Datasets

Applying TissueFormer to any new dataset typically involves the following steps:

1. Load preprocessed dataset
2. Load pretrained model checkpoints 
3. Optional: finetune the model on new datasets
4. Apply the model for prediction (e.g., predict gene expression from histology images)
5. Apply the model for analysis (extract the cell embeddings and attention maps)

### Application Demo 1

Here we use the test Xenium samples (TENX126, TENX123, TENX124, TENX121, TENX119, TENX118) as an example to demonstrate how to use our model on test data and reproduce the results.

1. Load preprocessed dataset

One can download the preprocessed data from this [Google Drive](https://drive.google.com/drive/folders/1ujmHmkAOfZtqpUwLSDIAU3KC2AbfpTpj?usp=sharing) into a folder `./data`

2. Load pretrained model checkpoints

The same google drive repository contains the pretrained model checkpoints and one can download them into a folder `./checkpoint`.

3. Apply the model for prediction

Then one can refer to `./prediction/main_evaluate_xenium.py` and modify the directory paths storing the dataset, model checkpoint and results:
```python
    dir_path = './data/hest_data_xenium_protein_preprocess'

    pretrain_model_path = './checkpoint/ours_pretrain_xenium_sample+_small.pth'

    result_path = f'./result/gene_expression_prediction/{test_sample}'
```
After modifying the paths, one can run the following script to execute prediction on these test samples (the `batch_size` can be adjusted to balance the memory and time costs):
```bash
    python main_evaluate_xenium.py --domain_protocol sample --hvg_gene_tops 400 --method ours --gene_emb_dim 128 \
  --enc1_hidden_channels 128 --enc2_hidden_channels 128 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
  --neighbor_num 1000 --batch_size 1000 --device 7
```
The prediction results are stored into the path `./result/gene_expression_prediction/`.

4. Visualization

The visualization code for our results (Fig. 3a, Supplementary Fig. 5) is provided in this [demo1]().

### Application Demo 1

For applying TissueFormer to user-provided datasets, we provide a [demo2]() as an example.
One can use this demo by replacing the dataset with one's own and following the instruction below. 

First, one needs to specify the directory path storing the dataset and load the dataset that is split for training and test:
```python
    dir_path = '/ewsc/wuqitian/lung_preprocess'
    meta_info = pd.read_csv("../../data/meta_info_lung.csv")
    
    # train data can be used as the reference for in-context learning or for finetuning the model
    train_samples = meta_info[meta_info['affect'] == 'Unaffected']['sample'].tolist()[:-1]
    
    # test data for evaluation
    test_samples = meta_info[meta_info['affect'] == 'Unaffected']['sample'].tolist()[-1:]
    
    # create dataloader
    train_datasets = dataset_create(dir_path, train_samples)
    train_dataloader = DataLoader(train_datasets, batch_size=1, shuffle=True)
    test_datasets = dataset_create(dir_path, test_samples)
    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False)
```

Second, load the pretrained model checkpoint (one can choose the pretrained model version):
```python 
    pretrained_state_dict = torch.load('../../model_checkpoints/ours_pretrain_xenium_lung.pth') # one can choose the model version
    encoder1_pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k.startswith("encoder1.")}
    model_state_dict = model_ours.state_dict()
    encoder1_model_dict = {k: v for k, v in model_state_dict.items() if k.startswith("encoder1.")}
    for k, v in encoder1_pretrained_dict.items():
        assert (k in encoder1_model_dict)
        assert (v.size() == encoder1_model_dict[k].size())
    model_state_dict.update(encoder1_pretrained_dict)
    model_ours.load_state_dict(model_state_dict)
```

Later on, one can use the model for prediction, analysis (extract cell-level embeddings and attentions) or finetuning the model with downstream labels 
by following the scripts in [demo2]().

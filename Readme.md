# TissueFormer: A Multi-Modal Foundation Model for Spatial Biology

Code for paper "Multi-Modal Foundation Model with Whole-Slide Attention Enables Transferrable Digital Pathology at Single-Cell Resolution".

TissueFormer is pretrained over 1.2K paired tissue slides each of which includes a haematoxylin and eosin (H&E)-stained whole-slide image and its corresponding spatial transcriptomic profile. 
From these tissue slides, we derive 17 million image-expression pairs and a unified gene panel that contains over 20K protein-coding genes for pretraining. 
At inference time, the model can be applied to cross-modality generation (e.g., predict gene expression at cellular resolutions from histology images), predictive tasks at cell / region / slide levels, as well as analysis of intercellular communication and cell subtype identification.

## Model Overview

<img width="700" alt="image" src="https://files.mdnice.com/user/23982/3c433a8d-faf4-45f7-a4bd-c599e3288077.png">

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

We provide the [instruction]() for using our data preprocessing pipeline.

3. Training (pretraining and finetuning)

TissueFormer adopts two-stage curriculum learning for pretraining: the model was first trained with spot-resolution data (e.g., Visium slides) and further trained with cell-resolution data (e.g., Xenium slides).
For applications, the pretrained TissueFormer can be directly applied to zero-shot predictions on new datasets or finetuned with new data. 
We provide the [instruction]() for using our pipeline of pretraining and finetuning.

4. Inference for prediction

After pretraining, TissueFormer is capable of handling various predictive tasks only using histology images of test samples. 
One typical task is cross-modality generation, i.e., predicting spatial gene expression from histology images. 
Furthermore, the pretrained model can be applied to predictions at different biological scales (cell-level, region-level, slide-level) from histology images.
We provide the [instruction]() for applying TissueFormer to predictions.

5. Inference for analysis

Apart from predictive tasks, TissueFormer supports analysis of intercellular communication and cell subtying from histology images. 
For these, the pretrained model provides whole-slide cell-cell attention maps and cell-level embeddings for interpreting and analyzing the mechanism. 
We provide the [instruction]() for applying TissueFormer to analysis.

6. Visualization

All illustrative figures (Fig. 1 and Supplementary Fig. 1-2) in this study were made using Draw.io, PowerPoint and Adobe Illustrator.

Pointers for nonillustrative figures:

- `./analysis/gene_exp_pred_visium.ipynb`: Fig. 2, Supplementary Fig. 3-4
- `./analysis/gene_exp_pred_xenium.ipynb`: Fig. 3, Supplementary Fig. 5-7
- `./analysis/diagnosis_pred_lung.ipynb`: Fig. 4, Supplementary Fig. 8
- `./analysis/analysis_lung_fibrosis.ipynb`: Fig. 5, Supplementary Fig. 9-10
- `./analysis/analysis_breast_tumor.ipynb`: Supplementary Fig. 11

## Apply TissueFormer to User-provided Datasets

Applying TissueFormer to any new dataset typically involves the following steps.

1. Load preprocessed dataset
2. Load pretrained model checkpoints 
3. Optional: finetune the model on new datasets
4. Apply the model for prediction (e.g., predict gene expression from histology images)
5. Apply the model for analysis (extract the cell embeddings and attention maps)

For detailed guideline, please refer to this [Demo]().
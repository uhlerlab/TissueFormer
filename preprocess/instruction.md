# Instruction for Data Preprocessing

This instruction provides step-by-step guide for running our data preprocessing pipeline.

## Downloading data

As the first step, create a folder (e.g., `../data`) to store the downloaded datasets (you will need at least 1.6T disk storage).
We used three datasets in this study that are all publicly available:

- HEST-1K data: https://huggingface.co/datasets/MahmoodLab/hest.
- Xenium human lung tissue slides with pulmonary fibrosis: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250346.
- Xenium human breast tissue slides: https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast.

## Preprocessing HEST-1K Data

The file `preprocess_hest.py` implements the preprocessing pipeline for HEST-1K.
To run this pipeline, one needs to modify the data directory paths at the end of `preprocess_hest.py`:

```python
    if __name__ == '__main__':
        dir_path1 = '/data/wuqitian/hest_converted' # modify the path storing (visium slides from) hest-1k data
        save_path1 = '/data/wuqitian/hest_data_visium_protein' # a new directory path for storing preprocessed visium slides
        preprocess_save_path1 = '/data/wuqitian/hest_data_visium_protein_preprocess' # another new directory path for storing preprocessed visium slides
        meta_info_save_path1 = '../data/meta_info_visium.csv' # a file path for storing meta info of visium slides
        convert_filter_gene_names(dir_path1, save_path1) # gene name conversion for a unified gene panel
        gene_preprocess(save_path1, preprocess_save_path1, tech='visium') # preprocess gene expression
        meta_data_info(preprocess_save_path1, meta_info_save_path1) # extract meta info of samples
    
        dir_path2 = '/data/wuqitian/hest_converted/xenium_cell' # modify the path storing (xenium slides from) hest-1k data
        save_path2 = '/data/wuqitian/hest_data_xenium_protein' # a new directory path for storing preprocessed xenium slides
        preprocess_save_path2 = '/data/wuqitian/hest_data_xenium_protein_preprocess' # another new directory path for storing preprocessed xenium slides
        meta_info_save_path2 = '../data/meta_info_xenium.csv' # a file path for storing meta info of xenium slides
        convert_filter_gene_names(dir_path2, save_path2) # gene name conversion for a unified gene panel
        gene_preprocess(save_path2, preprocess_save_path2, tech='xenium') # preprocess gene expression
        meta_data_info(preprocess_save_path2, meta_info_save_path2) # extract meta info of samples
```

Then one can run this pipeline in the terminal:
```bash
    python ./preprocess_hest.py
```
This step will take several hours to complete.

## Preprocessing Lung Fibrosis Data

The file `preprocess_lung.py` implements the preprocessing pipeline for lung fibrosis data.
Similarly, one needs to modify the data directory paths before running this pipeline:

```python
    dir_path = '/data/wuqitian/lung' # modify the path storing lung fibrosis data
    preprocess_save_path = '/data/wuqitian/lung_preprocess' # a new directory path for storing preprocessed data

    files = os.listdir(dir_path)

    meta_data = sc.read('/data/wuqitian/lung/GSE250346_cell_type_niche_centroid.h5ad') # modify the path storing lung fibrosis data
    he_annotate_data = pd.read_csv(open('/data/wuqitian/lung/HE_annotations/cells_partitioned_by_annotation.csv', 'r', encoding='utf-8')) # modify the path storing lung fibrosis data
```

Then one can run this pipeline in the terminal:
```bash
    python ./preprocess_lung.py
```
This step will again take several hours to complete.


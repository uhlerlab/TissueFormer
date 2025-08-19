# Instruction for Prediction

This instruction provides step-by-step guide for running the pipeline for predictions.

TissueFormer can be applied to various predictive tasks from histology images at different biological scales (cell-level, region-level, slide-level).
In our study, we apply the model to predicting spatial gene expression (at spot-level resolution and cell-level resolution) as well as 
predicting diagnostic annotations from histology images.

Before running the prediction pipeline, we suggest creating a directory path (e.g., `../result/`) for storing the results.

## Gene Expression Prediction (Visium)

The file `main_evaluate_visium.py` implements the pipeline that applies TissueFormer to gene expression prediction on Visium data.
To run this pipeline, one needs to modify the data / model directory paths:

```python
    dir_path = '/data/wuqitian/hest_data_visium_protein_preprocess' # modify the path storing the preprocessed visium slides
    meta_info = pd.read_csv("../../data/meta_info_visium.csv") # modify the path for the meta info file of visium slides

    pretrain_model_path = f'../model_checkpoints/ours_pretrain_visium_{args.domain_protocol}.pth' # modify the path storing the pretrained model checkpoints
    evaluation_model_path = f'../model_checkpoints/{args.method}_evaluate_visium_{args.domain_protocol}.pth' # optinal, the path for storing evaluation model checkpoints
    result_path = f'../result/{args.method}_visium_{args.domain_protocol}.csv' # create a file path for storing the results
```

Then run the following commands to obtain the prediction results for the gene expression prediction task on test Visium slides from HEST-1K.
```bash
    # generalization across organs, test the model (pretrained with visium slides of top five frequent organs) 
    # on held-out visium slides of the top five frequent organs as well as visium slides of other unseen organs
    python main_evaluate_visium.py --domain_protocol organ --method ours --neighbor_num 5000 --batch_size 10 \
    --enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 1

    # generalization from mouse to human, test the model (pretrained with visium slides of mouse)
    # on visium slides of human
    python main_evaluate_visium.py --domain_protocol mouse2human --method ours --neighbor_num 5000 --batch_size 10 \
    --enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 2
```

## Gene Expression Prediction (Xenium)

The file `main_evaluate_xenium.py` implements the pipeline that applies TissueFormer to gene expression prediction on Xenium data.
Similarly, to run this pipeline, one needs to modify the data / model directory paths. For example:

```python
    dir_path = '/data/wuqitian/hest_data_xenium_protein_preprocess' # modify the path storing the preprocessed xenium slides
    meta_info = pd.read_csv("../../data/meta_info_xenium.csv") # modify the path for the meta info file of xenium slides
    
    pretrain_model_path = f'../model_checkpoints/ours_pretrain_xenium_sample+_small.pth' # modify the path storing the pretrained model checkpoints

    result_path = f'/data/wuqitian/analysis_pred_data/gene_expression_prediction/{test_sample}' # create a file path for storing the results
```

Then please refer to the following commands that run the prediction pipeline for the gene expression prediction task on test Xenium slides from HEST-1K.
```bash
    # test the model on xenium samples from different organs
    python main_evaluate_xenium.py --domain_protocol sample --hvg_gene_tops 400 --method ours --gene_emb_dim 128 \
    --enc1_hidden_channels 128 --enc2_hidden_channels 128 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
    --neighbor_num 1000 --batch_size 1000 --device 7
    
    # test the model on bone tissue slides with new unseen genes
    python main_evaluate_xenium.py --domain_protocol bone --hvg_gene_tops 400 --method ours --gene_emb_dim 128 \
    --enc1_hidden_channels 128 --enc2_hidden_channels 128 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --enc2_num_layers_mlp 1 \
    --neighbor_num 1000 --batch_size 200 --device 7
```

## Digital Diagnosis Predictive Tasks

The file `main_evaluate_lung.py` implements the pipeline that applies TissueFormer to various predictive tasks of digital diagnosis on lung fibrosis data.
In this study, the predictive tasks include:

- ROI disease severity inference: predict the disease severity labels (i.e., pseudo times) for airspace regions (regression task)
- Disease-associated cell detection: predict the cell types (classification task)
- Pathology feature annotation: predict the pathology annotation feature (classification task)
- Spatial niche classification: predict the spatial niche labels (classification task)

Similar to gene expression prediction, one needs to modify the data / model directory paths:

```python
    dir_path = '/data/wuqitian/lung_preprocess' # modify to the path storing the preprocessed lung fibrosis data
    meta_info = pd.read_csv("../../data/meta_info_lung.csv") # modify to the path storing the meta info of lung fibrosis data
    he_annotate_data = pd.read_csv(open('/data/wuqitian/lung/HE_annotations/cells_partitioned_by_annotation.csv', 'r', encoding='utf-8')) # modify to the path storing the annotations of lung fibrosis data
    
    pretrain_model_path = f'../model_checkpoints/ours_pretrain_xenium_lung+.pth' # modify to the path storing the pretrained model checkpoints
    evaluation_model_path = f'../model_checkpoints/{args.method}_evaluate_xenium_lung.pth' # optional, create a new path for storing the evaluation model checkpoints
```

Depending on specific tasks, one needs to specify the directory path for storing the results. For example, for pathology feature annotation, modify the `result_path`:
```python
    result_path = f'/data/wuqitian/analysis_pred_data/he_annotation_classification/{args.he_annotation_type}' # create a new path for storing the results
```

Then please refer to the following commands to obtain the predictions from the model.
```bash
    ## ROI disease severity inference
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task region_time_prediction --method ours-MLP --metrics RMSE --evaluation_epochs 394 --lr_evaluation 1e-4 --no_image_encoder --use_pred_gene --device 5
    
    ## Disease-associated cell detection
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task cell_type_classification --method ours-MLP --metrics AUC --cell_type Transitional_AT2 --evaluation_epochs 100 --lr_evaluation 1e-5 --device 4 --use_pred_gene --no_image_encoder
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task cell_type_classification --method ours-MLP --metrics AUC --cell_type Activated_Fibrotic_FBs --evaluation_epochs 100 --lr_evaluation 1e-4 --device 4 --use_pred_gene --no_image_encoder
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task cell_type_classification --method ours-MLP --metrics AUC --cell_type Capillary --evaluation_epochs 100 --lr_evaluation 1e-4 --device 4 --use_pred_gene --no_image_encoder
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task cell_type_classification --method ours-MLP --metrics AUC --cell_type SPP1+_Macrophages --evaluation_epochs 100 --lr_evaluation 1e-5 --device 4 --use_pred_gene --no_image_encoder
    
    ## Pathology feature annotation
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type severe_fibrosis --evaluation_epochs 10 --lr_evaluation 1e-4 --device 3
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type epithelial_detachment --evaluation_epochs 100 --lr_evaluation 1e-3 --device 5
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type fibroblastic_focus --evaluation_epochs 100 --lr_evaluation 1e-7 --device 3
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type hyperplastic_aec --evaluation_epochs 100 --lr_evaluation 1e-2 --device 5
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type large_airway --evaluation_epochs 100 --lr_evaluation 1e-5 --device 5
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type giant_cell --evaluation_epochs 100 --lr_evaluation 1e-3 --device 5
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type advanced_remodeling --evaluation_epochs 1 --lr_evaluation 1e-2 --device 5
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type granuloma --evaluation_epochs 1 --lr_evaluation 5e-2 --device 5
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type small_airway --evaluation_epochs 10 --lr_evaluation 1e-2 --device 5
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type venule --evaluation_epochs 10 --lr_evaluation 1e-2 --device 5
    
    ## Spatial niche classification
    # ours using measured gene expression
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T1 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T2 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T3 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T4 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T5 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T6 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T7 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T8 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T9 --evaluation_epochs 100 --lr_evaluation 1e-3 --device 3
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T10 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T11 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T12 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
    # ours using predicted gene expression
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T1 --evaluation_epochs 100 --lr_evaluation 1e-4 --use_pred_gene --no_image_encoder --device 6
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T2 --evaluation_epochs 100 --lr_evaluation 1e-4 --use_pred_gene --no_image_encoder --device 6
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T3 --evaluation_epochs 100 --lr_evaluation 1e-4 --use_pred_gene --no_image_encoder --device 6
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T4 --evaluation_epochs 100 --lr_evaluation 1e-4 --use_pred_gene --no_image_encoder --device 6
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T5 --evaluation_epochs 100 --lr_evaluation 1e-4 --use_pred_gene --no_image_encoder --device 6
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T6 --evaluation_epochs 100 --lr_evaluation 1e-4 --use_pred_gene --no_image_encoder --device 6
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T7 --evaluation_epochs 100 --lr_evaluation 1e-4 --use_pred_gene --no_image_encoder --device 6
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T8 --evaluation_epochs 100 --lr_evaluation 1e-4 --use_pred_gene --no_image_encoder --device 6
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T9 --evaluation_epochs 100 --lr_evaluation 1e-4 --use_pred_gene --no_image_encoder --device 6
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T10 --evaluation_epochs 100 --lr_evaluation 1e-4 --use_pred_gene --no_image_encoder --device 6
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T11 --evaluation_epochs 100 --lr_evaluation 1e-4 --use_pred_gene --no_image_encoder --device 6
    python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T12 --evaluation_epochs 100 --lr_evaluation 1e-4 --use_pred_gene --no_image_encoder --device 6
```

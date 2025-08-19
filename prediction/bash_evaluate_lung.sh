## lung, pathology annotation prediction
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


## lung niche classification
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

## cell type identification
python main_evaluate_lung.py --domain_protocol lung --evaluate_task cell_type_classification --method ours-MLP --metrics AUC --cell_type Transitional_AT2 --evaluation_epochs 100 --lr_evaluation 1e-5 --device 4 --use_pred_gene --no_image_encoder
python main_evaluate_lung.py --domain_protocol lung --evaluate_task cell_type_classification --method ours-MLP --metrics AUC --cell_type Activated_Fibrotic_FBs --evaluation_epochs 100 --lr_evaluation 1e-4 --device 4 --use_pred_gene --no_image_encoder
python main_evaluate_lung.py --domain_protocol lung --evaluate_task cell_type_classification --method ours-MLP --metrics AUC --cell_type Capillary --evaluation_epochs 100 --lr_evaluation 1e-4 --device 4 --use_pred_gene --no_image_encoder
python main_evaluate_lung.py --domain_protocol lung --evaluate_task cell_type_classification --method ours-MLP --metrics AUC --cell_type SPP1+_Macrophages --evaluation_epochs 100 --lr_evaluation 1e-5 --device 4 --use_pred_gene --no_image_encoder

## region time prediction
python main_evaluate_lung.py --domain_protocol lung --evaluate_task region_time_prediction --method ours-MLP --metrics RMSE --evaluation_epochs 394 --lr_evaluation 1e-4 --no_image_encoder --use_pred_gene --device 5


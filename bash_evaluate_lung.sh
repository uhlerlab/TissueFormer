
## lung
#python main_evaluate_lung.py --domain_protocol lung --evaluate_task gene_regression --hvg_gene_tops 50 100 200 400 --method ours --neighbor_num 10000 --batch_size 10 \
#--enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 3
#python main_evaluate_lung.py --domain_protocol lung --evaluate_task gene_regression --hvg_gene_tops 50 100 200 400 --method ours-MLP --evaluation_epochs 100 --lr_evaluation 1e-4 \
#--enc1_hidden_channels 1024 --enc1_num_layers_prop 2 --enc1_num_layers_mlp 2 --device 3
#python main_evaluate_lung.py --domain_protocol lung --evaluate_task gene_regression --hvg_gene_tops 50 100 200 400 --method hoptimus-MLP --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
#
## lung, niche classification
#python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours --metrics AUC --niche_type C1 --device 7
#python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours --metrics AUC --niche_type C2 --device 7
#python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours --metrics AUC --niche_type C3 --device 7
#

# lung, pathology annotation prediction
#python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type severe_fibrosis --evaluation_epochs 10 --lr_evaluation 1e-4 --device 3
#python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type epithelial_detachment --evaluation_epochs 100 --lr_evaluation 1e-3 --device 5
#python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type fibroblastic_focus --evaluation_epochs 100 --lr_evaluation 1e-7 --device 3
#python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type hyperplastic_aec --evaluation_epochs 100 --lr_evaluation 1e-2 --device 5
#python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type large_airway --evaluation_epochs 100 --lr_evaluation 1e-5 --device 5
#python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type giant_cell --evaluation_epochs 100 --lr_evaluation 1e-3 --device 5

python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type advanced_remodeling --evaluation_epochs 1 --lr_evaluation 1e-2 --device 5
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type granuloma --evaluation_epochs 1 --lr_evaluation 5e-2 --device 5
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type airway_smooth_muscle --evaluation_epochs 1 --lr_evaluation 1e-2 --device 5
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type remodeled_epithelium --evaluation_epochs 10 --lr_evaluation 1e-4 --device 5
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type small_airway --evaluation_epochs 10 --lr_evaluation 1e-2 --device 5
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method ours-MLP --metrics AUC --he_annotation_type venule --evaluation_epochs 10 --lr_evaluation 1e-2 --device 5


# lung niche classification
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T1 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T2 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T6 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T7 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours-MLP --metrics AUC --niche_type T8 --evaluation_epochs 100 --lr_evaluation 1e-5 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method ours --metrics AUC --niche_type T1 --neighbor_num 200 --batch_size 1000 --device 3

# lung, pathology annotation prediction, baseline
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method uni-MLP --image_model uni \
--metrics AUC --he_annotation_type severe_fibrosis --evaluation_epochs 10 --lr_evaluation 1e-7 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --he_annotation_type severe_fibrosis --evaluation_epochs 1 --lr_evaluation 1e-2 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method hoptimus-MLP \
--metrics AUC --he_annotation_type severe_fibrosis --evaluation_epochs 1 --lr_evaluation 1e-7 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method uni-MLP --image_model uni \
--metrics AUC --he_annotation_type epithelial_detachment --evaluation_epochs 10 --lr_evaluation 1e-7 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --he_annotation_type epithelial_detachment --evaluation_epochs 100 --lr_evaluation 1e-3 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method hoptimus-MLP \
--metrics AUC --he_annotation_type epithelial_detachment --evaluation_epochs 100 --lr_evaluation 1e-2 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method uni-MLP --image_model uni \
--metrics AUC --he_annotation_type fibroblastic_focus --evaluation_epochs 1 --lr_evaluation 1e-3 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --he_annotation_type fibroblastic_focus --evaluation_epochs 10 --lr_evaluation 1e-5 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method hoptimus-MLP \
--metrics AUC --he_annotation_type fibroblastic_focus --evaluation_epochs 1 --lr_evaluation 1e-7 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method uni-MLP --image_model uni \
--metrics AUC --he_annotation_type giant_cell --evaluation_epochs 1 --lr_evaluation 1e-6 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --he_annotation_type giant_cell --evaluation_epochs 100 --lr_evaluation 1e-2 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method hoptimus-MLP \
--metrics AUC --he_annotation_type giant_cell --evaluation_epochs 1 --lr_evaluation 1e-6 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method uni-MLP --image_model uni \
--metrics AUC --he_annotation_type hyperplastic_aec --evaluation_epochs 100 --lr_evaluation 1e-2 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --he_annotation_type hyperplastic_aec --evaluation_epochs 100 --lr_evaluation 1e-3 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method hoptimus-MLP \
--metrics AUC --he_annotation_type hyperplastic_aec --evaluation_epochs 1 --lr_evaluation 1e-3 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method uni-MLP --image_model uni \
--metrics AUC --he_annotation_type large_airway --evaluation_epochs 10 --lr_evaluation 1e-2 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --he_annotation_type large_airway --evaluation_epochs 1 --lr_evaluation 1e-3 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method hoptimus-MLP \
--metrics AUC --he_annotation_type large_airway --evaluation_epochs 1 --lr_evaluation 1e-7 --device 3



python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method uni-MLP --image_model uni \
--metrics AUC --he_annotation_type advanced_remodeling --evaluation_epochs 1 --lr_evaluation 1e-6 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --he_annotation_type advanced_remodeling --evaluation_epochs 100 --lr_evaluation 1e-6 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method hoptimus-MLP \
--metrics AUC --he_annotation_type advanced_remodeling --evaluation_epochs 1 --lr_evaluation 1e-7 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method uni-MLP --image_model uni \
--metrics AUC --he_annotation_type granuloma --evaluation_epochs 10 --lr_evaluation 1e-4 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --he_annotation_type granuloma --evaluation_epochs 1 --lr_evaluation 1e-2 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method hoptimus-MLP \
--metrics AUC --he_annotation_type granuloma --evaluation_epochs 1 --lr_evaluation 1e-7 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method uni-MLP --image_model uni \
--metrics AUC --he_annotation_type airway_smooth_muscle --evaluation_epochs 10 --lr_evaluation 1e-2 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --he_annotation_type airway_smooth_muscle --evaluation_epochs 10 --lr_evaluation 1e-2 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method hoptimus-MLP \
--metrics AUC --he_annotation_type airway_smooth_muscle --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method uni-MLP --image_model uni \
--metrics AUC --he_annotation_type remodeled_epithelium --evaluation_epochs 100 --lr_evaluation 1e-5 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --he_annotation_type remodeled_epithelium --evaluation_epochs 100 --lr_evaluation 1e-3 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method hoptimus-MLP \
--metrics AUC --he_annotation_type remodeled_epithelium --evaluation_epochs 100 --lr_evaluation 1e-2 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method uni-MLP --image_model uni \
--metrics AUC --he_annotation_type small_airway --evaluation_epochs 10 --lr_evaluation 1e-2 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --he_annotation_type small_airway --evaluation_epochs 1 --lr_evaluation 1e-2 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method hoptimus-MLP \
--metrics AUC --he_annotation_type small_airway --evaluation_epochs 1 --lr_evaluation 1e-3 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method uni-MLP --image_model uni \
--metrics AUC --he_annotation_type venule --evaluation_epochs 1 --lr_evaluation 1e-5 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --he_annotation_type venule --evaluation_epochs 1 --lr_evaluation 1e-2 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method hoptimus-MLP \
--metrics AUC --he_annotation_type venule --evaluation_epochs 1 --lr_evaluation 1e-3 --device 3



# lung, niche classification, baseline
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method uni-MLP --image_model uni \
--metrics AUC --niche_type T1 --evaluation_epochs 100 --lr_evaluation 1e-2 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --niche_type T1 --evaluation_epochs 100 --lr_evaluation 1e-4 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method hoptimus-MLP \
--metrics AUC --niche_type T1 --evaluation_epochs 10 --lr_evaluation 1e-3 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method uni-MLP --image_model uni \
--metrics AUC --niche_type T2 --evaluation_epochs 10 --lr_evaluation 1e-2 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --niche_type T2 --evaluation_epochs 100 --lr_evaluation 1e-3 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method hoptimus-MLP \
--metrics AUC --niche_type T2 --evaluation_epochs 10 --lr_evaluation 1e-7 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method uni-MLP --image_model uni \
--metrics AUC --niche_type T6 --evaluation_epochs 1 --lr_evaluation 1e-3 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --niche_type T6 --evaluation_epochs 1 --lr_evaluation 1e-4 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method hoptimus-MLP \
--metrics AUC --niche_type T6 --evaluation_epochs 100 --lr_evaluation 1e-3 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method uni-MLP --image_model uni \
--metrics AUC --niche_type T7 --evaluation_epochs 1 --lr_evaluation 1e-5 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --niche_type T7 --evaluation_epochs 10 --lr_evaluation 1e-3 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method hoptimus-MLP \
--metrics AUC --niche_type T7 --evaluation_epochs 1 --lr_evaluation 1e-2 --device 3

python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method uni-MLP --image_model uni \
--metrics AUC --niche_type T8 --evaluation_epochs 1 --lr_evaluation 1e-2 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method gigapath-MLP --image_model gigapath \
--metrics AUC --niche_type T8 --evaluation_epochs 1 --lr_evaluation 1e-6 --device 3
python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method hoptimus-MLP \
--metrics AUC --niche_type T8 --evaluation_epochs 10 --lr_evaluation 1e-3 --device 3

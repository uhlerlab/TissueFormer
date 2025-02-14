#methods=("ours-MLP" "uni-MLP" "gigapath-MLP" "hoptimus-MLP")
#image_models=("hoptimus" "uni" "gigapath" "hoptimus")
##methods=("ours-MLP")
##image_models=("hoptimus")
##he_annotations1=('severe_fibrosis' 'epithelial_detachment' 'fibroblastic_focus' 'giant_cell' 'hyperplastic_aec' 'large_airway')
##he_annotations2=('small_airway-1' 'small_airway-2' 'venule-1' 'venule-2' 'artery-1' 'artery-2')
##he_annotations3=('minimally_remodeled_alveoli' 'remodeled_epithelium' 'advanced_remodeling' 'microscopic_honeycombing' 'mixed_inflammation' 'remnant_alveoli' 'muscularized_artery' 'TLS' 'granuloma' 'airway_smooth_muscle')
#he_annotations3=('small_airway' 'venule')
#
#learning_rates=(1e-7 1e-6 1e-5 1e-4 1e-3 1e-2)
#epoch_nums=(100 10 1)
#dev=6
#
#for he in ${he_annotations3[@]}
#  do
#  for i in "${!methods[@]}"
#  do
#    for lr in ${learning_rates[@]}
#    do
#      for epoch_num in ${epoch_nums[@]}
#      do
#          m=${methods[i]}
#          e=${image_models[i]}
#          python main_evaluate_lung.py --domain_protocol lung --evaluate_task he_annotation_classification --method $m --image_model $e --metrics AUC --he_annotation_type $he --evaluation_epochs $epoch_num --lr_evaluation $lr --device $dev
#      done
#    done
#  done
#done


#methods=("ours-MLP" "uni-MLP" "gigapath-MLP" "hoptimus-MLP")
#image_models=("hoptimus" "uni" "gigapath" "hoptimus")
#niche_types=('T1' 'T2' 'T3' 'T4' 'T5' 'T6' 'T7' 'T8' 'T9' 'T10' 'T11' 'T12')
#
#learning_rates=(1e-7 1e-6 1e-5 1e-4 1e-3 1e-2)
#epoch_nums=(100 10 1)
#dev=7
#
#for type in ${niche_types[@]}
#  do
#  for i in "${!methods[@]}"
#  do
#    for lr in ${learning_rates[@]}
#    do
#      for epoch_num in ${epoch_nums[@]}
#      do
#          m=${methods[i]}
#          e=${image_models[i]}
#          python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method $m --image_model $e --metrics AUC --niche_type $type --evaluation_epochs $epoch_num --lr_evaluation $lr --device $dev
#      done
#    done
#  done
#done


methods=("ours-MLP")
image_models=("hoptimus")
niche_types=('T1' 'T2' 'T3' 'T4' 'T5' 'T6' 'T7' 'T8' 'T9' 'T10' 'T11' 'T12')

learning_rates=(1e-7 1e-6 1e-5 1e-4 1e-3 1e-2)
epoch_nums=(100 10 1)
dev=7

for type in ${niche_types[@]}
  do
  for i in "${!methods[@]}"
  do
    for lr in ${learning_rates[@]}
    do
      for epoch_num in ${epoch_nums[@]}
      do
          m=${methods[i]}
          e=${image_models[i]}
          python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method $m --image_model $e --metrics AUC --niche_type $type --evaluation_epochs $epoch_num --lr_evaluation $lr --device $dev --use_pred_gene
      done
    done
  done
done


#methods=("ours-MLP" "uni-MLP" "gigapath-MLP" "hoptimus-MLP")
#image_models=("hoptimus" "uni" "gigapath" "hoptimus")
#niche_types=('C1' 'C2' 'C3' 'C4' 'C5' 'C6' 'C7' 'C8' 'C9' 'C10' 'C11' 'C12')
#
#learning_rates=(1e-7 1e-6 1e-5 1e-4 1e-3 1e-2)
#epoch_nums=(100 10 1)
#dev=6
#
#for type in ${niche_types[@]}
#  do
#  for i in "${!methods[@]}"
#  do
#    for lr in ${learning_rates[@]}
#    do
#      for epoch_num in ${epoch_nums[@]}
#      do
#          m=${methods[i]}
#          e=${image_models[i]}
#          python main_evaluate_lung.py --domain_protocol lung --evaluate_task niche_classification --method $m --image_model $e --metrics AUC --niche_type $type --evaluation_epochs $epoch_num --lr_evaluation $lr --device $dev
#      done
#    done
#  done
#done
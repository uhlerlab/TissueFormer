
# xenium
#python main.py --device 3 --mode pretrain --method transformer \
#--gene_encoder_trainable --gene_encoder_pretrained --save_model

python main.py --device 2 --method transformer --mode evaluation --split_protocol in_sample
python main.py --device 2 --method transformer --mode supervise --split_protocol in_sample
python main.py --device 2 --method linear --mode evaluation --split_protocol in_sample



python main.py --device 2 --method transformer --mode evaluation --split_protocol out_sample
python main.py --device 2 --method transformer --mode supervise --split_protocol out_sample
python main.py --device 2 --method linear --mode evaluation --split_protocol out_sample

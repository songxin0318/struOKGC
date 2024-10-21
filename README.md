# struOKGC

# datasets
Unzip the dataset.zip and get the FB15K-237-N and the Wiki27K folder.</br>
The UMLS-PubMed is provided by the CPL model. (GitHub: https://github.com/INK-USC/CPL#datasets)</br>
The dataset process steps are followed by the PKGC model (https://github.com/THU-KEG/PKGC).</br>

# run the code
(1) for the wiki27k and FB15k-237-N dataset
python3 run_args.py \
  --model_name bert-based-uncased \
  --max_epoch 100 \
  --batch_size 16 \
  --early_stop 20 \
  --lr 5e-5 \
  --lm_lr 1e-6 \
  --seed 234 \
  --decay_rate 0.99 \
  --weight_decay 0.0005 \
  --data_dir ./dataset/Wiki27K \
  --out_dir ./checkpoint/Wiki27K \
  --valid_step 10000 \
  --use_lm_finetune \
  --recall_k 30 \
  --pos_K 30 \
  --neg_K 30 \
  --random_neg_ratio 0.5 \
  --keg_neg all \
  --test_open \
  --link_prediction \
  --add_definition \
(2) for the UMLS dataset
python3 run_args.py \
  --model_name sapbert \
  --max_epoch 10 \
  --batch_size 16 \
  --early_stop 10 \
  --lr 5e-5 \
  --lm_lr 1e-6 \
  --seed 234 \
  --decay_rate 0.99 \
  --weight_decay 0.0005 \
  --lstm_dropout 0.0 \
  --data_dir ./dataset/UMLS+PubMed \
  --out_dir ./checkpoint/UMLS+PubMed \
  --valid_step 5000 \
  --use_lm_finetune \
  --recall_k 30 \
  --pos_K 30 \
  --neg_K 30 \
  --random_neg_ratio 0.5 \
  --kge_neg all \
  --link_prediction \
  --add_definition \

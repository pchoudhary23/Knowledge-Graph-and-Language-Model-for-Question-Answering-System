#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
dt=`date '+%Y%m%d_%H%M%S'`


dataset="obqa"
model="distilroberta-base"

elr="1e-4"
dlr="1e-3"
bs=128
mbs=1
n_epochs=20
num_relation=38 #(17 +2) * 2: originally 17, add 2 relation types (QA context -> Q node; QA context -> A node), and double because we add reverse edges
inhouse=False
facts=""
k=2 #num of gnn layers
gnndim=200
fc_layer_num=1
echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "inhouse : $inhouse"
echo "fc_layer_num : $fc_layer_num"
echo "******************************"

save_dir_pref='saved_models_k2_fc1_SP_v1'
mkdir -p $save_dir_pref
mkdir -p logs

###### Training ######
for seed in 0; do
  python3 -u qagnn_updated.py --dataset $dataset \
      --encoder $model -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs -mbs $mbs --fp16 true --seed $seed \
      --num_relation $num_relation \
      --inhouse $inhouse \
      --fc_layer_num $fc_layer_num \
      --n_epochs $n_epochs --max_epochs_before_stop 50  \
      --train_adj data/${dataset}/graph/train.graph${facts}.adj.pk \
      --dev_adj   data/${dataset}/graph/dev.graph${facts}.adj.pk \
      --test_adj  data/${dataset}/graph/test.graph${facts}.adj.pk \
      --train_statements data/${dataset}/statement/train-fact.statement.jsonl \
      --dev_statements   data/${dataset}/statement/dev-fact.statement.jsonl \
      --test_statements  data/${dataset}/statement/test-fact.statement.jsonl \
      --save_dir ${save_dir_pref}/${dataset}/enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args \
  > logs/train_${dataset}__enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt}.log.txt
done

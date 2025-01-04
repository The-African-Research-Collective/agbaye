#!/bin/bash

function train(){
    python -m agbaye.lid.train_fasttext_model train-model \
    --train_dataset "data/lid_training_data/tiny.tsv" \
    --model_dir "models" \
    --threshold 1e-4 \
    --lr 0.1 \
    --lrupdaterate 100 \
    --bucket 2_000_000 \
    --ws 5 \
    --dim 256 \
    --loss ns \
    --neg 5 \
    --seed 84 \
    --epoch 1 \
    --wordngrams 2 \
    --minn 3 \
    --maxn 6 \
    --report_to_wandb
}

function evaluate() {
    python -m agbaye.lid.train_fasttext_model evaluate-model \
    --model_name_or_path "OpenLID" \
    --eval_dataset "data/lid_training_data/lid_eval.tsv" \
    --report_to_wandb
}

evaluate
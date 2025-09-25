#!/bin/bash

function train(){
    python -m agbaye.lid.training.train_fasttext train-model \
    --train-dataset "/home/theyorubayesian/scratch/agbaye/data/all_glotlid_data.tsv" \
    --validation-dataset-or-dir /store/shared/flores_plus/devtest \
    --output-dir "/store/shared/wmqds/experiments/20250923224618/" \
    --at-k 1 \
    --at-k 3 \
    --at-k 5 \
    --at-k 10 \
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
    --threads 4
}

function evaluate() {
    python -m agbaye.lid.train_fasttext_model evaluate-model \
    --model_name_or_path "OpenLID" \
    --output_dir \
    --eval_dataset "data/lid_training_data/lid_eval.tsv" \
    --report_to_wandb \
    --wandb_entity taresco-hq \
    --wandb_project wmqds-lid
}

# evaluate
# train
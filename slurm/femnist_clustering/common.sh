#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate flower

export PYTHONPATH=$PYTHONPATH:.
export HF_DATASETS_IN_MEMORY_MAX_SIZE=8589934592
export HF_DATASETS_CACHE=/home/radovib/.cache/huggingface/data/

batch_size=32
proximal_mu=0.001
fraction_fit=0.1
n_holdout_clients=1000

COMMON_ARGS="dataset=femnist partitioning=natural train_config.fraction_fit=$fraction_fit train_config.proximal_mu=$proximal_mu train_config.batch_size=$batch_size final_evaluation.n_holdout_clients=$n_holdout_clients"

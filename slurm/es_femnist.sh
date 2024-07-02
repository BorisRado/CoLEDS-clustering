#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=28:00:00
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=1G
#SBATCH --out=logs/es_femnist.txt


source ../.venv/flower/bin/activate

FOLDER=es_femnist

echo "submitting..."
srun python3 -u scripts/train_es_model.py \
    train_config.batch_size=32     \
    dataset=femnist                         \
    partitioning=natural                    \
    train_config.fraction_fit=0.1           \
    train_config.cem_evaluation_freq=10     \
    train_config.tot_rounds=1               \
    train_config.ae_weight=1.0              \
    wandb.log_to_wandb=false                \
    +temp_run_id=$FOLDER

FOLDER=data/raw/es_femnist

for n_clusters in 1 2 4 6 8; do
    echo "submitting..."
    python -u scripts/train_clustering.py \
            folder=$FOLDER                         \
            train_config.n_clusters=$n_clusters    \
            dataset=femnist                        \
            partitioning=natural                   \
            train_config.fraction_fit=0.1          \
            +general.n_holdout_clients=1000
done

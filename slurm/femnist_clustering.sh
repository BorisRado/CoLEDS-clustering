#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=1G
#SBATCH --out=logs/clustering_femnist_set.txt


batch_size=32
temperature=0.2
fraction_fit=0.5

srun python -u scripts/train_cl_model.py \
    train_config.batch_size=$batch_size     \
    model=set2set                           \
    train_config.temperature=$temperature   \
    train_config.fraction_fit=$fraction_fit \
    dataset=femnist                         \
    partitioning=natural                    \
    train_config.n_iterations=300           \
    wandb.log_to_wandb=false                \
    +temp_run_id=set2set_femnist

FOLDER=data/raw/set2set_femnist

for n_clusters in 1 2 4 6 8; do
    echo "submitting..."
    srun python -u scripts/train_clustering.py \
        folder=$FOLDER                         \
        train_config.n_clusters=$n_clusters    \
        dataset=femnist                        \
        partitioning=natural                   \
        train_config.fraction_fit=0.1          \
        +general.n_holdout_clients=1500
done

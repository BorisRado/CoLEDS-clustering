#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=48
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=1G
#SBATCH --out=logs/clustering_wdc.txt


source ../.venv/flower/bin/activate

FOLDER=data/raw/leij8i5o  # For PointNet
FOLDER=data/raw/znyweltl

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

wait

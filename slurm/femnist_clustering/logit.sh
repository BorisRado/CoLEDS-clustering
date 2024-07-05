#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=1G
#SBATCH --out=logs/clustering_femnist_logit.txt


source slurm/femnist_clustering/common.sh


srun python -u scripts/evaluate_logit_cem.py  \
    dataset=femnist                           \
    partitioning=natural                      \
    cem.public_dataset_name=fashion_mnist     \
    +temp_run_id=logit_femnist +dry_run=true


FOLDER=data/raw/logit_femnist

for n_clusters in 1 2 4 6 8; do
    echo "submitting..."
    srun python -u scripts/train_clustering.py \
        folder=$FOLDER                         \
        train_config.n_clusters=$n_clusters    \
        $COMMON_ARGS

done

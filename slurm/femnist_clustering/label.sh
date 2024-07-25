#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=1G
#SBATCH --out=logs/clustering_femnist_random.txt


source slurm/femnist_clustering/common.sh


srun python -u scripts/eval_trivial_cems.py   \
    dataset=femnist                           \
    partitioning=natural                      \
    cem="random"                               \
    +temp_run_id=random_femnist +dry_run=true


FOLDER=data/raw/random_femnist

for n_clusters in 1 2 4 6 8; do
    echo "submitting..."
    srun python -u scripts/train_clustering.py \
        folder=$FOLDER                         \
        train_config.n_clusters=$n_clusters    \
        $COMMON_ARGS

done

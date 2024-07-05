#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=1G
#SBATCH --out=logs/clustering_femnist_cl.txt


source slurm/femnist_clustering/common.sh


srun python -u scripts/train_cl_model.py    \
    model=set2set                           \
    dataset=femnist                         \
    partitioning=natural                    \
    train_config.batch_size=32              \
    train_config.temperature=0.2            \
    train_config.fraction_fit=0.25          \
    train_config.n_iterations=250           \
    +temp_run_id=cl_femnist_set2set


FOLDER=data/raw/cl_femnist_set2set

for n_clusters in 1 2 4 6 8; do
    echo "submitting..."
    srun python -u scripts/train_clustering.py \
        folder=$FOLDER                         \
        train_config.n_clusters=$n_clusters    \
        $COMMON_ARGS

done

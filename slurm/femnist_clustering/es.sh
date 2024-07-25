#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=1G
#SBATCH --out=logs/clustering_femnist_es.txt


source slurm/femnist_clustering/common.sh


srun python -u scripts/train_es_model.py    \
    dataset=femnist                         \
    partitioning=natural                    \
    cem.reduction_stats="mean"              \
    train_config.fraction_fit=0.1           \
    train_config.ae_weight=0.5              \
    train_config.cem_evaluation_freq=25     \
    +temp_run_id=es_femnist


FOLDER=data/raw/es_femnist

for n_clusters in 1 2 4 6 8; do
    echo "submitting..."
    srun python -u scripts/train_clustering.py \
        folder=$FOLDER                         \
        train_config.n_clusters=$n_clusters    \
        $COMMON_ARGS

done

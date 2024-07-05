#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1G
#SBATCH --gpus-per-task=1
#SBATCH --out=logs/embedding_quality_es.txt


source ~/miniforge3/etc/profile.d/conda.sh
conda activate flower

export PYTHONPATH=$PYTHONPATH:.
export HF_DATASETS_IN_MEMORY_MAX_SIZE=8589934592
export HF_DATASETS_CACHE=/home/radovib/.cache/huggingface/data/

dirichlet_alpha=0.2
fraction_fit=0.5

for DATASET in mnist cifar10 cifar100; do

    if [ "$DATASET" = "cifar100" ]; then
        partition_by="fine_label"
    else
        partition_by="label"
    fi

    for ae_weight in 0.0 0.5 1.0; do
        for statistics in mean std covariance correlation; do
            echo "submitting..."
            srun --exclusive -N1 --ntasks=1 --cpus-per-task=24 --mem-per-cpu=1G  --gpus-per-task=1 python -u scripts/train_es_model.py \
                partitioning.alpha=$dirichlet_alpha     \
                cem.reduction_stats=$statistics         \
                train_config.fraction_fit=$fraction_fit \
                train_config.ae_weight=$ae_weight       \
                dataset=$DATASET                        \
                partitioning.partition_by=$partition_by \
                +temp_run_id="es_${DATASET}_${statistics}_ae${ae_weight}" &
        done
        wait
    done
done



wait

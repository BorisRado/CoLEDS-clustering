#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --out=logs/embedding_quality_logit.txt


source ~/miniforge3/etc/profile.d/conda.sh
conda activate flower
export PYTHONPATH=$PYTHONPATH:.
export HF_DATASETS_IN_MEMORY_MAX_SIZE=8589934592
export HF_DATASETS_CACHE=/home/radovib/.cache/huggingface/data/

for DATASET in mnist cifar10 cifar100; do
    if [ "$DATASET" = "cifar100" ]; then
        partition_by="fine_label"
    else
        partition_by="label"
    fi

    if [ "$DATASET" = "mnist" ]; then
        public_dataset="fashion_mnist"
    elif [ "$DATASET" = "cifar10" ]; then
        public_dataset="cifar100"
    elif [ "$DATASET" = "cifar100" ]; then
        public_dataset="cifar10"
    else
        echo "Unknown dataset..."
        exit 1
    fi

    echo "submitting..."
    srun python -u scripts/evaluate_logit_cem.py \
        partitioning.alpha=0.2                    \
        cem.ft_epochs=1                           \
        dataset=$DATASET                          \
        partitioning.partition_by=$partition_by   \
        cem.public_dataset_name=$public_dataset   \
        +temp_run_id="logit_${DATASET}"

done

wait

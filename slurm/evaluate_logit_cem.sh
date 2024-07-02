#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --out=logs/logit_all.txt


source ../.venv/flower/bin/activate


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
    srun --ntasks=1 --cpus-per-task=32 --mem-per-cpu=2G python -u scripts/evaluate_logit_cem.py \
        partitioning.alpha=0.2                    \
        cem.ft_epochs=1                           \
        dataset=$DATASET                          \
        partitioning.partition_by=$partition_by   \
        cem.public_dataset_name=$public_dataset   \
        wandb.log_to_wandb=true                   \
        wandb.loggin_keys=[partitioning.alpha,cem.ft_epochs]

done

wait

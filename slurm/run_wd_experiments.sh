#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --out=logs/wd_cifar100.txt


source ../.venv/flower/bin/activate

DATASET=cifar100

if [ "$DATASET" = "cifar100" ]; then
  partition_by="fine_label"
else
  partition_by="label"
fi

for dirichlet_alpha in 0.2; do
    for ft_epochs in 1 2; do
        echo "submitting..."
        srun --ntasks=1 --cpus-per-task=32 --mem-per-cpu=2G python -u scripts/train_wd_model.py \
            partitioning.alpha=$dirichlet_alpha     \
            cem.ft_epochs=$ft_epochs                \
            dataset=$DATASET                        \
            partitioning.partition_by=fine_label    \
            wandb.loggin_keys=[partitioning.alpha,cem.ft_epochs]
    done
done

wait

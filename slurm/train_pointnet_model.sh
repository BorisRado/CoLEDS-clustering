#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/cl_pointnet_cifar100.txt


source ../.venv/flower/bin/activate


DATASET=cifar100

if [ "$DATASET" = "cifar100" ]; then
  partition_by="fine_label"
else
  partition_by="label"
fi

for batch_size in 32 64; do
    for dirichlet_alpha in 0.2; do
        for statistics in mean; do
            for temperature in 0.05 0.2 0.5 1.0; do
                for head_sizes in \[32\]; do
                    for fraction_fit in 0.5; do
                        echo "submitting..."
                        srun --ntasks=1 --cpus-per-task=16 --mem-per-cpu=2G python -u scripts/train_cl_model.py \
                            train_config.batch_size=$batch_size     \
                            partitioning.alpha=$dirichlet_alpha     \
                            model.reduction_stats=$statistics       \
                            model.head_sizes=$head_sizes            \
                            train_config.temperature=$temperature   \
                            train_config.fraction_fit=$fraction_fit \
                            dataset=$DATASET                        \
                            partitioning.partition_by=fine_label    \
                            wandb.loggin_keys=[train_config.batch_size,partitioning.alpha,train_config.temperature] &
                    done
                done
            done
        done
    done
done

wait

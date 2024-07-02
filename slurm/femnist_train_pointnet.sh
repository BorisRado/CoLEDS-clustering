#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/cl_pointnet_femnist.txt


source ../.venv/flower/bin/activate

for batch_size in 32; do
        for statistics in mean; do
            for temperature in 0.5; do
                for head_sizes in \[32\]; do
                    for fraction_fit in 0.2; do
                        echo "submitting..."
                        srun --ntasks=1 --cpus-per-task=16 --mem-per-cpu=2G python -u scripts/train_cl_model.py \
                            train_config.batch_size=$batch_size     \
                            model.reduction_stats=$statistics       \
                            model.head_sizes=$head_sizes            \
                            train_config.temperature=$temperature   \
                            train_config.fraction_fit=$fraction_fit \
                            dataset=femnist                         \
                            partitioning=natural                    \
                            train_config.n_iterations=200           \
                            wandb.loggin_keys=[train_config.batch_size,train_config.temperature]
                    done
                done
            done
        done
    done
done

wait

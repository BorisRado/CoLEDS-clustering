#!/bin/bash

#SBATCH --ntasks=9
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --out=logs/cl_transformer_encoder.txt


source ../.venv/flower/bin/activate

# batch size: 16 32 64
# dirichlet alpha: 0.2 0.5
# statistics: mean, max, std
# temperature: 0.02 0.2
# head sizes [[], 32 64 128]
# fraction fit: [0.1 0.5 1.0]

# 3 * 2 * 3 * 2 * 4 * 3
# 6 * 6 * 12
# 432 / 16 = 27

fraction_fit=0.5
for batch_size in 32 48 64; do
    for dirichlet_alpha in 0.2 0.5 1.0; do
        for temperature in 0.05 0.2 0.5 1.0 2.0; do
            echo "submitting..."
            srun --ntasks=1 python -u scripts/train_cl_model.py \
                model=te                                \
                train_config.batch_size=$batch_size     \
                partitioning.alpha=$dirichlet_alpha     \
                train_config.temperature=$temperature   \
                train_config.fraction_fit=$fraction_fit \
                wandb.loggin_keys=[train_config.batch_size,partitioning.alpha,train_config.temperature,train_config.fraction_fit] &
        done
    done
done

wait

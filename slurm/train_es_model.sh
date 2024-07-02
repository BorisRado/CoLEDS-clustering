#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --gpus=1
#SBATCH --out=logs/es_cifar100.txt


source ../.venv/flower/bin/activate

DATASET=cifar100

if [ "$DATASET" = "cifar100" ]; then
  partition_by="fine_label"
else
  partition_by="label"
fi

for dirichlet_alpha in 0.2; do
    for ae_weight in 0.0 0.5 1.0; do
        for statistics in mean std covariance correlation; do
            for fraction_fit in 1.0; do
                echo "submitting..."
                srun python -u scripts/train_es_model.py \
                    partitioning.alpha=$dirichlet_alpha     \
                    cem.reduction_stats=$statistics         \
                    train_config.fraction_fit=$fraction_fit \
                    train_config.ae_weight=$ae_weight       \
                    dataset=$DATASET                        \
                    partitioning.partition_by=fine_label    \
                    wandb.loggin_keys=[cem.reduction_stats,train_config.ae_weight]
            done
        done
    done
done

wait

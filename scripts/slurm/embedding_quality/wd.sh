#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --out=logs/embedding_quality_wd.txt


source scripts/slurm/embedding_quality/common.sh


EVAL_ITERATIONS=4
FL_EPOCHS_PER_ITERATION=25

COMMON_ARGS="
    partitioning.alpha=$DIRICHLET_ALPHA                    \
    wandb.log_to_wandb=true                                \
    experiment.name=$EXP_NAME                              \
    general.eval_iterations=$EVAL_ITERATIONS               \
    general.epochs_per_iteration=$FL_EPOCHS_PER_ITERATION
"


for seed in "${SEEDS[@]}"; do
for DATASET in mnist fashion_mnist cifar10 cifar100 cinic10; do
for ft_epochs in 1; do
for optimizer in adam; do

set_partition_by "$DATASET"

srun -Q -N1 --ntasks=1  python -u scripts/py/evaluate_wd_profiling.py     \
    cem.ft_epochs=$ft_epochs                                              \
    dataset=$DATASET                                                      \
    partitioning.partition_by=$partition_by                               \
    general.seed=$seed                                                    \
    optimizer=$optimizer

done
done
done
done

wait

#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/pacfl_embedding_quality.txt

source scripts/slurm/embedding_quality/common.sh

COMMON_ARGS="
    partitioning.alpha=$DIRICHLET_ALPHA  \
    wandb.log_to_wandb=true              \
    experiment.name=$EXP_NAME
"

for seed in "${SEEDS[@]}"; do
for DATASET in mnist fashion_mnist cifar10 cifar100 cinic10; do

set_partition_by "$DATASET"

python -u scripts/py/evaluate_pacfl_profiling.py   \
    $COMMON_ARGS                                   \
    dataset=$DATASET                               \
    partitioning.partition_by=$partition_by        \
    general.seed=$seed

done
done

#!/bin/bash -l

#SBATCH --ntasks=5
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/embedding_quality_logit.txt


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


PUBLIC_DATASET_SIZE=1000

for seed in "${SEEDS[@]}"; do
for DATASET in mnist fashion_mnist cifar10 cifar100 cinic10; do
for ft_epochs in 1; do
for optimizer in adam; do

set_partition_by "$DATASET"
case "$DATASET" in
    mnist) PUBLIC_DATASET="fashion_mnist" ;;
    fashion_mnist) PUBLIC_DATASET="mnist" ;;
    cifar10) PUBLIC_DATASET="cifar100" ;;
    cifar100) PUBLIC_DATASET="cifar10" ;;
    cinic10) PUBLIC_DATASET="cifar100" ;;
    *) echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

run_bg python scripts/py/evaluate_logit_profiling.py     \
    $COMMON_ARGS                                         \
    profiling.ft_epochs=$ft_epochs                       \
    dataset=$DATASET                                     \
    dataset@public_dataset=$PUBLIC_DATASET               \
    profiling.public_dataset_size=$PUBLIC_DATASET_SIZE   \
    partitioning.partition_by=$partition_by              \
    general.seed=$seed                                   \
    optimizer=$optimizer

done
done
done
done

wait

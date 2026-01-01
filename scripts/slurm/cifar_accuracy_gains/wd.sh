#!/bin/bash -l

#SBATCH --ntasks=2
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/wd_cifar10_accuracy.txt

source scripts/slurm/cifar_accuracy_gains/common.sh


for seed in "${SEEDS[@]}"; do

run_bg python scripts/py/evaluate_wd_profiling.py   \
    dataset=cifar10                                 \
    partitioning=dirichlet                          \
    general.seed=$seed                              \
    hydra.run.dir=${HOME_FOLDER}/wd_seed_${seed}    \
    +dry_run=true

done

wait

for seed in "${SEEDS[@]}"; do

profiler_folder=${HOME_FOLDER}/wd_seed_${seed}
submit_experiment "$seed" "$profiler_folder"

done

wait

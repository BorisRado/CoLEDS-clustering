#!/bin/bash -l

#SBATCH --ntasks=2
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --out=logs/wd_femnist_accuracy.txt

source scripts/slurm/accuracy_gains/common.sh


for seed in "${SEEDS[@]}"; do

run_async python -u scripts/py/evaluate_wd_profiling.py      \
    dataset=femnist                                          \
    partitioning=natural                                     \
    general.seed=$seed                                       \
    hydra.run.dir=${HOME_FOLDER}/wd_seed_${seed}             \
    +dry_run=true

done

wait

for seed in "${SEEDS[@]}"; do
for num_clusters in "${TEST_NUM_CLUSTERS[@]}"; do

run_async python -u scripts/py/train_with_clustering.py   \
    general.seed=$seed                                    \
    ${CLUSTERING_MODEL_TRAINING_ARGS}                     \
    train_config.n_clusters=$num_clusters                 \
    folder=${HOME_FOLDER}/wd_seed_${seed}

done
done

wait

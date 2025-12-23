#!/bin/bash -l

#SBATCH --ntasks=3
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/femnist_wo_clustering.txt

source scripts/slurm/accuracy_gains/common.sh

for seed in "${SEEDS[@]}"; do

run_bg python scripts/py/train_classification_model.py   \
    $DATASET_ARGS                                        \
    general.seed=$seed                                   \
    ${CLUSTERING_MODEL_TRAINING_ARGS}                    \
    hydra.run.dir=${HOME_FOLDER}/without_clustering_seed_${seed}

done

wait

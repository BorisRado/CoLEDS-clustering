#!/bin/bash -l

#SBATCH --ntasks=5
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --out=logs/femnist_wo_clustering.txt

source scripts/slurm/accuracy_gains/common.sh

for seed in "${SEEDS[@]}"; do

srun -Q -N1 --ntasks=1 python scripts/py/train_classification_model.py   \
    $DATASET_ARGS                                                        \
    general.seed=$seed                                                   \
    ${CLUSTERING_MODEL_TRAINING_ARGS}                                    \
    hydra.run.dir=${HOME_FOLDER}/without_clustering_seed_${seed} &

done

wait

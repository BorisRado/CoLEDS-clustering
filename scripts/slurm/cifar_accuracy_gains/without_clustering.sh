#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/cifar10_wo_clustering.txt

source scripts/slurm/cifar_accuracy_gains/common.sh

for seed in "${SEEDS[@]}"; do
for proximal_mu in "${TEST_PROXIMAL_MUS[@]}"; do
for num_ho_clients in 0 50; do

run_bg python scripts/py/train_classification_model.py   \
    $DATASET_ARGS                                        \
    general.seed=$seed                                   \
    ${CLUSTERING_MODEL_TRAINING_ARGS}                    \
    train_config.proximal_mu=$proximal_mu                \
    final_evaluation.n_holdout_clients=$num_ho_clients   \
    hydra.run.dir=${HOME_FOLDER}/without_clustering_seed_${seed}_mu_${proximal_mu}_ho_${num_ho_clients}

done
done
done

wait

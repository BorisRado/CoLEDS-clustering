#!/bin/bash -l

#SBATCH --ntasks=4
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/label_femnist_accuracy.txt

source scripts/slurm/accuracy_gains/common.sh


for seed in "${SEEDS[@]}"; do
for num_clusters in "${TEST_NUM_CLUSTERS[@]}"; do

experiment_folder=${HOME_FOLDER}/label_seed_${seed}

# write the bare minimum YAML file needed for train_with_clustering to run
mkdir -p "$experiment_folder"
mkdir -p "$experiment_folder/.hydra"

cat > "$experiment_folder/.hydra/config.yaml" <<'YAML'
profiler: label
dataset:
    dataset_name: femnist
YAML

run_bg python scripts/py/train_with_clustering.py   \
    general.seed=$seed                              \
    ${CLUSTERING_MODEL_TRAINING_ARGS}               \
    train_config.n_clusters=$num_clusters           \
    folder=$experiment_folder

done
done

wait

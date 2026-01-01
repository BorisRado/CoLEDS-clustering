#!/bin/bash -l

#SBATCH --ntasks=2
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/label_cifar10_accuracy.txt

source scripts/slurm/cifar_accuracy_gains/common.sh


for seed in "${SEEDS[@]}"; do

profiler_folder=${HOME_FOLDER}/label_seed_${seed}

# write the bare minimum YAML file needed for train_with_clustering to run
mkdir -p "$profiler_folder"
mkdir -p "$profiler_folder/.hydra"

cat > "$profiler_folder/.hydra/config.yaml" <<'YAML'
profiler: label
dataset:
    dataset_name: cifar10
YAML

submit_experiment "$seed" "$profiler_folder"
done

wait

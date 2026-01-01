#!/bin/bash -l

#SBATCH --ntasks=4
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/coleds_cifar10.txt


source scripts/slurm/cifar_accuracy_gains/common.sh


MAX_EPOCHS=20
ITERATIONS_PER_EPOCH=100
PATIENCE=10
NUM_CLIENT_UPDATES=1
TEMPERATURE=0.2
COLEDS_MODEL=set2set
VIEW_SIZE=8
FRACTION_FIT=0.25

COMMON_ARGS="
    $DATASET_ARGS
    general.max_epochs=$MAX_EPOCHS
    general.patience=$PATIENCE
    train_config.num_iterations=$ITERATIONS_PER_EPOCH
    train_config.fraction_fit=$FRACTION_FIT
    train_config.batch_size=$VIEW_SIZE
    experiment.save_model=true
    model.processing_steps=2
    model.num_layers=1
"


# start by training the coleds profiling model
for seed in "${SEEDS[@]}"; do

profiler_folder="$HOME_FOLDER/coleds_seed_${seed}"
echo "Training COLEDS profiling model -- saving to ${profiler_folder}."
rm -rf $profiler_folder

run_bg python scripts/py/train_coleds.py                  \
    $COMMON_ARGS                                          \
    model=$COLEDS_MODEL                                   \
    train_config.num_client_updates=$NUM_CLIENT_UPDATES   \
    train_config.temperature=$TEMPERATURE                 \
    general.seed=$seed                                    \
    hydra.run.dir=$profiler_folder

done

wait

echo "Trained COLEDS -- continuing with clustering experiments."
for seed in "${SEEDS[@]}"; do

profiler_folder="$HOME_FOLDER/coleds_seed_${seed}"
submit_experiment "$seed" "$profiler_folder"

done

wait

#!/bin/bash -l

mamba activate slower

export PYTHONPATH=$PYTHONPATH:.
export PYTHONUNBUFFERED=1

SEEDS=(4 8 15 16 23) # 4 8 15 16 23 42
TEST_NUM_CLUSTERS=(2 4 8 16 32 64)

DATASET_ARGS="dataset=femnist partitioning=natural"
HOME_FOLDER="outputs/accuracy_gains"

# configuration for model training with clustering
CLASSIFICATION_MODEL="simple_net"
OPTIMIZER="adam"
BATCH_SIZE=32
PROXIMAL_MU=0.001
EPOCHS=250
NUM_HOLDOUT_CLIENTS=1097
FRACTION_FIT=0.05
FRACTION_EVALUATE=0.5
EVALUATION_FREQUENCY=5

CLUSTERING_MODEL_TRAINING_ARGS="
    model=${CLASSIFICATION_MODEL}
    optimizer=${OPTIMIZER}
    train_config.batch_size=${BATCH_SIZE}
    train_config.proximal_mu=${PROXIMAL_MU}
    train_config.num_epochs=${EPOCHS}
    final_evaluation.n_holdout_clients=${NUM_HOLDOUT_CLIENTS}
    strategy.fraction_fit=${FRACTION_FIT}
    strategy.fraction_evaluate=${FRACTION_EVALUATE}
    strategy.evaluation_frequency=${EVALUATION_FREQUENCY}
    client_resources.num_cpus=2
    client_resources.num_gpus=0.125
"

# Helper function to run commands asynchronously with srun
run_bg() {
    srun -Q --ntasks=1 \
        --cpus-per-task=${SLURM_CPUS_PER_TASK} \
        --mem-per-cpu=${SLURM_MEM_PER_CPU} \
        --gpus-per-task=${SLURM_GPUS_PER_TASK} \
        "$@" &
}

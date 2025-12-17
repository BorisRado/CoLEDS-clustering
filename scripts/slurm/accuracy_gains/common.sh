#!/bin/bash -l

mamba activate slower

export PYTHONPATH=$PYTHONPATH:.

SEEDS=(4 8 15 16 23) # 4 8 15 16 23 42

DATASET_ARGS="dataset=femnist partitioning=natural"
HOME_FOLDER="outputs/accuracy_gains"

# configuration for model training with clustering
CLASSIFICATION_MODEL="simple_net"
OPTIMIZER="adam"
BATCH_SIZE=32
PROXIMAL_MU=0.001
EPOCHS=250
NUM_HOLDOUT_CLIENTS=1097
FRACTION_FIT=0.04
FRACTION_EVALUATE=0.4
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
"

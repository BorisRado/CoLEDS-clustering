#!/bin/bash -l

mamba activate slower

export PYTHONPATH=$PYTHONPATH:.
export PYTHONUNBUFFERED=1

SEEDS=(4 8 15) # 4 8 15 16 23 42
TEST_NUM_CLUSTERS=(2 4 8 16 32 64)
TEST_CLUSTERING_ALGORITHMS=("kmeans" "hierarchical_ward" "hierarchical_complete")
TEST_PROXIMAL_MUS=(-1.0 0.001)

DATASET_ARGS="dataset=cifar10 partitioning=dirichlet"
HOME_FOLDER="outputs/cifar10_ablation"

# configuration for model training with clustering
CLASSIFICATION_MODEL="simple_net"
OPTIMIZER="adam"
BATCH_SIZE=32
PROXIMAL_MU=-1
EPOCHS=250
NUM_HOLDOUT_CLIENTS=50
FRACTION_FIT=0.25
FRACTION_EVALUATE=1.0
EVALUATION_FREQUENCY=5

CLUSTERING_MODEL_TRAINING_ARGS="
    model=${CLASSIFICATION_MODEL}
    optimizer=${OPTIMIZER}
    train_config.batch_size=${BATCH_SIZE}
    train_config.num_epochs=${EPOCHS}
    strategy.fraction_fit=${FRACTION_FIT}
    strategy.fraction_evaluate=${FRACTION_EVALUATE}
    strategy.evaluation_frequency=${EVALUATION_FREQUENCY}
    client_resources.num_cpus=2
    client_resources.num_gpus=0.125
    ${DATASET_ARGS}
"

# Helper function to run commands asynchronously with srun
run_bg() {
    srun -Q --ntasks=1 \
        --cpus-per-task=${SLURM_CPUS_PER_TASK} \
        --mem-per-cpu=${SLURM_MEM_PER_CPU} \
        --gpus-per-task=${SLURM_GPUS_PER_TASK} \
        "$@" &
}


get_clustering_experiment_folder() {
    local profiler_folder="$1" clustering_algorithm="$2" num_clusters="$3" proximal_mu="$4" n_ho_clients="$5"
    local folder_name="$profiler_folder/clustering_${clustering_algorithm}_clusters_${num_clusters}_mu_${proximal_mu}_ho_${n_ho_clients}"
    if [ ! -d "$folder_name" ]; then
        mkdir -p "$folder_name"
    fi
    echo "$folder_name"
}


submit_experiment() {
    local seed="$1" profiler_folder="$2"

    # experiments with different clustering algorithms
    proximal_mu=0.001
    n_ho_clients=0
    for num_clusters in "${TEST_NUM_CLUSTERS[@]}"; do
    for clustering_algorithm in "${TEST_CLUSTERING_ALGORITHMS[@]}"; do

    output_folder=$(get_clustering_experiment_folder "$profiler_folder" "$clustering_algorithm" "$num_clusters" "$proximal_mu" "$n_ho_clients")

    run_bg python scripts/py/train_with_clustering.py        \
        general.seed=$seed                                   \
        ${CLUSTERING_MODEL_TRAINING_ARGS}                    \
        train_config.n_clusters=$num_clusters                \
        final_evaluation.n_holdout_clients=$n_ho_clients     \
        folder=${profiler_folder}                            \
        hydra.run.dir=$output_folder                         \
        clustering.algorithm=$clustering_algorithm           \
        train_config.proximal_mu=$proximal_mu
    done
    done

    # experiments with different mus
    clustering_algorithm="kmeans"
    for num_clusters in "${TEST_NUM_CLUSTERS[@]}"; do
    for proximal_mu in "${TEST_PROXIMAL_MUS[@]}"; do

    output_folder=$(get_clustering_experiment_folder "$profiler_folder" "$clustering_algorithm" "$num_clusters" "$proximal_mu" "$n_ho_clients")

    n_ho_clients=${NUM_HOLDOUT_CLIENTS}
    run_bg python scripts/py/train_with_clustering.py        \
        general.seed=$seed                                   \
        ${CLUSTERING_MODEL_TRAINING_ARGS}                    \
        train_config.n_clusters=$num_clusters                \
        final_evaluation.n_holdout_clients=$n_ho_clients     \
        folder=${profiler_folder}                            \
        hydra.run.dir=$output_folder                         \
        clustering.algorithm=$clustering_algorithm           \
        train_config.proximal_mu=$proximal_mu
    done
    done


}

#!/bin/bash -l

#SBATCH --ntasks=6
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/es_femnist_accuracy.txt

source scripts/slurm/accuracy_gains/common.sh

PROFILER_FL_EPOCHS=250
TRAIN_PROF_FF=0.02
TRAIN_PROF_EVAL_FREQ=125
TRAIN_PROF_EVAL_FRAC=0.2

COMMON_ARGS="
    $DATASET_ARGS
    general.eval_iterations=1
    general.epochs_per_iteration=$PROFILER_FL_EPOCHS
    strategy.fraction_fit=$TRAIN_PROF_FF
    strategy.evaluation_frequency=$TRAIN_PROF_EVAL_FREQ
    strategy.fraction_evaluate=$TRAIN_PROF_EVAL_FRAC
"


train the encoder model
for seed in "${SEEDS[@]}"; do
for encoder_model in simple_net beta_vae; do
run_bg python scripts/py/evaluate_es_profiling.py    \
    $COMMON_ARGS                                     \
    general.seed=$seed                               \
    model=$encoder_model                             \
    hydra.run.dir=${HOME_FOLDER}/es_model_${encoder_model}_seed_${seed}

done
done

wait

# get the accuracy with clustering

for seed in "${SEEDS[@]}"; do
for encoder_model in simple_net beta_vae; do
for num_clusters in "${TEST_NUM_CLUSTERS[@]}"; do

run_bg python scripts/py/train_with_clustering.py   \
    general.seed=$seed                              \
    ${CLUSTERING_MODEL_TRAINING_ARGS}               \
    train_config.n_clusters=$num_clusters           \
    folder=${HOME_FOLDER}/es_model_${encoder_model}_seed_${seed}

done
done
done

wait

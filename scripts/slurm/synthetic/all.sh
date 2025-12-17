#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --gpus=1
#SBATCH --out=logs/synthetic.txt


mamba activate slower
export PYTHONPATH=$PYTHONPATH:.

BASE_FOLDER="outputs/synthetic"

rm -drf $BASE_FOLDER

TRAINING_TARGET_PROBABILITY=0.8
CLIENT_DATASET_SIZE=40
N_DATASETS=600
N_EVAL_DATASETS=200
SEED=1602

COMMON_ARGS="
    dataset=synthetic
    dataset.p=$TRAINING_TARGET_PROBABILITY
    dataset.dataset_size=$CLIENT_DATASET_SIZE
    dataset.n_datasets=$N_DATASETS
    general.seed=$SEED
"


# COLEDS
COLEDS_COMMON_ARGS="
    $COMMON_ARGS
    train_config.num_iterations=60
    general.max_epochs=6
    experiment.save_model=true
    train_config.batch_size=16
    train_config.num_client_updates=2
"
srun python scripts/py/train_coleds.py $COLEDS_COMMON_ARGS \
    model=set2set \
    hydra.run.dir=$BASE_FOLDER/coleds_s2s

srun python scripts/py/train_coleds.py $COLEDS_COMMON_ARGS \
    model=clmean \
    hydra.run.dir=$BASE_FOLDER/coleds_clmean
unset COLEDS_COMMON_ARGS

# EMBEDDING SPACE PROFILING
ES_COMMON_ARGS="
    $COMMON_ARGS
    general.eval_iterations=4
    general.epochs_per_iteration=20
    strategy.fraction_fit=0.2
"
srun python scripts/py/evaluate_es_profiling.py \
    $ES_COMMON_ARGS \
    model=simple_net \
    hydra.run.dir=$BASE_FOLDER/esc_clf
srun python scripts/py/evaluate_es_profiling.py \
    $ES_COMMON_ARGS \
    model=beta_vae \
    hydra.run.dir=$BASE_FOLDER/esc_vae
unset ES_COMMON_ARGS



# WEIGHT-DIFFERENCE PROFILING
srun python scripts/py/evaluate_wd_profiling.py $COMMON_ARGS hydra.run.dir=$BASE_FOLDER/wd +dry_run=true
srun python scripts/py/evaluate_wd_profiling.py $COMMON_ARGS \
    hydra.run.dir=$BASE_FOLDER/wd_training \
    general.eval_iterations=1 \
    general.epochs_per_iteration=50


echo ""
echo ""
echo ""
echo ""
echo "STARTING VISUALIZATION"

# Note that seed must be different here to avoid generating the same clients as in training
VISUAL_GENERATION_ARGS="
    general.seed=1907
    final_evaluation.dataset_size=$CLIENT_DATASET_SIZE
    final_evaluation.p=0.9
    final_evaluation.n_holdout_clients=$N_EVAL_DATASETS
"

srun python scripts/py/visualize_synthetic_dataset_profiles.py $VISUAL_GENERATION_ARGS folder=$BASE_FOLDER/coleds_s2s
srun python scripts/py/visualize_synthetic_dataset_profiles.py $VISUAL_GENERATION_ARGS folder=$BASE_FOLDER/coleds_clmean
srun python scripts/py/visualize_synthetic_dataset_profiles.py $VISUAL_GENERATION_ARGS folder=$BASE_FOLDER/esc_vae
srun python scripts/py/visualize_synthetic_dataset_profiles.py $VISUAL_GENERATION_ARGS folder=$BASE_FOLDER/esc_clf
srun python scripts/py/visualize_synthetic_dataset_profiles.py $VISUAL_GENERATION_ARGS folder=$BASE_FOLDER/wd
srun python scripts/py/visualize_synthetic_dataset_profiles.py $VISUAL_GENERATION_ARGS folder=$BASE_FOLDER/wd_training

wait

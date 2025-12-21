#!/bin/bash -l

#SBATCH --ntasks=6
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --out=logs/coleds_femnist.txt


source scripts/slurm/accuracy_gains/common.sh


MAX_EPOCHS=20
ITERATIONS_PER_EPOCH=250
PATIENCE=5
NUM_CLIENT_UPDATES=1
TEMPERATURE=0.2
COLEDS_MODEL=set2set

COMMON_ARGS="
    $DATASET_ARGS
    general.max_epochs=$MAX_EPOCHS                        \
    general.patience=$PATIENCE                            \
    train_config.num_iterations=$ITERATIONS_PER_EPOCH     \
    experiment.save_model=true
"


TEST_BATCH_SIZES=(32 64)
TEST_FRACTION_FIT=(0.1 0.25)

# start by training the coleds profiling model
for seed in "${SEEDS[@]}"; do
for coleds_batch_size in "${TEST_BATCH_SIZES[@]}"; do
for coleds_fraction_fit in "${TEST_FRACTION_FIT[@]}"; do

tmp_folder=${HOME_FOLDER}/coleds_bs_${coleds_batch_size}_ff_${coleds_fraction_fit}_seed_${seed}
rm -rf $tmp_folder

run_bg python scripts/py/train_coleds.py                  \
    $COMMON_ARGS                                          \
    model=$COLEDS_MODEL                                   \
    train_config.fraction_fit=$coleds_fraction_fit        \
    train_config.num_client_updates=$NUM_CLIENT_UPDATES   \
    train_config.temperature=$TEMPERATURE                 \
    general.seed=$seed                                    \
    train_config.batch_size=$coleds_batch_size            \
    hydra.run.dir=$tmp_folder

done
done
done

wait

echo "Trained COLEDS -- continuing with clustering experiments."

# now train the actual model using the coleds profiling model
for seed in "${SEEDS[@]}"; do
for coleds_batch_size in "${TEST_BATCH_SIZES[@]}"; do
for coleds_fraction_fit in "${TEST_FRACTION_FIT[@]}"; do

for num_clusters in "${TEST_NUM_CLUSTERS[@]}"; do

tmp_folder=${HOME_FOLDER}/coleds_bs_${coleds_batch_size}_ff_${coleds_fraction_fit}_seed_${seed}

echo "Running on ${tmp_folder} with ${num_clusters} clusters."
run_bg python scripts/py/train_with_clustering.py   \
    general.seed=$seed                                             \
    train_config.n_clusters=$num_clusters                          \
    ${CLUSTERING_MODEL_TRAINING_ARGS}                              \
    folder=$tmp_folder

done
done
done
done

wait

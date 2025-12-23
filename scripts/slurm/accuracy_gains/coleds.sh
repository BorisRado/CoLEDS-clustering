#!/bin/bash -l

#SBATCH --ntasks=8
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/coleds_femnist.txt


source scripts/slurm/accuracy_gains/common.sh


MAX_EPOCHS=10
ITERATIONS_PER_EPOCH=250
PATIENCE=10
NUM_CLIENT_UPDATES=1
TEMPERATURE=0.2
COLEDS_MODEL=set2set

COMMON_ARGS="
    $DATASET_ARGS
    general.max_epochs=$MAX_EPOCHS
    general.patience=$PATIENCE
    train_config.num_iterations=$ITERATIONS_PER_EPOCH
    experiment.save_model=true
    model.processing_steps=2
    model.num_layers=1
"

make_exp_folder() {
    local bs="$1" ff="$2" save="$3" seed="$4"
    printf "%s/coleds_bs_%s_ff_%s_save_%s_seed_%s" "$HOME_FOLDER" "$bs" "$ff" "$save" "$seed"
}

TEST_BATCH_SIZES=(32)
TEST_FRACTION_FIT=(0.25)

# start by training the coleds profiling model
for seed in "${SEEDS[@]}"; do
for coleds_batch_size in "${TEST_BATCH_SIZES[@]}"; do
for coleds_fraction_fit in "${TEST_FRACTION_FIT[@]}"; do
for always_save in "true" "false"; do

if [ "$always_save" = "true" ]; then
    SAVE_MODEL_ARG="+experiment.always_save_model=true"
else
    SAVE_MODEL_ARG=""
fi

exp_folder=$(make_exp_folder "$coleds_batch_size" "$coleds_fraction_fit" "$always_save" "$seed")
echo "Training COLEDS profiling model -- saving to ${exp_folder}."
rm -rf $exp_folder

run_bg python scripts/py/train_coleds.py                  \
    $COMMON_ARGS                                          \
    $SAVE_MODEL_ARG                                       \
    model=$COLEDS_MODEL                                   \
    train_config.fraction_fit=$coleds_fraction_fit        \
    train_config.num_client_updates=$NUM_CLIENT_UPDATES   \
    train_config.temperature=$TEMPERATURE                 \
    general.seed=$seed                                    \
    train_config.batch_size=$coleds_batch_size            \
    hydra.run.dir=$exp_folder

done
done
done
done

wait

echo "Trained COLEDS -- continuing with clustering experiments."

# now train the actual model using the coleds profiling model
for seed in "${SEEDS[@]}"; do
for coleds_batch_size in "${TEST_BATCH_SIZES[@]}"; do
for coleds_fraction_fit in "${TEST_FRACTION_FIT[@]}"; do
for always_save in "true" "false"; do

for num_clusters in "${TEST_NUM_CLUSTERS[@]}"; do

exp_folder=$(make_exp_folder "$coleds_batch_size" "$coleds_fraction_fit" "$always_save" "$seed")

echo "Running on ${exp_folder} with ${num_clusters} clusters."
run_bg python scripts/py/train_with_clustering.py   \
    general.seed=$seed                                             \
    train_config.n_clusters=$num_clusters                          \
    ${CLUSTERING_MODEL_TRAINING_ARGS}                              \
    folder=$exp_folder

done

done
done
done
done

wait

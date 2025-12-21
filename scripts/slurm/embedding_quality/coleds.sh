#!/bin/bash -l

#SBATCH --ntasks=8
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/coleds_embedding_quality.txt


source scripts/slurm/embedding_quality/common.sh

MAX_EPOCHS=50
ITERATIONS_PER_EPOCH=40
PATIENCE=6

COMMON_ARGS="
    partitioning.alpha=$DIRICHLET_ALPHA                   \
    wandb.log_to_wandb=true                               \
    experiment.name=$EXP_NAME                             \
    general.max_epochs=$MAX_EPOCHS                        \
    general.patience=$PATIENCE                            \
    train_config.num_iterations=$ITERATIONS_PER_EPOCH
"

###############################################################################
##### 1. Grid search for batch size / temperature for cifar10 && cinic10 ######
###############################################################################
#### 9 * 4 = 36 tests / dataset / seed ==>>>> 216 tests
for DATASET in cifar10 cinic10; do
for seed in "${SEEDS[@]}"; do
for bs in 1 2 4 8 16 32 48 64 96; do
set_partition_by "$DATASET"

all_temperatures="[0.05,0.2,0.5,1.0]"
all_fraction_fits="[0.5]"
all_num_client_updates="[4]"
model="set2set"

run_bg python scripts/py/train_coleds.py                      \
    $COMMON_ARGS                                              \
    model=$model                                              \
    train_config.fraction_fit=$all_fraction_fits              \
    train_config.num_client_updates=$all_num_client_updates   \
    train_config.temperature=$all_temperatures                \
    dataset=$DATASET                                          \
    general.seed=$seed                                        \
    train_config.batch_size=$bs                               \
    partitioning.partition_by=$partition_by

done
done
done


###########################################################################
##### 2. Grid search for fraction fit / model for cifar10 && cinic10 ######
###########################################################################
##### 3 * 4 = 12 tests / dataset / seed => 72 tests
for DATASET in cifar10 cinic10; do
for seed in "${SEEDS[@]}"; do
set_partition_by "$DATASET"

all_models=("set2set" "clmean" "gru")

for model in "${all_models[@]}"; do

all_fraction_fits="[0.05,0.25,0.5,1.0]"
all_batch_sizes="[16]"
all_temperatures="[0.2]"
all_num_client_updates="[4]"

ARGS="
    $COMMON_ARGS                                              \
    model=$model                                              \
    train_config.fraction_fit=$all_fraction_fits              \
    train_config.num_client_updates=$all_num_client_updates   \
    train_config.batch_size=$all_batch_sizes                  \
    train_config.temperature=$all_temperatures                 \
    dataset=$DATASET                                          \
    general.seed=$seed                                        \
    partitioning.partition_by=$partition_by
"

run_bg python scripts/py/train_coleds.py $ARGS

done
done
done


####################################################################################
##### 3. Grid search for gradient estimation steps / fraction fit for cifar10 ######
####################################################################################
#####  4 * 4 = 16 tests / seed / dataset ===>>>> 48 tests
for DATASET in cifar10; do
for seed in "${SEEDS[@]}"; do
set_partition_by "$DATASET"


all_num_client_updates="[1,2,4,8]"
model="set2set"
all_batch_sizes="[8,16]"
all_fraction_fits="[0.05,0.25,0.5,1.0]"
all_temperatures="[0.2]"

run_bg python scripts/py/train_coleds.py   \
    $COMMON_ARGS                                              \
    model=$model                                              \
    train_config.batch_size=$all_batch_sizes                  \
    train_config.fraction_fit=$all_fraction_fits              \
    train_config.num_client_updates=$all_num_client_updates   \
    train_config.temperature=$all_temperatures                \
    dataset=$DATASET                                          \
    general.seed=$seed                                        \
    partitioning.partition_by=$partition_by

done
done

##########################################################################################
##### 4. Grid search through reasonable values for mnist / fashion_mnist / cifar100 ######
##########################################################################################
#### 2 ** 5 = 32 tests / dataset / seed ===>>> 288 tests
for DATASET in mnist fashion_mnist cifar100; do
for seed in "${SEEDS[@]}"; do
set_partition_by "$DATASET"

# 2 * 2 * 2 * 2 * 2 = 32 tests / dataset / seed
# ===> 288 tests
all_models=("set2set" "clmean")
all_batch_sizes=(16 32)
all_fraction_fits=(0.25 0.5)
all_temperatures=(0.2 0.5)
all_num_client_updates=(4 8)

for model in "${all_models[@]}"; do
for batch_size in "${all_batch_sizes[@]}"; do
for fraction_fit in "${all_fraction_fits[@]}"; do
for temperature in "${all_temperatures[@]}"; do
for num_client_updates in "${all_num_client_updates[@]}"; do

# run individual tests to balance the load across the tasks
run_bg python scripts/py/train_coleds.py                 \
    $COMMON_ARGS                                         \
    model=$model                                         \
    train_config.batch_size=$batch_size                  \
    train_config.fraction_fit=$fraction_fit              \
    train_config.num_client_updates=$num_client_updates  \
    train_config.temperature=$temperature                \
    dataset=$DATASET                                     \
    general.seed=$seed                                   \
    partitioning.partition_by=$partition_by


done
done
done
done
done
done
done


#### TOTAL NUMBER OF TESTS = 216 + 72 + 48 + 288 = 624
#### However, there are also duplicate tests between different sections
#### Thus, the actual number of tests is 606

wait

#!/bin/bash -l

#SBATCH --ntasks=6
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/cl_embedding_quality.txt


source scripts/slurm/embedding_quality/common.sh

COMMON_ARGS="
    partitioning.alpha=$DIRICHLET_ALPHA                   \
    wandb.log_to_wandb=true                               \
    experiment.name=$EXP_NAME                             \
    general.max_epochs=$MAX_EPOCHS                        \
    train_config.num_iterations=$ITERATIONS_PER_EPOCH
"

##### 1. Grid search for batch size / temperature for cifar10 && cinic10

#### 2*2*4*8 = 128 tests / dataset / seed ==>>>> 768 tests
for DATASET in cifar10 cinic10; do
for seed in "${SEEDS[@]}"; do
set_partition_by "$DATASET"

all_batch_sizes1="[2,8,32,48]"
all_batch_sizes2="[1,4,16,64]"
all_temeratures="[0.05,0.2,0.5,1.0]"
all_fraction_fits="[0.25,0.5]"
all_num_client_updates="[2,4]"
model="set2set"

ARGS="
    $COMMON_ARGS                                              \
    model=$model                                              \
    train_config.fraction_fit=$all_fraction_fits              \
    train_config.num_client_updates=$all_num_client_updates   \
    train_config.temperature=$all_temeratures                 \
    dataset=$DATASET                                          \
    general.seed=$seed                                        \
    partitioning.partition_by=$partition_by
"

srun -Q -N1 --ntasks=1 python -u scripts/py/train_coleds.py train_config.batch_size=$all_batch_sizes1 $ARGS &
srun -Q -N1 --ntasks=1 python -u scripts/py/train_coleds.py train_config.batch_size=$all_batch_sizes2 $ARGS &

done
done


##### 2. Grid search for fraction fit / model for cifar10 && cinic10
##### 96 tests / seed / dataset ===>>>> 576 tests
for DATASET in cifar10 cinic10; do
for seed in "${SEEDS[@]}"; do
set_partition_by "$DATASET"
# 2 * 4 * 2 * 2 * 2 = 64 tests / dataset / seed => 384 tests

all_models=("set2set" "clmean" "gru")

for model in "${all_models[@]}"; do

all_fraction_fits="[0.05,0.25,0.5,1.0]"
all_batch_sizes="[8,16]"
all_temeratures="[0.2,0.5]"
all_num_client_updates="[2,4]"

ARGS="
    $COMMON_ARGS                                              \
    model=$model                                              \
    train_config.fraction_fit=$all_fraction_fits              \
    train_config.num_client_updates=$all_num_client_updates   \
    train_config.batch_size=$all_batch_sizes                  \
    train_config.temperature=$all_temeratures                 \
    dataset=$DATASET                                          \
    general.seed=$seed                                        \
    partitioning.partition_by=$partition_by
"

srun -Q -N1 --ntasks=1 python -u scripts/py/train_coleds.py $ARGS &

done
done
done


##### 3. Grid search for gradient estimation steps / fraction fit for cifar10
#####  64 tests / seed / dataset ===>>>> 384 tests

for DATASET in cifar10; do
for seed in "${SEEDS[@]}"; do
set_partition_by "$DATASET"


all_num_client_updates="[1,2,4,8]"
model="set2set"
all_batch_sizes="[8,16]"
all_fraction_fits="[0.05,0.25,0.5,1.0]"
all_temeratures="[0.2,0.5]"

srun -Q -N1 --ntasks=1 python -u scripts/py/train_coleds.py    \
    model=$model                                              \
    train_config.batch_size=$all_batch_sizes                  \
    train_config.fraction_fit=$all_fraction_fits              \
    train_config.num_client_updates=$all_num_client_updates   \
    train_config.temperature=$all_temeratures                 \
    dataset=$DATASET                                          \
    general.seed=$seed                                        \
    partitioning.partition_by=$partition_by                   &

done
done

##########################################################################################
##### 4. Grid search through reasonable values for mnist / fashion_mnist / cifar100 ######
##########################################################################################
for DATASET in mnist fashion_mnist cifar100; do
for seed in "${SEEDS[@]}"; do
set_partition_by "$DATASET"

# 2 * 2 * 2 * 2 * 2 = 32 tests / dataset / seed
# ===> 288 tests
all_models=(set2set "clmean")
all_batch_sizes=(16 32)
all_fraction_fits=(0.25 0.5)
all_temeratures=(0.2 0.5)
all_num_client_updates=(4 8)

for model in "${all_models[@]}"; do
for batch_size in "${all_batch_sizes[@]}"; do
for fraction_fit in "${all_fraction_fits[@]}"; do
for temperature in "${all_temeratures[@]}"; do
for num_client_updates in "${all_num_client_updates[@]}"; do

# run individual tests to balance the load across the tasks
srun -Q -N1 --ntasks=1 python -u scripts/py/train_coleds.py    \
    model=$model                                              \
    train_config.batch_size=$batch_size                       \
    train_config.fraction_fit=$fraction_fit                   \
    train_config.num_client_updates=$num_client_updates       \
    train_config.temperature=$temperature                     \
    dataset=$DATASET                                          \
    general.seed=$seed                                        \
    partitioning.partition_by=$partition_by                   &


done
done
done
done
done
done
done


#### TOTAL NUMBER OF TESTS = 1440 + 288
#### With 8 GPUs: 2016 / 8 = 252 tests per GPU

wait

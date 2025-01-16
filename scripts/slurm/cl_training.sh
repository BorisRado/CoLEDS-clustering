#!/bin/bash -l

#SBATCH --time=10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB
#SBATCH --partition=debug
#SBATCH --out=test.log

#SBATCH hetjob
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2GB


mamba activate test

first_group_hosts=$(srun --het-group=0 hostname)
echo "First group hosts: ---${first_group_hosts}---"


NUM_CLIENT_UPDATES=1
FRACTION_FIT=1.0
NUM_ITERATIONS=401
BATCH_SIZE=64
SEQ_PROC_MODEL=set2set
TEMPERATURE=0.5

NUM_CLIENTS=12

ARGS="train_config.num_client_updates=$NUM_CLIENT_UPDATES \
train_config.fraction_fit=$FRACTION_FIT \
+num_rounds=$NUM_ITERATIONS \
+num_clients=$NUM_CLIENTS \
train_config.batch_size=$BATCH_SIZE \
model=$SEQ_PROC_MODEL \
partitioning.num_partitions=$NUM_CLIENTS \
train_config.temperature=$TEMPERATURE"

srun -N1 --ntasks=1 --het-group=0 python scripts/py/run_server.py $ARGS &

sleep 5

for i in $(seq 0 $(($NUM_CLIENTS - 1))); do
    srun -N1 --ntasks=1 --het-group=1 --output=/dev/null --error=/dev/null python scripts/py/run_client.py \
        +server_ip=$first_group_hosts \
        +client_idx=$i \
        $ARGS &
done

wait

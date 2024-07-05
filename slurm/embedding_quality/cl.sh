#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/embedding_quality_cl.txt

source ~/miniforge3/etc/profile.d/conda.sh
conda activate flower

export PYTHONPATH=$PYTHONPATH:.
export HF_DATASETS_IN_MEMORY_MAX_SIZE=8589934592
export HF_DATASETS_CACHE=/home/radovib/.cache/huggingface/data/

model="set2set"
dirichlet_alpha=0.2
fraction_fit=0.5

if [ "$model" = "pointnet" ]; then
    MODEL_ARGS="model=pointnet statistics=mean head_sizes=\[32\]"
elif [ "$model" = "set2set" ]; then
    MODEL_ARGS="model=set2set"
else
    echo "error"
    exit 1
fi


for DATASET in mnist cifar10 cifar100; do

    if [ "$DATASET" = "cifar100" ]; then
        partition_by="fine_label"
    else
        partition_by="label"
    fi

    for batch_size in 32 64; do
        for temperature in 0.05 0.2 0.5 1.0; do
                echo "submitting..."
                srun --exclusive -N1 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=2G python -u scripts/train_cl_model.py \
                    $MODEL_ARGS                             \
                    train_config.batch_size=$batch_size     \
                    partitioning.alpha=$dirichlet_alpha     \
                    train_config.temperature=$temperature   \
                    train_config.fraction_fit=$fraction_fit \
                    dataset=$DATASET                        \
                    partitioning.partition_by=$partition_by \
                    +temp_run_id="cl_${model}_${DATASET}_bs${batch_size}_temp${temperature}" &
        done
    done
    wait
done

wait

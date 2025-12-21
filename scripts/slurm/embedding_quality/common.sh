#!/bin/bash -l

mamba activate slower

export PYTHONPATH=$PYTHONPATH:.
export PYTHONUNBUFFERED=1
export HF_DATASETS_IN_MEMORY_MAX_SIZE=8589934592
export HF_DATASETS_CACHE="/mnt/scratch/radovib"

DIRICHLET_ALPHA=0.2
EXP_NAME="embedding_quality"
SEEDS=(4 8 15 16 23) # 4 8 15 16 23 42

set_partition_by() {
    local dataset=$1
    if [ "$dataset" = "cifar100" ]; then
        partition_by="fine_label"
    else
        partition_by="label"
    fi
}

# Helper function to run commands asynchronously with srun
run_bg() {
    srun -Q --ntasks=1 "$@" &
}

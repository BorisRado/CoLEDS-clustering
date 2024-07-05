#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --out=logs/embedding_quality_wd.txt


source ~/miniforge3/etc/profile.d/conda.sh
conda activate flower
export PYTHONPATH=$PYTHONPATH:.
export HF_DATASETS_IN_MEMORY_MAX_SIZE=8589934592
export HF_DATASETS_CACHE=/home/radovib/.cache/huggingface/data/

for DATASET in mnist cifar10 cifar100; do
    if [ "$DATASET" = "cifar100" ]; then
        partition_by="fine_label"
    else
        partition_by="label"
    fi

  if [ "$DATASET" = "cifar100" ]; then
    partition_by="fine_label"
  else
    partition_by="label"
  fi

  echo "submitting..."
  srun python -u scripts/train_wd_model.py \
      partitioning.alpha=0.2                   \
      cem.ft_epochs=1                          \
      dataset=$DATASET                         \
      partitioning.partition_by=$partition_by  \
      +temp_run_id="wd_${DATASET}"
done

wait

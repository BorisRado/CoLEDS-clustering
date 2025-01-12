#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --out=logs/embedding_quality_wd.txt


mamba activate slower

export PYTHONPATH=$PYTHONPATH:.
export HF_DATASETS_IN_MEMORY_MAX_SIZE=8589934592
export HF_DATASETS_CACHE=/home/radovib/.cache/huggingface/data/

dirichlet_alpha=0.2

for DATASET in mnist cifar10 cifar100; do
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
      partitioning.partition_by=$partition_by
done

wait

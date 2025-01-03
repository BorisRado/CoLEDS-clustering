#!/bin/bash -l

#SBATCH --ntasks=10
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/embedding_quality_cl_cnt.txt
#SBATCH --constraint=gpu_p100|gpu_v100


mamba activate slower

export PYTHONPATH=$PYTHONPATH:.
export HF_DATASETS_CACHE=/home/radovib/.cache/huggingface/data/

dirichlet_alpha=0.2


for DATASET in mnist cifar10 cifar100; do

if [ "$DATASET" = "cifar100" ]; then
    partition_by="fine_label"
else
    partition_by="label"
fi

for seed in 7; do
for model in clmean set2set; do

for batch_size in 1 2 4 8 16 32; do
for fraction_fit in 0.05 0.25 0.5 1.0; do
for num_client_updates in 1 2 4; do

for temperature in 0.05 0.2 0.5 1.0; do
    srun -Q -N1 --ntasks=1 python -u scripts/train_cl_model.py \
        model=$model                            \
        train_config.batch_size=$batch_size     \
        train_config.fraction_fit=$fraction_fit \
        train_config.num_client_updates=$num_client_updates \
        partitioning.alpha=$dirichlet_alpha     \
        train_config.temperature=$temperature   \
        train_config.fraction_fit=$fraction_fit \
        dataset=$DATASET                        \
        wandb.log_to_wandb=true                 \
        general.seed=$seed                      \
        experiment_name="embedding_quality"     \
        partitioning.partition_by=$partition_by &


done
done
done
done
done
done

# wait # complete one dataset before moving to the next

done

wait

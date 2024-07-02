#!/bin/bash

#SBATCH --ntasks=6
#SBATCH --time=20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --out=logs/baseline.txt


source ../.venv/flower/bin/activate

# batch size: 16 32 64
# dirichlet alpha: 0.2 0.5
# statistics: mean, max, std
# temperature: 0.02 0.2
# head sizes [[], 32 64 128]
# fraction fit: [0.1 0.5 1.0]

# 3 * 2 * 3 * 2 * 4 * 3
# 6 * 6 * 12
# 432 / 16 = 27

for dataset in cifar10 cifar100 mnist; do

    if [ "$dataset" = "cifar100" ]; then
        partition_by="fine_label"
    else
        partition_by="label"
    fi

    for dirichlet_alpha in 0.2; do
        for cem in label random; do
            echo "submitting..."
            srun --ntasks=1 --cpus-per-task=8 --mem-per-cpu=1G python -u scripts/eval_trivial_cems.py \
                partitioning.alpha=$dirichlet_alpha     \
                dataset=$dataset                        \
                partitioning.partition_by=$partition_by \
                cem=$cem                                \
                wandb.loggin_keys=[partitioning.alpha] &
        done
    done
done
wait

#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --out=logs/concept_drift.txt

mamba activate slower

BASE_FOLDER="outputs/concept_drift"

rm -rf $BASE_FOLDER

for target_label in 1 7 3; do

    srun python -u scripts/py/femnist_feature_skew.py \
        +target_label=${target_label} \
        train_config.batch_size=4 \
        dataset=femnist \
        train_config.fraction_fit=0.25 \
        model=set2set \
        general.max_epochs=30 \
        hydra.run.dir=$BASE_FOLDER/${target_label}

    rm $BASE_FOLDER/${target_label}/model_weights.pth
done

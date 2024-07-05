#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=1G
#SBATCH --gpus=1
#SBATCH --out=logs/synthetic.txt


source ~/miniforge3/etc/profile.d/conda.sh
conda activate flower
export PYTHONPATH=$PYTHONPATH:.
export HF_DATASETS_IN_MEMORY_MAX_SIZE=8589934592
export HF_DATASETS_CACHE=/home/radovib/.cache/huggingface/data/

COMMONS="dataset=synthetic dataset.p=0.7"

srun python3 -u scripts/train_cl_model.py $COMMONS model=set2set +temp_run_id=synthetic_clc_set train_config.n_iterations=250
srun python3 -u scripts/train_cl_model.py $COMMONS model=pointnet +temp_run_id=synthetic_clc_pointnet train_config.n_iterations=250
srun python3 -u scripts/evaluate_logit_cem.py $COMMONS +temp_run_id=synthetic_logit
srun python3 -u scripts/train_wd_model.py $COMMONS +temp_run_id=synthetic_wdc
srun python3 -u scripts/tmp_train.py $COMMONS +temp_run_id=synthetic_esc_ae \
    dataset.n_datasets=1 dataset.dataset_size=10000 train_config.ae_weight=1.0
srun python3 -u scripts/tmp_train.py $COMMONS +temp_run_id=synthetic_esc_clf \
    dataset.n_datasets=1 dataset.dataset_size=10000 train_config.ae_weight=0.0

echo ""
echo ""
echo ""
echo ""
echo "STARTING VISUALIZATION"

srun python3 -u scripts/visualize_synthetic_cem.py folder=data/raw/synthetic_clc_set
srun python3 -u scripts/visualize_synthetic_cem.py folder=data/raw/synthetic_clc_pointnet
srun python3 -u scripts/visualize_synthetic_cem.py folder=data/raw/synthetic_esc_ae
srun python3 -u scripts/visualize_synthetic_cem.py folder=data/raw/synthetic_wdc
srun python3 -u scripts/visualize_synthetic_cem.py folder=data/raw/synthetic_esc_clf
srun python3 -u scripts/visualize_synthetic_cem.py folder=data/raw/synthetic_logit

wait

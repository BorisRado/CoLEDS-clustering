#!/bin/bash


source ~/miniforge3/etc/profile.d/conda.sh
conda activate flower
export PYTHONPATH=$PYTHONPATH:.

RUN_ID="tmp_times"

# training based CEMs
for dataset_size in 250 500 750 1000 1250 1500 1750 2000; do
    ARGS="+temp_run_id=${RUN_ID} dataset.dataset_size=${dataset_size}"
    echo $ARGS

    python scripts/measure_cem_time.py +cem=wd model=simple_net $ARGS
    python scripts/measure_cem_time.py +cem=logit model=simple_net $ARGS

    # label-free CEMs
    python scripts/measure_cem_time.py +cem=es model=simple_net $ARGS
    python scripts/measure_cem_time.py model=pointnet $ARGS
    python scripts/measure_cem_time.py model=set2set $ARGS
    python scripts/measure_cem_time.py model=te $ARGS

    # baseline CEMs
    python scripts/measure_cem_time.py +cem=random $ARGS
    python scripts/measure_cem_time.py +cem=label $ARGS
done

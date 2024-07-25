#!/bin/bash

export HYDRA_FULL_ERROR=1
RUN_ID="tmp_times"

python3 -m pip install torch-geometric

# training based CEMs
for dataset_size in 250 500 750 1000 1250 1500 1750 2000; do
    ARGS="+temp_run_id=${RUN_ID} dataset.dataset_size=${dataset_size}"
    echo $ARGS

    python3 scripts/measure_cem_time.py +cem=wd model=simple_net $ARGS
    python3 scripts/measure_cem_time.py +cem=logit model=simple_net $ARGS

    # label-free CEMs
    python3 scripts/measure_cem_time.py +cem=es model=sup_ae $ARGS
    python3 scripts/measure_cem_time.py model=pointnet $ARGS
    python3 scripts/measure_cem_time.py model=set2set $ARGS
    python3 scripts/measure_cem_time.py model=te $ARGS

    # baseline CEMs
    python3 scripts/measure_cem_time.py +cem=random $ARGS
    python3 scripts/measure_cem_time.py +cem=label $ARGS
done

echo "Evaluation has ended. Sleeping to give time to download the data"
echo "Sleeping for 600 seconds"
sleep 600  # this way there is time to download the data

echo "Shutting down"

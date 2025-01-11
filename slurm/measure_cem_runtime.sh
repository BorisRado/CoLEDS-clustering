#!/bin/bash

export HYDRA_FULL_ERROR=1
RUN_ID="tmp_times"

python3 -m pip install torch-geometric

FOLDER="results"

# training based CEMs
for dataset_size in 250 500 750 1000 1250 1500 1750 2000; do
    ARGS="dataset.dataset_size=${dataset_size} +folder=${FOLDER}"
    echo $ARGS

    python3 scripts/measure_cem_time.py +cem=wd model=simple_net $ARGS
    python3 scripts/measure_cem_time.py +cem=logit model=simple_net $ARGS

    # label-free CEMs
    python3 scripts/measure_cem_time.py model=clmean $ARGS
    python3 scripts/measure_cem_time.py model=set2set $ARGS
    python3 scripts/measure_cem_time.py model=te $ARGS

done

echo "Evaluation has ended. Sleeping to give time to download the data"
echo "Sleeping for 600 seconds"
sleep 600  # this way there is time to download the data

echo "Shutting down"

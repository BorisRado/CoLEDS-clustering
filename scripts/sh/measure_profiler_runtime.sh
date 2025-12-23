#!/bin/bash

export HYDRA_FULL_ERROR=1

FOLDER="results"

# training based CEMs
for dataset_size in 250 500 750 1000 1250 1500 1750 2000; do
    ARGS="dataset.dataset_size=${dataset_size} n_evaluations=16 +folder=${FOLDER}"
    echo $ARGS

    python3 scripts/py/measure_profiler_time.py profiler=wdp +cem=wdp model=simple_net $ARGS
    python3 scripts/py/measure_profiler_time.py profiler=lgp +cem=logit model=simple_net $ARGS
    python3 scripts/py/measure_profiler_time.py profiler=coleds model=clmean $ARGS
    python3 scripts/py/measure_profiler_time.py profiler=coleds model=set2set $ARGS
    python3 scripts/py/measure_profiler_time.py profiler=es model=simple_net $ARGS
    python3 scripts/py/measure_profiler_time.py profiler=es model=beta_vae $ARGS

done

echo "Evaluation has ended. Sleeping to give time to download the data"
echo "Sleeping for 600 seconds"
sleep 600  # this way there is time to download the data

echo "Shutting down"

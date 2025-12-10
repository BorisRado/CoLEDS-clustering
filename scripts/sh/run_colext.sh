#!/bin/bash

# execute some bash commands before starting the experiment
export PYTHONPATH=.

echo "Executing command: $@"
"$@"

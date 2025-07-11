#!/bin/sh
# Detect the server and set the PYTHONPATH accordingly

if [[ $(hostname) == "focs" ]]; then
    export MLSP_ROOT="/home/focs/Documents/Experiments/MLSP2025/"
    export MLSP_DATA_ROOT="/home/focs/Documents/Experiments/data/"
else
    export MLSP_ROOT="/dummy/MLSP2025/"
    export MLSP_DATA_ROOT="/dummy/"
fi

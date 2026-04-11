#!/bin/bash
# Wait for sweep jobs to finish on all workers, then pull the output .pt files back.
# Requires scripts/workers.txt; pulls output files matching the sweep config below.
set -e

LM_WEIGHTS=(0.1 0.3 1.0 3.0)
SHARPNESS=50.0
K=32
N=64

if [ ! -f scripts/workers.txt ]; then
    echo "scripts/workers.txt not found. Run scripts/discover_workers.sh first." >&2
    exit 1
fi
mapfile -t WORKER_IPS < <(awk '$2!="w-0"{print $1}' scripts/workers.txt)

mkdir -p data/sweep_collected

for i in "${!LM_WEIGHTS[@]}"; do
    ip=${WORKER_IPS[$i]}
    lm_w=${LM_WEIGHTS[$i]}
    out="data/sweep_lmw${lm_w}_shp${SHARPNESS}_K${K}_N${N}.pt"

    echo "waiting for worker $((i+1)): $out"
    while true; do
        status=$(ssh -o BatchMode=yes $ip \
            "pgrep -f 'python scripts/step2_lm_prior.py' > /dev/null && echo RUN || echo DONE" 2>&1)
        if [ "$status" = "DONE" ]; then
            break
        fi
        sleep 20
    done
    echo "  DONE; pulling $(basename $out)"
    scp -q -o BatchMode=yes $ip:coconut-cot-gen/$out data/sweep_collected/$(basename $out)
done

echo "all collected into data/sweep_collected/"
ls -la data/sweep_collected/

#!/bin/bash
# Wait for eval jobs and collect their log files.
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

mkdir -p outputs/sweep_evals

for attempt in $(seq 1 60); do
    all_done=1
    for i in "${!LM_WEIGHTS[@]}"; do
        ip=${WORKER_IPS[$i]}
        running=$(ssh -o BatchMode=yes $ip \
            "pgrep -f 'python scripts/evaluate_lmprior.py' > /dev/null && echo Y || echo N" 2>&1 || echo Y)
        if [ "$running" = "Y" ]; then
            all_done=0
            break
        fi
    done
    if [ $all_done -eq 1 ]; then
        break
    fi
    sleep 60
done

for i in "${!LM_WEIGHTS[@]}"; do
    ip=${WORKER_IPS[$i]}
    lm_w=${LM_WEIGHTS[$i]}
    log="/tmp/eval_sweep_lmw${lm_w}_shp${SHARPNESS}_K${K}_N${N}.log"
    dest="outputs/sweep_evals/$(basename $log)"
    scp -q $ip:$log $dest 2>&1 && echo "pulled $(basename $log)" \
        || echo "failed $log from worker $((i+1))"
done

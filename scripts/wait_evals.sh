#!/bin/bash
# Wait for eval jobs and collect their log files.
set -e

JOBS=(
    "10.164.0.48 /tmp/eval_sweep_lmw0.1_shp50.0_K32_N64.log"
    "10.164.0.41 /tmp/eval_sweep_lmw0.3_shp50.0_K32_N64.log"
    "10.164.0.38 /tmp/eval_sweep_lmw1.0_shp50.0_K32_N64.log"
    "10.164.0.40 /tmp/eval_sweep_lmw3.0_shp50.0_K32_N64.log"
)

mkdir -p outputs/sweep_evals

for attempt in $(seq 1 60); do
    all_done=1
    for job in "${JOBS[@]}"; do
        ip=$(echo $job | awk '{print $1}')
        running=$(ssh -o BatchMode=yes $ip "pgrep -xf 'python scripts/evaluate_lmprior.py --lmprior .*' > /dev/null && echo Y || echo N" 2>&1 || echo Y)
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

for job in "${JOBS[@]}"; do
    ip=$(echo $job | awk '{print $1}')
    log=$(echo $job | awk '{print $2}')
    dest="outputs/sweep_evals/$(basename $log)"
    scp -q $ip:$log $dest 2>&1 && echo "pulled $(basename $log)" || echo "failed $log from $ip"
done

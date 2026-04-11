#!/bin/bash
# Block until all sweep jobs finish, then pull outputs.
# NOTE: this version matches the *actual* python process, not the bash wrapper.
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

while true; do
    all_done=1
    for i in "${!LM_WEIGHTS[@]}"; do
        ip=${WORKER_IPS[$i]}
        running=$(ssh -o BatchMode=yes $ip \
            "pgrep -f 'python scripts/step2_lm_prior.py' > /dev/null && echo Y || echo N" 2>&1 || echo Y)
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

echo "all done, pulling outputs..."
for i in "${!LM_WEIGHTS[@]}"; do
    ip=${WORKER_IPS[$i]}
    lm_w=${LM_WEIGHTS[$i]}
    out="data/sweep_lmw${lm_w}_shp${SHARPNESS}_K${K}_N${N}.pt"
    scp -q $ip:coconut-cot-gen/$out data/sweep_collected/$(basename $out) \
        2>&1 || echo "  FAILED: $out from worker $((i+1))"
done

echo "collected:"
ls -la data/sweep_collected/

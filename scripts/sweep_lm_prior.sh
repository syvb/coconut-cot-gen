#!/bin/bash
# Sweep LM prior configs in parallel across workers (one lm_weight per worker).
# Assigns configs to worker-1 .. worker-N (skips worker-0, which is primary).
# Requires scripts/workers.txt populated; run scripts/discover_workers.sh first.
set -e

TARGETS="data/targets_500.pt"
N=64
STEPS=500
K=32
SHARPNESS=50.0
BATCH=8

# lm_weight values to sweep — one per worker (in worker-1..N order)
LM_WEIGHTS=(0.1 0.3 1.0 3.0)

if [ ! -f scripts/workers.txt ]; then
    echo "scripts/workers.txt not found. Run scripts/discover_workers.sh first." >&2
    exit 1
fi

# Get the IPs of worker-1, worker-2, ... (excluding worker-0)
mapfile -t WORKER_IPS < <(awk '$2!="w-0"{print $1}' scripts/workers.txt)

if [ ${#WORKER_IPS[@]} -lt ${#LM_WEIGHTS[@]} ]; then
    echo "need ${#LM_WEIGHTS[@]} workers but only ${#WORKER_IPS[@]} available" >&2
    exit 1
fi

for i in "${!LM_WEIGHTS[@]}"; do
    lm_w=${LM_WEIGHTS[$i]}
    ip=${WORKER_IPS[$i]}
    tag="lmw${lm_w}_shp${SHARPNESS}_K${K}_N${N}"
    out="data/sweep_${tag}.pt"
    log="/tmp/sweep_${tag}.log"

    cmd="python scripts/step2_lm_prior.py \
        --targets $TARGETS \
        --out $out \
        --K $K --steps $STEPS --batch $BATCH --n $N \
        --lr 1e-2 --init_std 0.02 \
        --lm_weight $lm_w --lm_sharpness $SHARPNESS \
        --lm_ref base --lm_exclude_specials \
        > $log 2>&1"
    echo "dispatching lm_weight=$lm_w to worker $((i+1))"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no $ip \
        "cd coconut-cot-gen && git pull --quiet 2>&1 && nohup bash -c '$cmd' >/dev/null 2>&1 &"
done
echo "all launched"

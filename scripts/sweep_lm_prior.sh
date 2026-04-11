#!/bin/bash
# Sweep LM prior configs in parallel across workers (4 configs → 4 workers).
# Each worker runs step2_lm_prior on the SAME 64 problems with a different lm_weight.
set -e

TARGETS="data/targets_500.pt"
N=64
STEPS=500
K=32
SHARPNESS=50.0
BATCH=8

# (lm_weight, worker_ip, worker_name)
declare -a CONFIGS=(
    "0.1 10.164.0.48 w-1"
    "0.3 10.164.0.41 w-2"
    "1.0 10.164.0.38 w-3"
    "3.0 10.164.0.40 w-4"
)

for cfg in "${CONFIGS[@]}"; do
    lm_w=$(echo $cfg | awk '{print $1}')
    ip=$(echo $cfg | awk '{print $2}')
    name=$(echo $cfg | awk '{print $3}')
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
    echo "dispatching lm_weight=$lm_w to $name ($ip)"
    ssh -f -o BatchMode=yes -o StrictHostKeyChecking=no $ip \
        "cd coconut-cot-gen && git pull --quiet 2>&1 && $cmd &" &
done
wait
echo "all launched"

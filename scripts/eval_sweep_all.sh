#!/bin/bash
# Run evaluate_lmprior.py on all sweep outputs in parallel across workers.
set -e

JOBS=(
    "10.164.0.48 data/sweep_lmw0.1_shp50.0_K32_N64.pt"
    "10.164.0.41 data/sweep_lmw0.3_shp50.0_K32_N64.pt"
    "10.164.0.38 data/sweep_lmw1.0_shp50.0_K32_N64.pt"
    "10.164.0.40 data/sweep_lmw3.0_shp50.0_K32_N64.pt"
)

for job in "${JOBS[@]}"; do
    ip=$(echo $job | awk '{print $1}')
    lmp=$(echo $job | awk '{print $2}')
    tag=$(basename $lmp .pt)
    logfile="/tmp/eval_${tag}.log"

    cmd="cd coconut-cot-gen && git pull --quiet && python scripts/evaluate_lmprior.py --lmprior $lmp > $logfile 2>&1"
    ssh -o BatchMode=yes $ip "nohup bash -c '$cmd' > /dev/null 2>&1 &" &
done
wait
echo "dispatched evals"

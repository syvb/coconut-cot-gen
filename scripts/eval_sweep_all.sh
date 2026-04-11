#!/bin/bash
# Run evaluate_lmprior.py on all sweep outputs in parallel across workers.
# Each output file is evaluated on the same worker that produced it (so the
# file is available locally — no pull required).
# Requires scripts/workers.txt populated; run scripts/discover_workers.sh first.
set -e

# Same config list as sweep_lm_prior.sh (kept in sync by convention)
LM_WEIGHTS=(0.1 0.3 1.0 3.0)
SHARPNESS=50.0
K=32
N=64

if [ ! -f scripts/workers.txt ]; then
    echo "scripts/workers.txt not found. Run scripts/discover_workers.sh first." >&2
    exit 1
fi
mapfile -t WORKER_IPS < <(awk '$2!="w-0"{print $1}' scripts/workers.txt)

for i in "${!LM_WEIGHTS[@]}"; do
    lm_w=${LM_WEIGHTS[$i]}
    ip=${WORKER_IPS[$i]}
    tag="sweep_lmw${lm_w}_shp${SHARPNESS}_K${K}_N${N}"
    lmp="data/${tag}.pt"
    logfile="/tmp/eval_${tag}.log"

    cmd="cd coconut-cot-gen && git pull --quiet && python scripts/evaluate_lmprior.py --lmprior $lmp > $logfile 2>&1"
    ssh -o BatchMode=yes $ip "nohup bash -c '$cmd' > /dev/null 2>&1 &"
done
echo "dispatched evals"

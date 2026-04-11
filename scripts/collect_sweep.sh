#!/bin/bash
# Wait for sweep jobs to finish on all workers, then pull the output .pt files back.
set -e

declare -a JOBS=(
    "10.164.0.48 data/sweep_lmw0.1_shp50.0_K32_N64.pt"
    "10.164.0.41 data/sweep_lmw0.3_shp50.0_K32_N64.pt"
    "10.164.0.38 data/sweep_lmw1.0_shp50.0_K32_N64.pt"
    "10.164.0.40 data/sweep_lmw3.0_shp50.0_K32_N64.pt"
)

mkdir -p data/sweep_collected

# Wait for each to finish (poll)
for job in "${JOBS[@]}"; do
    ip=$(echo $job | awk '{print $1}')
    out=$(echo $job | awk '{print $2}')
    echo "waiting for $ip: $out"
    while true; do
        status=$(ssh -o BatchMode=yes $ip "pgrep -f step2_lm_prior > /dev/null && echo RUNNING || echo DONE" 2>&1)
        if [ "$status" = "DONE" ]; then
            break
        fi
        sleep 20
    done
    echo "  DONE; pulling $out"
    scp -q -o BatchMode=yes $ip:coconut-cot-gen/$out data/sweep_collected/$(basename $out)
done

echo "all collected into data/sweep_collected/"
ls -la data/sweep_collected/

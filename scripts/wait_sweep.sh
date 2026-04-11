#!/bin/bash
# Block until all sweep jobs finish, then pull outputs. Minimal noise.
set -e

JOBS=(
    "10.164.0.48 data/sweep_lmw0.1_shp50.0_K32_N64.pt"
    "10.164.0.41 data/sweep_lmw0.3_shp50.0_K32_N64.pt"
    "10.164.0.38 data/sweep_lmw1.0_shp50.0_K32_N64.pt"
    "10.164.0.40 data/sweep_lmw3.0_shp50.0_K32_N64.pt"
)

mkdir -p data/sweep_collected

while true; do
    all_done=1
    for job in "${JOBS[@]}"; do
        ip=$(echo $job | awk '{print $1}')
        running=$(ssh -o BatchMode=yes $ip "pgrep -f 'python scripts/step2_lm_prior' > /dev/null && echo Y || echo N" 2>&1 || echo Y)
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
for job in "${JOBS[@]}"; do
    ip=$(echo $job | awk '{print $1}')
    out=$(echo $job | awk '{print $2}')
    scp -q $ip:coconut-cot-gen/$out data/sweep_collected/$(basename $out) 2>&1 || echo "  FAILED: $out from $ip"
done

echo "collected:"
ls -la data/sweep_collected/

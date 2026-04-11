#!/bin/bash
# Parallel setup on all workers (excluding worker-0).
set -e

WORKERS=$(awk '$2!="w-0"{print $1}' scripts/workers.txt)

# Push the setup script
for ip in $WORKERS; do
    scp -q -o StrictHostKeyChecking=no scripts/setup_worker.sh $ip:/tmp/setup_worker.sh
done

# Launch in parallel
pids=()
for ip in $WORKERS; do
    ssh -o StrictHostKeyChecking=no -o BatchMode=yes $ip "bash /tmp/setup_worker.sh" \
        > /tmp/worker_setup_${ip}.log 2>&1 &
    pids+=($!)
done

echo "Launched ${#pids[@]} setup jobs; PIDs: ${pids[@]}"
echo "Logs: /tmp/worker_setup_*.log"

# Wait for all
for pid in "${pids[@]}"; do
    wait $pid
done
echo "All worker setup jobs finished."

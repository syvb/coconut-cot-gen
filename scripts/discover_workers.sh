#!/bin/bash
# Populate workers.txt from the TPU VM metadata service. Run once on worker-0.
# workers.txt is gitignored — this is a local, per-deployment artifact.
set -e

out="scripts/workers.txt"
> $out

i=0
for ip in $(curl -sSf -H 'Metadata-Flavor: Google' \
        http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-network-endpoints \
        | tr ',' '\n' | awk -F: '{print $3}'); do
    echo "$ip w-$i" >> $out
    i=$((i + 1))
done

echo "wrote $out:"
cat $out

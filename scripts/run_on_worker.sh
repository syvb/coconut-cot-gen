#!/bin/bash
# Run a command on a specific worker. Usage: run_on_worker.sh <ip> <command>
set -e
ip=$1
shift
ssh -o BatchMode=yes -o StrictHostKeyChecking=no $ip "cd coconut-cot-gen && git pull --quiet && $*"

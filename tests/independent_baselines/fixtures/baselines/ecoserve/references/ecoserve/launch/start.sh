#!/bin/bash
#
# EcoServe launcher for the pdsh path. Edit the config block below, provide a
# `hostfile` (one instance IP per line), then run: bash ecoserve/launch/start.sh
#
# Roles started:
#   - ecoserve.launch.api_server_start     on the head node (client-facing API)
#   - ecoserve.launch.macro_instance_start on the head node (scheduler)
#   - ecoserve.launch.instance_start       on each host via pdsh (one engine each)

# Run everything from the repo root so `import ecoserve` resolves.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# ===== Config =====
HEAD_IP=127.0.0.1                       # node running the api server + macro
API_PORT=8000                           # client-facing API port
MACRO_PORT=12345                        # macro instance input port
STATE_PORT=12346                        # instance -> macro state port
INSTANCE_BASE_PORT=12347                # base port for per-instance sockets
OUTPUT_PORT=23456                       # instance -> api server output port
MODEL=facebook/opt-125m                 # model path or HuggingFace id
PREFILL_DATA=ecoserve/launch/prefill.csv  # prefill profile CSV (Length,Prefill Time)
HOSTFILE=ecoserve/launch/hostfile       # file with one instance IP per line
TP=1                                    # tensor parallel size per instance
INSTANCES_PER_NODE=1
MAX_NODES=4
TTFT=5000
TPOT=100
PYTHON=python                           # python on worker nodes (pdsh ssh does
                                        # not inherit the local conda env)
# ==================

# Start the API server (rank 0 role) on the head node.
$PYTHON -m ecoserve.launch.api_server_start \
  --head-ip "$HEAD_IP" \
  --port "$API_PORT" \
  --macro-port "$MACRO_PORT" \
  --output-port "$OUTPUT_PORT" &

# Read the host list and apply the node-count limit.
hosts=( $(cat "$HOSTFILE") )
total_nodes=${#hosts[@]}
if [ "$MAX_NODES" -gt 0 ] && [ "$MAX_NODES" -lt "$total_nodes" ]; then
  hosts=("${hosts[@]:0:$MAX_NODES}")
  total_nodes=$MAX_NODES
fi
total_instances=$(( total_nodes * INSTANCES_PER_NODE ))

# Start the macro instance (rank 1 role) on the head node.
$PYTHON -m ecoserve.launch.macro_instance_start \
  --nodes-num "$total_nodes" \
  --instances-per-node "$INSTANCES_PER_NODE" \
  --file "$HOSTFILE" \
  --head-ip "$HEAD_IP" \
  --macro-port "$MACRO_PORT" \
  --state-port "$STATE_PORT" \
  --instance-base-port "$INSTANCE_BASE_PORT" \
  --prefill-data-path "$PREFILL_DATA" \
  --TTFT "$TTFT" \
  --TPOT "$TPOT" &

echo "===== EcoServe distributed launch ====="
echo "Nodes: $total_nodes"
echo "Instances per node: $INSTANCES_PER_NODE"
echo "Total instances: $total_instances"
echo "======================================="

instance_counter=0

# Launch each instance on its host via pdsh.
for host in "${hosts[@]}"; do
  for (( node_instance=0; node_instance<INSTANCES_PER_NODE; node_instance++ )); do
    port=$(( INSTANCE_BASE_PORT + node_instance ))
    pdsh -R ssh -w "$host" "cd $REPO_ROOT; $PYTHON -m ecoserve.launch.instance_start \
      --instance-id $instance_counter \
      --port $port \
      -tp $TP \
      --model $MODEL \
      --instances-per-node $INSTANCES_PER_NODE \
      --head-ip $HEAD_IP \
      --api-port $OUTPUT_PORT \
      --state-port $STATE_PORT" &
    instance_counter=$(( instance_counter + 1 ))
  done
done

# Wait for all background jobs.
wait

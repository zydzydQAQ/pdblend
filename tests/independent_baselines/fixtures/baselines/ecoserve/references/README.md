# EcoServe

**Efficient LLM Serving on Commodity GPU Clusters with Data-Reduced Cross-Instance Orchestration** (OSDI '26)

EcoServe is an LLM serving system built for **commodity GPU clusters** — clusters
without high-performance interconnects such as NVLink or InfiniBand. It follows a
**partially disaggregated (PaDG)** strategy: within each instance, prefill and decode
are disaggregated along the *time* dimension to mitigate prefill–decode interference;
across instances, a cooperating group is cyclically ("rolling") activated — forming a
**macro instance** — so that prefills are always in flight and TTFT stays low. 
An adaptive scheduler routes requests within a macro instance while a mitosis scaling
mechanism adjusts capacity online, together improving goodput.

This repository provides **EcoServe's macro instance implementation** — the core
cross-instance orchestration layer (rolling activation, adaptive scheduling, and the
API-server / macro-instance / instance roles) that realizes the PaDG strategy on top
of vLLM.

> Built on [vLLM 0.7.3](https://github.com/vllm-project/vllm) — EcoServe code lives in
> [`ecoserve/`](ecoserve/), and most of the vLLM source is kept as-is apart from a few
> minor changes to the scheduler.
> See the [paper](https://www.usenix.org/conference/osdi26/presentation/du) for details.

## Architecture

EcoServe wires the **API server**, the **macro-instance**, and the **vLLM
instances** together over ZeroMQ — only requests, control signals, and generated tokens
cross the network, never KV cache (the "data-reduced" part of the design).

```
              request                route + control
  clients ──►  API server  ─────────►  Macro instance  ─────────►  Instances
          ◄──  (/generate)             (scheduler)     ◄─ state ──  (vLLM engines)
              tokens  ▲                                                  │
                      └──────────────  generated tokens  ────────────────┘
```

- **API server** (head node) — the client-facing `/generate` endpoint; forwards each
  request to the macro instance and streams back the tokens it receives from instances.
- **Macro instance** (head node) — the cross-instance scheduler. It tracks every
  instance's state and decides *when and where* requests run, driving rolling activation
  by sending control signals that toggle which instance currently takes prefills.
- **Instances** (one vLLM engine per host) — each does intra-instance scheduling
  (*when* to prefill vs. decode). vLLM's `step()` is split into `schedule()` +
  `execute()` so an instance reports its scheduler state to the macro instance before
  executing, then pushes generated tokens straight to the API server.

The group of cooperating instances driven by one scheduler is a **macro instance** —
EcoServe's basic serving unit.

## Build & Install

> **Recommended:** hand this README to a coding agent like **Claude Code** or **Codex** and let it drive the build and testing.

EcoServe is a source fork of vLLM 0.7.3 and is installed by building vLLM from source
(this compiles the CUDA kernels). It requires Linux, an NVIDIA GPU with CUDA, and
Python ≥ 3.9.

```bash
git clone https://github.com/MLSysU/EcoServe && cd EcoServe
conda create -n ecoserve python=3.11 -y && conda activate ecoserve

# Build vLLM 0.7.3 + the EcoServe layer (compiles CUDA kernels; takes a while).
pip install -e .

# The distributed launcher fans out over pdsh + ssh (pyzmq is already a dependency).
sudo apt-get install -y pdsh   # or build pdsh from source
```

## Launch

The scheduler requires a **prefill profiling config** that is generated on a separate
branch — see [Profiling config](#profiling-config) below. Once you have it:

1. Copy the profiling CSV to `ecoserve/launch/prefill.csv` (the path is configurable).
2. Create `ecoserve/launch/hostfile` with **one instance IP per line**
   (see [`hostfile.example`](ecoserve/launch/hostfile.example)).
3. Edit the config block at the top of
   [`ecoserve/launch/start.sh`](ecoserve/launch/start.sh) — `MODEL`, `TP`,
   `INSTANCES_PER_NODE`, `MAX_NODES`, the SLO targets `TTFT`/`TPOT`, the ports, etc.
4. Launch the whole cluster from the repo root:

```bash
bash ecoserve/launch/start.sh
```

This launches three roles: the client-facing **API server** and the **macro instance**
(scheduler) on the head node, plus one **engine instance** per host listed in `hostfile`
(started via pdsh). The HTTP `/generate` endpoint is served at
`http://<HEAD_IP>:<API_PORT>/generate`.

### Profiling config

The macro-instance scheduler consumes a per-model prefill profile (batched prefill
length → prefill time). Generate it on the **`llm_profiler`** branch:

```bash
git checkout llm_profiler
python llm_profiler.py           # writes a prefill profiling CSV
```

Then switch back to `ecoserve` and place the CSV at `ecoserve/launch/prefill.csv`
(or point `PREFILL_DATA` in `start.sh` to it).

## Benchmark

With the cluster running, drive it with EcoServe's serving benchmark:

```bash
python -m ecoserve.benchmarks.benchmark_serving_ecoserve \
  --backend vllm \
  --host <HEAD_IP> --port 8000 --endpoint /generate \
  --model <model-path-or-hf-id> \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000 \
  --request-rate 4 \
  --save-result
```

`--request-rate` sets the Poisson arrival rate in QPS (`inf` sends all prompts at
once). Results — throughput, TTFT/TPOT percentiles, and SLO achievement rate — are
printed to the console and, with `--save-result`, written to a JSON file.

See [`ecoserve/benchmarks/README.md`](ecoserve/benchmarks/README.md) for for more details.

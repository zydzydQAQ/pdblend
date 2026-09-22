#!/usr/bin/env python3
"""Run one reproducible experiment after verifying exclusive access to all eight GPUs.

Host uses only stdlib + docker. Source/profile/corpus hashes and the pinned image
are recorded; the source snapshot is mounted read-only. Existing results are
never overwritten. Does not stop, restart, or preempt any other GPU job.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("profile", "bench"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    p.add_argument("--profile", type=Path, default=Path("results/v2/profile-7b/profile.json"))
    p.add_argument("--policy", default="manual")
    p.add_argument("--layout", default="")
    p.add_argument("--clocks", default="P=2520,D=2520,M=2100")
    p.add_argument("--seed", type=int, default=701)
    p.add_argument("--split", default="evaluation")
    p.add_argument("--rate", type=float, default=8.109)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.out, a.source, a.profile = a.out.resolve(), a.source.resolve(), a.profile.resolve()
    corpus = root / "datasets/prepared/2026-09-21-7b-v2-half"
    with open("/tmp/pdblend4-gpu-experiment.lock", "a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if apps.strip():
            raise RuntimeError("GPU processes still active; no experiment started: " + apps)
        containers = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True).splitlines()
        if "pdb2-matrix-v2" in containers:
            raise RuntimeError("Matrix still owns GPUs, even if between points")
        if a.out.exists():
            raise FileExistsError(f"Use a new output directory: {a.out}")
        a.out.mkdir(parents=True)
        snapshot = a.out / "source"
        shutil.copytree(a.source, snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        image = subprocess.check_output(["docker", "image", "inspect", "pdblend:l20-cu128-vllm-v1", "--format", "{{.Id}}"], text=True).strip()
        files = [a.profile, corpus / "sharegpt.json", corpus / "manifest.json"]
        manifest = dict(image=image, started_s=time.time(), kind=a.kind,
                        inputs={str(f): sha(f) for f in files},
                        source={str(f.relative_to(snapshot)): sha(f) for f in snapshot.rglob("*.py")},
                        hardware=subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv"], text=True))
        if a.kind == "profile":
            code = ("from pathlib import Path; from pdblend.profile.profiler import Profiler; "
                    f"Profiler('Qwen2.5-7B-Instruct',[0,1],freqs=(2100,2520),mixed_freqs=(2100,2520),out_dir=Path({str(a.out / 'measurement')!r})).run()")
            action = ["python", "-B", "-c", code]
            manifest["energy_comparable"] = False
        else:
            action = ["python", "-B", "-m", "pdblend.cli", "bench", "--policy", a.policy,
                      "--profile", str(a.profile), "--corpus", str(corpus), "--dataset", "sharegpt",
                      "--split", a.split, "--rate", str(a.rate), "--duration", "300", "--seed", str(a.seed),
                      "--scale", "0.5", "--layout", a.layout, "--clocks", a.clocks, "--out", str(a.out / "measurement")]
            manifest["energy_comparable"] = True
        command = ["docker", "run", "--rm", "--name", "pdb4-opt-" + a.out.name,
                   "--ulimit", "nofile=65536:65536", "--gpus", "all", "--cap-add", "SYS_ADMIN",
                   "--ipc=host", "--shm-size=16g", "--network", "host",
                   "-v", f"{root}:{root}:ro", "-v", f"{a.out}:{a.out}",
                   "-v", f"{snapshot}:/opt/pdblend-src:ro", "-v", "/home/models:/models:ro",
                   "-e", "PYTHONPATH=/opt/pdblend-src", "-e", "PDBLEND_MODELS_DIR=/models",
                   "-w", str(root), image, *action]
        manifest["command"] = command
        (a.out / "execution.json").write_text(json.dumps(manifest, indent=2))
        with (a.out / "run.log").open("w") as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        manifest.update(finished_s=time.time(), returncode=proc.returncode,
                        inputs_unchanged=all(sha(Path(f)) == digest for f, digest in manifest["inputs"].items()))
        (a.out / "execution.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps({k: manifest[k] for k in ("kind", "returncode", "inputs_unchanged")}), flush=True)
        if proc.returncode or not manifest["inputs_unchanged"]:
            sys.exit(proc.returncode or 1)


if __name__ == "__main__":
    main()

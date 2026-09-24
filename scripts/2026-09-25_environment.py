#!/usr/bin/env python3
"""Capture, transport and check the v3 engine using only host stdlib and Docker.

CPU verification never requests a GPU. --hardware is an explicit target-host
8 x L20 CUDA smoke test; it must be run when the GPUs are free.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "requirements/pdblend4-v3-image.json"
DEFAULT_IMAGE = "pdblend:l20-cu128-vllm-v1"
PATCH = ROOT / "engine_patches/vllm-0.10.1.1/manifest.json"
PROBE = r'''
import hashlib, importlib.metadata as m, json, platform, sys
import torch
paths = json.loads(sys.argv[1])
site = m.distribution("vllm").locate_file("")
print(json.dumps({
    "python": platform.python_version(),
    "machine": platform.machine(),
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "packages": {d.metadata["Name"].lower().replace("_", "-"): d.version
                 for d in m.distributions() if d.metadata["Name"].lower() != "pdblend"},
    "patches": {p: hashlib.sha256((site / p).read_bytes()).hexdigest() for p in paths},
}, sort_keys=True))
'''
GPU_PROBE = r'''
import json, torch
assert torch.cuda.device_count() == 8, "requires exactly 8 visible GPUs"
results = []
for i in range(8):
    name = torch.cuda.get_device_name(i)
    assert name == "NVIDIA L20", (i, name)
    with torch.cuda.device(i):
        a = torch.ones((32, 32), device=f"cuda:{i}", dtype=torch.bfloat16)
        assert (a @ a).float().mean().item() == 32.0
        torch.cuda.synchronize()
    results.append({"index": i, "name": name, "bf16_matmul": "passed"})
print(json.dumps(results))
'''


def run(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def inspect(image: str) -> dict:
    value = json.loads(run(["docker", "image", "inspect", image]))[0]
    return {"id": value["Id"], "repo_tags": value.get("RepoTags", []),
            "repo_digests": value.get("RepoDigests", []),
            "architecture": value["Architecture"], "os": value["Os"],
            "created": value["Created"], "rootfs": value["RootFS"],
            "config": value["Config"]}


def probe(image: str) -> dict:
    paths = [f["path"] for f in json.loads(PATCH.read_text())["files"]]
    return json.loads(run(["docker", "run", "--rm", "--network", "none",
                           "--entrypoint", "/opt/venv/bin/python", image,
                           "-c", PROBE, json.dumps(paths)]))


def capture(image: str, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to replace captured identity: {output}; use --identity with a new path")
    observed = inspect(image)
    runtime = probe(image)
    value = {"schema": "pdblend4-v3-engine-identity/v1",
             "captured_at_utc": datetime.now(timezone.utc).isoformat(),
             "image": observed, "runtime": runtime,
             "patch_manifest_sha256": sha(PATCH),
             "container_lock_sha256": sha(ROOT / "requirements/pdblend4-v3-container.lock"),
             "host_lock_sha256": sha(ROOT / "requirements/pdblend4-v3-host.lock"),
             "note": "CUDA base 12.8.1; actual engine Torch 2.7.1+cu126 uses CUDA 12.6. Source is mounted separately by experiment launchers."}
    write(output, value)
    print(json.dumps({"identity": str(output), "image_id": observed["id"]}))


def verify(image: str, identity: Path, rebuilt: bool = False, hardware: bool = False) -> dict:
    expected = json.loads(identity.read_text())
    observed = inspect(image)
    for key in ("architecture", "os"):
        if observed[key] != expected["image"][key]:
            raise ValueError(f"image {key} mismatch")
    # Docker's classic and containerd stores can report different IDs for the
    # same OCI image; uncompressed layer digests plus runtime config are stable.
    if not rebuilt:
        for key in ("rootfs", "config"):
            if observed[key] != expected["image"][key]:
                raise ValueError(f"image {key} mismatch; this is not the captured image")
    for path, field in ((PATCH, "patch_manifest_sha256"),
                        (ROOT / "requirements/pdblend4-v3-container.lock", "container_lock_sha256"),
                        (ROOT / "requirements/pdblend4-v3-host.lock", "host_lock_sha256")):
        if sha(path) != expected[field]:
            raise ValueError(f"tracked environment input changed: {path}")
    runtime = probe(image)
    for key in ("machine", "torch_version", "torch_cuda", "packages", "patches"):
        if runtime[key] != expected["runtime"][key]:
            raise ValueError(f"runtime {key} mismatch")
    if runtime["python"].split(".")[:2] != ["3", "10"]:
        raise ValueError("engine requires Python 3.10")
    check = run(["docker", "run", "--rm", "--network", "none", "--entrypoint",
                 "/opt/venv/bin/python", image, "-m", "pip", "check"])
    report = {"schema": "pdblend4-v3-environment-check/v1", "image_id": observed["id"],
              "captured_image_id": expected["image"]["id"],
              "exact_image_layers_and_config": not rebuilt,
              "runtime_versions_and_patches": "passed", "pip_check": check,
              "hardware": "not_requested"}
    if hardware:
        report["hardware"] = json.loads(run([
            "docker", "run", "--rm", "--network", "none", "--gpus", "all",
            "--entrypoint", "/opt/venv/bin/python", image, "-c", GPU_PROBE]))
    return report


def export_image(image: str, archive: Path, identity: Path) -> None:
    verify(image, identity)
    receipt = Path(str(archive) + ".json")
    if archive.exists() or receipt.exists():
        raise FileExistsError(f"refusing to overwrite archive or receipt: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(archive) + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    with partial.open("xb") as output:
        save = subprocess.Popen(["docker", "image", "save", image], stdout=subprocess.PIPE)
        try:
            compress = subprocess.run(["zstd", "-T2", "-3", "-c"], stdin=save.stdout, stdout=output)
            save.stdout.close()
            status = save.wait()
            if compress.returncode or status:
                raise RuntimeError(f"image export failed: docker={status} zstd={compress.returncode}")
        except BaseException:
            save.terminate()
            save.wait()
            raise
    partial.rename(archive)
    write(receipt, {"schema": "pdblend4-v3-image-archive/v1", "archive": archive.name,
                    "sha256": sha(archive), "bytes": archive.stat().st_size,
                    "identity_sha256": sha(identity), "image": inspect(image)})
    print(json.dumps({"archive": str(archive), "receipt": str(receipt)}))


def import_image(archive: Path, identity: Path, receipt: Path | None) -> None:
    receipt = receipt or Path(str(archive) + ".json")
    meta = json.loads(receipt.read_text())
    if meta["identity_sha256"] != sha(identity):
        raise ValueError("archive belongs to a different captured image identity")
    if archive.stat().st_size != meta["bytes"] or sha(archive) != meta["sha256"]:
        raise ValueError("image archive checksum mismatch")
    for tag in meta["image"]["repo_tags"]:
        found = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, text=True)
        if found.returncode == 0:
            existing = json.loads(found.stdout)[0]
            if existing["RootFS"] != meta["image"]["rootfs"] or existing["Config"] != meta["image"]["config"]:
                raise ValueError(f"existing image tag would be replaced: {tag}; preserve it under another tag first")
    unpack = subprocess.Popen(["zstd", "-d", "-c", str(archive)], stdout=subprocess.PIPE)
    try:
        loaded = subprocess.run(["docker", "image", "load"], stdin=unpack.stdout)
        unpack.stdout.close()
        status = unpack.wait()
        if loaded.returncode or status:
            raise RuntimeError(f"image import failed: docker={loaded.returncode} zstd={status}")
    except BaseException:
        unpack.terminate()
        unpack.wait()
        raise
    print(json.dumps(verify(meta["image"]["repo_tags"][0], identity), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "verify", "export", "import"):
        sub = commands.add_parser(name)
        sub.add_argument("--identity", type=Path, default=IDENTITY)
        if name != "import":
            sub.add_argument("--image", default=DEFAULT_IMAGE)
        if name in ("export", "import"):
            sub.add_argument("--archive", type=Path, required=True)
        if name == "import":
            sub.add_argument("--receipt", type=Path)
        if name == "verify":
            sub.add_argument("--allow-rebuilt", action="store_true",
                             help="check versions and patches; allow different image layers")
            sub.add_argument("--hardware", action="store_true", help="run BF16 CUDA smoke on 8 idle L20s")
            sub.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.image, args.identity)
    elif args.command == "verify":
        report = verify(args.image, args.identity, args.allow_rebuilt, args.hardware)
        if args.output:
            write(args.output, report)
        print(json.dumps(report, indent=2))
    elif args.command == "export":
        export_image(args.image, args.archive, args.identity)
    else:
        import_image(args.archive, args.identity, args.receipt)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))

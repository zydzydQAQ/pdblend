"""Pinned read-only legacy baseline dependencies; no hardware at import."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parents[1]
CAMPAIGNS = NEW.parent
PRIOR = CAMPAIGNS / "parallel-rate-20260908-v1"
sys.path.insert(0, str(NEW))
import slo_support as p

PRODUCER = PRIOR / "B/uniform-rate-20260909-v2/migration-14b-sharegpt/baseline_producer_v3.py"
VERIFIER = PRIOR / "B/uniform-rate-20260909-v2/migration-14b-sharegpt/baseline_verify.py"
PREPARATION = PRIOR / "A/uniform-rate-20260909-v2/baseline-preparation"
QUALIFIER = PREPARATION / "qualification-v3/qualify.py"
POLICY_ADAPTER = PREPARATION / "policy_adapter.py"
PROFILE = PRIOR / "B/distributed-14b-v1/frequency2100-registered-001/profiles.development.json"
EXECUTOR = PRIOR / "common/execution-until-complete-v1/run.py"
HOSTS = {"A": "iZwz9274emxme9019d2sjgZ", "C": "iZwz9gfq11hx1sbob59yrgZ"}
NATIVE_NODES = {"A": "Anew20260909", "C": "C"}


def files(directory):
    return {str(path): p.sha(path) for path in Path(directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def build_dependencies():
    """CPU stage list: existing different hashes must never be overwritten."""
    destination = HERE / "dependencies.json"
    if destination.exists():
        return p.read(destination)
    producer = p.load(PRODUCER, "slo14_original_baseline_producer")
    template_path = HERE / "original-B-template.json"
    if not template_path.exists():
        producer.template("8tp1", template_path)
    template = p.read(template_path)
    frozen = dict(template["files"])
    directories = [PREPARATION,
        CAMPAIGNS / "AC-baseline-deployment-v1", CAMPAIGNS / "AC-baseline-binding-v2",
        CAMPAIGNS / "AC-baseline-100s-preparation-v1", CAMPAIGNS / "AC-legacy-resident-correctness-v1",
        CAMPAIGNS / "A14B-legacy-heterogeneous-correctness-v1",
        PRIOR / "A/eco-drain31-v1/code", PRIOR / "A/uniform-rate-20260909-v1/meter-runtime",
        PRIOR / "A/uniform-rate-20260909-v1/isolated-power",
        PRIOR / "common/execution-until-complete-v1", PRIOR / "common/token-evidence-v2",
        CAMPAIGNS.parent / "releases/five-system100-A14B-baseline-v1-runtime",
        CAMPAIGNS.parent / "releases/five-system100-A14B-baseline-eco-drain-v1-runtime"]
    for directory in directories:
        frozen.update(files(directory))
    for path in [PRODUCER, VERIFIER, PROFILE, POLICY_ADAPTER, QUALIFIER,
        PRIOR / "A/uniform-rate-20260909-v1/stream.py",
        PRIOR / "common/uniform-rate-20260909-v2/support.py",
        PRIOR / "B/baseline-return-after-external-source-v1/execution.py"]:
        frozen[str(path)] = p.sha(path)
    # Historical mechanical policies read these cost files while keeping their
    # values unchanged. Retained weights themselves are freshly produced later.
    policy = p.load(CAMPAIGNS / "AC-baseline-100s-preparation-v1/bind.py", "slo14_original_policy_sources")
    def external(value, key=""):
        if isinstance(value, dict):
            for k, item in value.items():
                external(item, k)
        elif isinstance(value, list):
            for item in value:
                external(item, key)
        elif isinstance(value, str) and value.startswith("/root/workspace"):
            path = Path(value)
            if path.is_file() and key not in ("journal",):
                frozen[value] = p.sha(path)
    for system in ("mixed", "distserve", "dynamollm", "ecoserve"):
        config, _ = policy.configuration("14b", "sharegpt", system)
        external(config)
    for filename, digest in list(frozen.items()):
        p.need(p.sha(filename) == digest, "original frozen dependency differs: " + filename)
    result = dict(schema="slo14-baseline-frozen-dependency-list-v1", files=frozen,
        bytes=sum(Path(path).stat().st_size for path in frozen),
        original_sources_preserved=True, new_experiment_outputs_root=str(NEW),
        staging_policy="add missing original-path read-only dependencies; reuse same hash; refuse different hash",
        producer=p.ref(PRODUCER), qualifier=p.ref(QUALIFIER), verifier=p.ref(VERIFIER),
        policy_adapter=p.ref(POLICY_ADAPTER), profile=p.ref(PROFILE),
        original_template=p.ref(template_path), gpu_work_performed=False)
    p.save(destination, result)
    return result


def checked_dependencies():
    result = p.read(HERE / "dependencies.json")
    for path, digest in result["files"].items():
        p.need(p.sha(path) == digest, "frozen baseline dependency changed: " + path)
    p.need(result["new_experiment_outputs_root"] == str(NEW), "foreign experiment root")
    return result


if __name__ == "__main__":
    result = build_dependencies()
    print(json.dumps(dict(manifest=p.ref(HERE / "dependencies.json"), files=len(result["files"]), bytes=result["bytes"])))

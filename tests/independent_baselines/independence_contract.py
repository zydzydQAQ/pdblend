"""Audit policy isolation while permitting reviewed measurement infrastructure.

Permissions bind both the importing file and the imported symbol. A baseline
policy/profile model cannot inherit a collector's permission, and wildcard or
module imports cannot expose other PDBlend implementations through an alias.
"""
import ast


MEASUREMENT_IMPORTS = {
    "resident_campaign.py": {
        "pdblend.engine.launcher": {"Fleet"},
        "pdblend.bench.metering": {"Gpus"},
    },
    "ecoserve/auto_macro.py": {
        "pdblend.engine.launcher": {"Fleet"},
        "pdblend.bench.metering": {"Gpus"},
    },
    "distserve/gpu_probe.py": {
        "pdblend.bench.gates": {"random_prompt"},
        "pdblend.engine.launcher": {"Fleet"},
        "pdblend.bench.metering": {"Gpus"},
    },
    "distserve/stage_collect.py": {
        "pdblend.engine.launcher": {"Fleet"},
        "pdblend.bench.metering": {"Gpus"},
        "pdblend.measure.power": {"trapezoid_mean_power"},
        "pdblend.profile.wave": {"ProfileWave", "atomic_json"},
    },
    "dynamollm/profile_epochs.py": {
        "pdblend.profile.sampling_epochs": {"SamplingEpochs"},
    },
    "dynamollm/asset_preflight.py": {
        "pdblend.profile.sampling_epochs": {"SamplingEpochs"},
    },
}


def assert_independent_imports(source, relative_path):
    allowed = MEASUREMENT_IMPORTS.get(relative_path, {})
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            assert not any(n.name == "pdblend" or n.name.startswith("pdblend.")
                           for n in node.names), (relative_path, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "pdblend" or module.startswith("pdblend."):
                names = {n.name for n in node.names}
                assert module in allowed and names <= allowed[module], (
                    relative_path, node.lineno, module, names)

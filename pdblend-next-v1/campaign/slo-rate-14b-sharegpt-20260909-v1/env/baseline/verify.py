"""Independently replay the original baseline proof on the declared A/C host.

The sole extension to the frozen raw verifier is the explicit C/14B host name.
Native, numerical, frequency, source-equivalence and policy checks stay intact.
"""
from pathlib import Path
import sources as s
p = s.p


def verify(reference):
    dependencies = s.checked_dependencies()
    binding = p.checked(reference)
    matches = [node for node, host in s.HOSTS.items() if host == binding.get("hostname")]
    p.need(len(matches) == 1 and binding.get("model") == "14b"
        and set(binding.get("configs", {})) == {"sharegpt"}, "wrong baseline actual host/model/dataset")
    node = matches[0]
    evidence = binding.get("fresh_legacy_qualification", {})
    p.need(evidence.get("node") == s.NATIVE_NODES[node]
        and evidence.get("old_node_qualification_inherited") is False,
        "fresh baseline physical qualification required")
    original = p.load(dependencies["verifier"], "slo14_frozen_independent_baseline_verifier")
    original.HOSTNAMES = {s.NATIVE_NODES[n]: host for n, host in s.HOSTS.items()}
    result = original.verify(reference)
    p.need(result.get("passed") and result.get("independently_recomputed"), "raw qualification did not pass")
    result["files"].update({str(Path(__file__).resolve()): p.sha(__file__),
                            str(s.HERE / "sources.py"): p.sha(s.HERE / "sources.py"),
                            str(s.HERE / "dependencies.json"): p.sha(s.HERE / "dependencies.json")})
    result["slo_rate_host_binding"] = dict(node=node, hostname=s.HOSTS[node],
        original_verifier=dependencies["verifier"], original_native_frequency_policy_checks_unchanged=True,
        extension="explicit Anew/C 14B hostname registration only")
    return result

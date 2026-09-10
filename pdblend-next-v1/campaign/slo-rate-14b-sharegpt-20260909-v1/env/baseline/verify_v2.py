"""Original full raw baseline verifier plus exact B control registration proof."""
import prepare_v2 as producer
import verify as original
p = producer.p


def verify(reference):
    binding = p.checked(reference)
    evidence = binding["fresh_legacy_qualification"]
    declaration = producer.verify_control_bootstrap(evidence["deployment_bootstrap"])
    registration, _ = producer.registered()
    p.need(evidence["profile"] == registration["profile"], "qualification used another baseline profile")
    result = original.verify(reference)
    for path in (__file__, producer.__file__, producer.REGISTRATION):
        result["files"][str(path)] = p.sha(path)
    for ref in (declaration["registration"], declaration["source_bootstrap"], declaration["repair"]):
        result["files"][ref["path"]] = ref["sha256"]
    result["baseline_control_registration"] = declaration
    return result

"""Frozen qualification cache; live executor identity checks remain mandatory."""
import hashlib, types
CACHE = {'path': '/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/A/uniform-rate-20260909-v2/qualification-cache-parent-001/cache.json', 'sha256': 'a86d38dc975945e78c37e4cdd5b243a89b031c9669ca661223c64bc382b67f69'}
HELPER = {'path': '/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/common/uniform-rate-20260909-v2/qualification-cache-v1/cache.py', 'sha256': 'a22fc13569b52a20a255317eeb85be18f3aec7ebcacdc14f20cd8c67beac85b9'}
def verify(reference):
    with open(HELPER["path"], "rb") as stream:
        source = stream.read()
    if hashlib.sha256(source).hexdigest() != HELPER["sha256"]: raise ValueError("cache verifier changed")
    module = types.ModuleType("immutable_qualification_cache"); module.__file__ = HELPER["path"]
    exec(compile(source, HELPER["path"], "exec"), module.__dict__)
    return module.verify_cached(reference, CACHE)

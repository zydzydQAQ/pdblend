"""List pinned missing core dependencies after C checkpoint recovery; no network."""
import json
from pathlib import Path
import preflight as p


def collect():
    refs = {}
    def add(path, digest, kind):
        path = Path(path)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError('missing explicit dependency digest: ' + str(path))
        if path.is_file():
            if p.sha(path) != digest:
                raise ValueError('existing dependency hash mismatch: ' + str(path))
            return
        key = str(path)
        if key in refs and refs[key]['sha256'] != digest:
            raise ValueError('conflicting source digests: ' + key)
        refs[key] = dict(path=key, sha256=digest, kind=kind)
    d = p.read(p.ROOT / 'common/ascending-rate-execution-v2/release-001/declaration.json')
    for obs in d['reused_observations'] + d['stopped_P12_history']:
        if obs['measurement_host'] != 'C':
            continue
        cp = p.read(obs['checkpoint']['path'])
        for name in ('binding', 'receipt'):
            ref = cp.get(name)
            if isinstance(ref, str):
                add(ref, cp[name + '_sha256'], 'checkpoint_dependency')
            elif isinstance(ref, dict):
                add(ref['path'], ref['sha256'], 'checkpoint_dependency')
        # Core small request metrics are useful for independent recomputation;
        # larger power/event/log artifacts remain on their original machine.
        for path, digest in cp.get('artifacts', {}).items():
            if Path(path).name in ('bench.csv', 'summary.json', 'receipt.json', 'config.json'):
                add(path, digest, 'checkpoint_dependency')
    for name in ('hosts/7b-capacity-p12/manifest.json',):
        path = p.ROOT / name
        manifest = p.read(path)
        for rel, digest in manifest['files'].items():
            add(path.parent / rel, digest, 'runtime_source')
    for name in ('C/p4-completion-release/release.json', 'C/fixed-release-p4-strict/release.json'):
        release = p.read(p.ROOT / name)
        for field in ('binding', 'declaration', 'qualification'):
            ref = release[field]
            add(ref['path'], ref['sha256'], 'execution_dependency')
    return sorted(refs.values(), key=lambda x:x['path'])


if __name__ == '__main__':
    out = p.HERE / 'C-dependency-refs-001.json'
    refs = collect()
    with out.open('x') as stream:
        json.dump(refs, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(path=str(out), count=len(refs))))

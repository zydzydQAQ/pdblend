#!/usr/bin/env python3
"""Install immutable predictor assets with hardlink compatibility paths.

No byte is moved or deleted. Existing container mounts and checksum manifests
remain valid; this operation intentionally claims zero reclaimed disk bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results'))
    parser.add_argument('--assets',type=Path,default=Path('artifacts/predictors'))
    args=parser.parse_args(argv);entries=[]
    for weight in sorted(args.results.rglob('predictor/classifier.pt')):
        if weight.is_symlink():continue
        manifest=weight.parent/'manifest.json';value=json.loads(manifest.read_text())
        checksum=value['files']['classifier.pt']
        destination=args.assets/checksum
        # Verify every bound asset before creating any compatibility links.
        files={**value['files'],'manifest.json':sha(manifest)}
        for name,expected in files.items():
            source=weight.parent/name
            if source.resolve().is_relative_to(weight.parent.resolve()) is False or sha(source)!=expected:
                raise ValueError('predictor source bytes differ: '+str(source))
        for name,expected in files.items():
            source=weight.parent/name;target=destination/name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if not os.path.samefile(source,target):raise ValueError('asset path is not the verified hardlink: '+str(target))
            else:os.link(source,target)
        entries.append(dict(model_id=value.get('model_identity',{}).get('model'),
                            predictor_qualified=value.get('predictor_qualified') is True,
                            asset_path=str(destination.resolve()),compatibility_path=str(weight.parent.resolve()),
                            manifest_sha256=files['manifest.json'],weight_sha256=checksum,
                            weight_bytes=weight.stat().st_size,method='hardlink',reclaimed_bytes=0,
                            files=files))
    args.assets.mkdir(parents=True,exist_ok=True)
    report=dict(schema=1,immutable=True,entries=entries,reclaimed_bytes=0,
                note='Existing result paths are compatibility hardlinks; neither weights nor manifests may be edited in place.')
    (args.assets/'index.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(entries=len(entries),reclaimed_bytes=0,index=str((args.assets/'index.json').resolve()))))


if __name__=='__main__':main()

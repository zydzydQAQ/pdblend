"""Conservative transport envelopes with independent correctness provenance.

Merge repeated matching placements at each measured destination clock. Keep
900 and 2520 MHz separate; unmatched online clocks use a conservative envelope.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .evidence import sha256
from .interconnect import InterconnectTopology
from .transfers import TransferCost, upper_envelope
from .measurement import power_evidence
from .prefill_batch import build as build_transfer_costs


def merge_links(observations,topology):
    grouped={}
    for link,frequency in observations:
        cost=TransferCost(**link)
        if (type(frequency) is not int or frequency<=0
                or cost.decode_frequency_mhz not in (None,frequency)):
            raise ValueError('transfer destination frequency differs from its raw measurement')
        if (not cost.validated or not cost.source_sha256 or not cost.source_gpus or not cost.target_gpus
                or cost.topology_sha256!=topology.source_sha256
                or cost.interconnect_class!=topology.link_class(cost.source_gpus,cost.target_gpus)):
            raise ValueError('unverified transport placement')
        key=(cost.source_tp,cost.target_tp,cost.max_input_tokens,cost.profile_batch,
             cost.interconnect_class,topology.intra_class(cost.source_gpus),topology.intra_class(cost.target_gpus))
        grouped.setdefault(key,[]).append((cost,frequency))
    links=[];gaps=[]
    for key,values in sorted(grouped.items()):
        frequencies={f for _,f in values}
        if not {900,2520}<=frequencies:
            gaps.append(dict(key=key,missing_decode_frequencies=sorted({900,2520}-frequencies)))
            continue
        for frequency in sorted(frequencies):
            links.append(asdict(upper_envelope([c for c,f in values if f==frequency],frequency)))
    return links,gaps


def merge(manifest):
    topo_path=Path(manifest['interconnect']).resolve()
    topology=InterconnectTopology.parse(topo_path.read_text())
    artifacts={str(topo_path):sha256(topo_path)};validated=set();observations=[]
    image_ids=set();engine_sources=set()
    def record(path):
        path=Path(path).resolve();artifacts[str(path)]=sha256(path)
        raw=json.loads(path.read_text())
        if not raw.get('complete') or raw.get('sampling_error'):
            raise ValueError('incomplete transfer evidence: '+str(path))
        if not raw.get('engine_provenance'): raise ValueError('missing engine provenance')
        for engine in raw['engine_provenance']:
            image_ids.add(engine['image_id'])
            engine_sources.add(next((v for p,v in engine['source_files_at_import'].items()
                                     if p.endswith('/serving/engine.py')),None))
        return raw,artifacts[str(path)]
    for path in manifest['diagnostics']:
        raw,_=record(path)
        if not raw.get('passed') or not raw.get('cases') or not all(c['passed'] for c in raw['cases']):
            raise ValueError('transport correctness not established')
        validated.update((c['source_tp'],c['target_tp']) for c in raw['cases'])
    for directory in manifest['cost_directories']:
        root=Path(directory);raw,digest=record(root/'raw.json')
        proof=power_evidence(raw.get('power_samples',[]),raw.get('power_source'),raw.get('power_metadata'))
        if not proof['power_source_verified']:
            raise ValueError('transfer cost certification requires eight-GPU instant power evidence: '+str(root))
        if not raw.get('frequency_samples'): raise ValueError('missing observed transfer clocks')
        frequency=raw['decode_frequency_mhz'];commands=raw.get('commanded_frequencies',{})
        for role,f in (('prefill',2520),('decode',frequency)):
            if any(commands.get(str(g))!=f for g in raw['topology'][role]['gpus']):
                raise ValueError('transfer clock label differs from command')
        # Source hashes identify measurements, but alone cannot certify which
        # endpoint energy was included in a derived cost. Recompute the current
        # two-endpoint accounting from the immutable eight-GPU samples.
        expected_profiles,expected_links=build_transfer_costs(raw,digest,topology)
        profiles_path=(root/'profiles.json').resolve()
        profiles=json.loads(profiles_path.read_text())
        for key in ('transfer_energy_accounting','transfer_energy_components'):
            if key not in expected_profiles or profiles.get(key)!=expected_profiles[key]:
                raise ValueError('transfer cost lacks matching receiver-energy accounting: '+str(root))
        artifacts[str(profiles_path)]=sha256(profiles_path)
        path=(root/'transfers.json').resolve();artifacts[str(path)]=sha256(path)
        derived=json.loads(path.read_text())
        if derived!=expected_links:
            raise ValueError('transfer costs differ from raw two-endpoint energy reconstruction: '+str(root))
        for link in derived:
            if link['source_sha256']!=digest or (link['source_tp'],link['target_tp']) not in validated:
                raise ValueError('transfer cost lacks matching raw/correctness evidence')
            observations.append((link,frequency))
    if (image_ids!={manifest['engine_image']} or len(engine_sources)!=1 or None in engine_sources):
        raise ValueError('mixed runtime versions in transport evidence')
    links,gaps=merge_links(observations,topology)
    artifacts[str(Path(__file__).resolve())]=sha256(Path(__file__).resolve())
    builder_path=Path(build_transfer_costs.__code__.co_filename).resolve()
    artifacts[str(builder_path)]=sha256(builder_path)
    return dict(schema=3,links=links,coverage_gaps=gaps,certified=bool(links) and not gaps,
                instant_power_costs_verified=bool(observations),
                receiver_transfer_energy_included=bool(observations),
                transfer_energy_accounting=expected_profiles['transfer_energy_accounting'] if observations else None,
                engine_image=manifest['engine_image'],certification_artifacts=artifacts,
                method='maximum measured transport cost over matching placements at each decode frequency',
                missing_frequency_policy='componentwise maximum over the closest measured bucket; no interpolation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): parser.error('refusing to overwrite transport evidence')
    result=merge(json.loads(args.manifest.read_text()))
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(certified=result['certified'],links=len(result['links']),gaps=len(result['coverage_gaps']))))


if __name__=='__main__': main()

"""Read-only hardware inspection and immutable input qualification.

The presence of a file called 'certified' is not qualification. All referenced
evidence is checked, and a fresh mechanism proof must match the live engines.
"""
import argparse
import asyncio
import json
import re
import socket
import subprocess
from pathlib import Path

import aiohttp

from .artifacts import object_hash, read_json, sha256, source_files, write_json
from ecopadg.serving.interconnect import InterconnectTopology


REQUIRED_CHECKS = ('mixed_output', 'pd_output', 'frequency_commands',
                   'native_drain', 'cancellation', 'profile_coverage')


def _source_name(path):
    """Normalize module locations without treating a new mount path as new code."""
    path = str(path).replace('\\', '/')
    if '/ecopadg/serving/' in path:
        return 'serving/' + path.split('/ecopadg/serving/', 1)[1]
    if '/vllm/' in path:
        return 'vllm/' + path.split('/vllm/', 1)[1]
    # The self-contained deployed engine exposes its mounted engine.py only.
    if Path(path).name == 'engine.py':
        return 'serving/engine.py'
    return path


def _source_map(record):
    sources = {}
    for field in ('source_files_at_import', 'source_files', 'engine_source'):
        value = record.get(field)
        if value is None:
            continue
        if field == 'engine_source' and isinstance(value, str):
            value = {'engine.py': value}
        elif isinstance(value, dict) and set(('path', 'sha256')) <= set(value):
            value = {value['path']: value['sha256']}
        if not isinstance(value, dict):
            raise ValueError('invalid source hash map: ' + field)
        for path, digest in value.items():
            if isinstance(digest, dict):
                digest = digest.get('sha256')
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError('invalid SHA-256 in ' + field)
            name = _source_name(path)
            if name in sources and sources[name] != digest:
                raise ValueError('conflicting engine source hashes: ' + name)
            sources[name] = digest
    if 'serving/engine.py' not in sources:
        raise ValueError('missing actual engine.py source hash')
    return sources


def _measurement_engines(raw):
    """Read real legacy provenance and the equivalent explicit new raw shape."""
    if not isinstance(raw, dict):
        return []
    for field in ('engine_provenance', 'engine_identities', 'engines'):
        value = raw.get(field)
        if isinstance(value, dict):
            value = list(value.values())
        if isinstance(value, list):
            return [r.get('provenance', r) for r in value if isinstance(r, dict)]
    if any(field in raw for field in ('source_files_at_import', 'source_files', 'engine_source')):
        return [raw]
    return []


def measured_source_errors(config, live_records):
    """Bind original profile/transfer measurements to the actual loaded code.

    A fresh mechanism probe cannot repair a historical measurement's source
    identity. Each TP1 producer digest must resolve inside the corresponding
    frozen artifact index to raw JSON with engine provenance. Existing raw
    provenance is also checked, including numerical-transfer validation raw.
    """
    errors = []
    live_maps = []
    for record in live_records:
        try:
            live_maps.append((record, _source_map(record)))
        except ValueError as exc:
            errors.append('live engine source evidence invalid: ' + str(exc))
    if not live_maps:
        return errors + ['no live source identities to compare with measured inputs']
    for label, key in (('profile', 'profiles'), ('transfer', 'transfer_evidence')):
        path = config.get(key)
        if not path or not Path(path).is_file():
            errors.append(label + ' source evidence document unavailable')
            continue
        document = read_json(path)
        references = document.get('certification_artifacts', {})
        if not isinstance(references, dict):
            errors.append(label + ' certification artifact index is invalid')
            continue
        by_hash = {}
        for artifact_path, digest in references.items():
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                errors.append(label + ' has an invalid certification artifact SHA-256')
                continue
            by_hash.setdefault(digest, []).append(artifact_path)
        producers = set()
        for row in document.get('points' if label == 'profile' else 'links', []):
            relevant = row.get('tp') == 1 if label == 'profile' else row.get('source_tp') == row.get('target_tp') == 1
            if relevant:
                digest = row.get('source_sha256')
                if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                    errors.append(label + ' TP1 measurement lacks a valid producer SHA-256')
                else:
                    producers.add(digest)
        unmapped = producers - set(by_hash)
        if unmapped:
            errors.append(label + ' TP1 producer hashes absent from artifact index: ' + ', '.join(sorted(unmapped)))
        witnessed = set()
        found_provenance = False
        mismatch = False
        for artifact_path, expected_hash in references.items():
            artifact = Path(artifact_path)
            if artifact.suffix != '.json' or not artifact.is_file() or sha256(artifact) != expected_hash:
                continue  # Missing/changed artifacts are separate unconditional blockers.
            try:
                raw = read_json(artifact)
            except (ValueError, OSError) as exc:
                errors.append(label + ' invalid raw JSON source evidence: ' + str(artifact) + ': ' + str(exc))
                continue
            engines = _measurement_engines(raw)
            if not engines:
                continue
            found_provenance = True
            if raw.get('complete', raw.get('passed')) is not True:
                errors.append(label + ' source-bearing raw measurement is incomplete: ' + str(artifact))
                mismatch = True
                break
            try:
                for measured in engines:
                    measured_sources = _source_map(measured)
                    image = measured.get('image_id', measured.get('engine_image', raw.get('engine_image')))
                    if not image:
                        raise ValueError('raw source record lacks engine image identity')
                    for live, live_sources in live_maps:
                        if live.get('image_id') != image:
                            raise ValueError('measured engine image differs from live engine')
                        missing = set(live_sources) - set(measured_sources)
                        changed = {name for name in set(live_sources) & set(measured_sources)
                                   if live_sources[name] != measured_sources[name]}
                        if missing or changed:
                            raise ValueError('measured source differs from live engine '
                                + str(live.get('instance_id')) + '; missing=' + ','.join(sorted(missing))
                                + '; changed=' + ','.join(sorted(changed)))
                witnessed.add(expected_hash)
            except (ValueError, TypeError) as exc:
                errors.append(label + ' engine-source mismatch: ' + str(artifact) + ': ' + str(exc))
                mismatch = True
                break  # The group already fails; do not parse hundreds of MB needlessly.
        if not found_provenance:
            errors.append(label + ' has no original raw engine-source provenance')
        if not mismatch:
            unwitnessed = producers - witnessed
            if unwitnessed:
                errors.append(label + ' TP1 producers lack matching original raw source evidence: '
                              + ', '.join(sorted(unwitnessed)))
    return list(dict.fromkeys(errors))


def topology_errors(config, live_text):
    try:
        frozen = InterconnectTopology.parse(Path(config['interconnect']).read_text())
        live = InterconnectTopology.parse(live_text)
        errors = [] if frozen.matrix == live.matrix else ['live GPU topology differs from configured interconnect matrix']
        for link in config.get('transfers', []):
            if (link.get('topology_sha256') != frozen.source_sha256 or
                    link.get('interconnect_class') != frozen.link_class(link['source_gpus'], link['target_gpus'])):
                errors.append('configured transfer is not bound to the frozen physical topology')
        return list(dict.fromkeys(errors))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        return ['cannot verify live/frozen GPU topology: ' + str(exc)]


async def read_live_topology():
    process = await asyncio.create_subprocess_exec('nvidia-smi', 'topo', '-m',
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(stderr.decode()[:500])
    return stdout.decode()


def qualification_errors(proof, *, engine_identities, profile_sha256, frozen_files=None):
    errors = []
    if proof.get('scope') != 'hardware_qualification' or proof.get('passed') is not True:
        errors.append('fresh hardware qualification did not pass')
    if proof.get('smoke') is True or proof.get('cleanup_complete') is not True:
        errors.append('qualification is smoke or lacks verified cleanup')
    for name in REQUIRED_CHECKS:
        if proof.get('checks', {}).get(name) is not True:
            errors.append('fresh mechanism check missing: ' + name)
    if proof.get('profile_sha256') != profile_sha256:
        errors.append('qualification profile changed')
    if not engine_identities or proof.get('engine_identities') != engine_identities:
        errors.append('qualification differs from live engine identities')
    artifacts = proof.get('artifacts', {})
    if not isinstance(artifacts, dict) or not artifacts:
        return errors + ['qualification has no raw evidence']
    matching_raw = False
    for path, digest in artifacts.items():
        if frozen_files is not None and frozen_files.get(path) != digest:
            errors.append('qualification artifact is not covered by freeze: ' + path)
        if not Path(path).is_file() or sha256(path) != digest:
            errors.append('qualification evidence changed: ' + path)
            continue
        if Path(path).suffix != '.json':
            continue
        try:
            raw = read_json(path)
        except (ValueError, OSError):
            continue
        if not isinstance(raw, dict) or raw.get('scope') != 'diagnostic_raw':
            continue
        if (raw.get('measurement') == 'hardware' and raw.get('passed') is True
                and raw.get('smoke') is not True and raw.get('cleanup_complete') is True
                and raw.get('engine_identities') == engine_identities
                and all(raw.get('checks', {}).get(k) is True for k in REQUIRED_CHECKS)):
            matching_raw = True
        else:
            errors.append('qualification raw result does not support the passing proof: ' + path)
    if not matching_raw:
        errors.append('qualification lacks matching successful original hardware raw evidence')
    return errors


def reference_errors(document):
    errors = []
    references = document.get('certification_artifacts', {})
    if not references:
        return ['missing certification artifact references']
    for path, digest in references.items():
        if not Path(path).is_file():
            errors.append('missing evidence: ' + path)
        elif sha256(path) != digest:
            errors.append('changed evidence: ' + path)
    return errors


def check_profile_inputs(config):
    errors = []
    profiles = read_json(config['profiles'])
    if profiles.get('schema') != 2 or profiles.get('measurement') != 'hardware':
        errors.append('profiles are not schema-2 hardware measurements')
    for field in ('frequency_commands_verified', 'mixed_interference_measured',
                  'heldout_calibration_complete', 'instant_prefill_calibration_complete',
                  'instant_heldout_calibration_complete', 'resident_idle_measured'):
        if profiles.get(field) is not True:
            errors.append('profile qualification missing: ' + field)
    errors.extend(reference_errors(profiles))
    points = profiles.get('points', [])
    for role in ('mixed', 'prefill', 'decode'):
        frequencies = {p['frequency_mhz'] for p in points if p['tp'] == 1 and p['role'] == role}
        if frequencies != {900, 1500, 2100, 2520}:
            errors.append('TP1 frequency coverage differs for ' + role)
    transfer_path = config.get('transfer_evidence')
    if not transfer_path or not Path(transfer_path).is_file():
        errors.append('missing measured transfer evidence')
    else:
        transfers = read_json(transfer_path)
        errors.extend(reference_errors(transfers))
        for field in ('certified', 'instant_power_costs_verified', 'receiver_transfer_energy_included'):
            if transfers.get(field) is not True:
                errors.append('transfer qualification missing: ' + field)
        if transfers.get('engine_image') != profiles.get('engine_image'):
            errors.append('profile and transfer engine images differ')
        if config.get('transfers') != transfers.get('links'):
            errors.append('configured transfers differ from measured transfer table')
    if not config.get('frequency_costs') or not config.get('frequency_evidence'):
        errors.append('missing frequency transition measurements')
    else:
        from ecopadg.serving.frequency import verify_frozen_costs
        try:
            paths = [str(Path(p).resolve()) for p in config['frequency_evidence']]
            cost_freeze = dict(groups={'profiles': paths}, files={p:sha256(p) for p in paths},
                               identities={'engine_image': profiles.get('engine_image')})
            verify_frozen_costs(dict(config,power_mode='instant'),profiles,cost_freeze)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append('invalid measured frequency transitions: '+str(exc))
    return errors


def engine_identity(record):
    # PIDs, generations and instance IDs may change between independent runs;
    # executable source, image, model and physical configuration must not.
    return {k: record.get(k) for k in ('instance_id', 'tp', 'dtype', 'model',
        'max_model_len', 'cuda_visible_devices', 'source_files_at_import', 'image_id')}


def quiescent(state):
    return (not any(state.get(k) for k in ('active', 'running', 'waiting',
                'kv_allocations', 'transfer_allocations', 'transfer_buffered_tensors',
                'transfer_inflight_receives', 'transfer_inflight_sends'))
            and not state.get('error') and not state.get('runtime_error')
            and state.get('transport_healthy', True))


async def inspect_engines(config):
    records = []
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=10)) as session:
        for instance in config['instances']:
            row = {'instance_id': instance['id'], 'errors': []}
            try:
                async with session.get(instance['url'] + '/provenance') as response:
                    response.raise_for_status()
                    row.update(await response.json())
                async with session.get(instance['url'] + '/runtime') as response:
                    response.raise_for_status()
                    row['runtime'] = await response.json()
                container = instance.get('container', 'pdb-v2-' + instance['id'])
                proc = await asyncio.create_subprocess_exec('docker', 'inspect', '--format', '{{.Image}}',
                    container, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                try:
                    out, err = await asyncio.wait_for(proc.communicate(), 10)
                except BaseException:
                    if proc.returncode is None:
                        proc.kill()
                        await proc.wait()
                    raise
                if proc.returncode:
                    raise RuntimeError(err.decode()[:500])
                row['image_id'] = out.decode().strip()
                if row.get('tp') != 1 or row.get('dtype') != 'bfloat16':
                    row['errors'].append('requires TP1 bfloat16')
                if row.get('cuda_visible_devices') != ','.join(map(str, instance['gpus'])):
                    row['errors'].append('physical GPU identity mismatch')
                if row.get('model', '').rstrip('/').split('/')[-1] != 'Qwen2.5-14B-Instruct':
                    row['errors'].append('model mismatch')
                if not quiescent(row['runtime']):
                    row['errors'].append('engine is not healthy and drained')
                if not row.get('source_files_at_import'):
                    row['errors'].append('engine source identity unavailable')
                for path, digest in row.get('source_files_at_import', {}).items():
                    if not Path(path).is_file() or sha256(path) != digest:
                        row['errors'].append('live engine source changed: ' + path)
            except Exception as exc:
                row['errors'].append(type(exc).__name__ + ': ' + str(exc))
            records.append(row)
    return records


async def inspect(config, *, qualification=None):
    errors = check_profile_inputs(config)
    records = await inspect_engines(config)
    errors.extend(f"{r['instance_id']}: {e}" for r in records for e in r['errors'])
    errors.extend(await asyncio.to_thread(measured_source_errors, config, records))
    topology_observation = None
    try:
        topology_observation = await read_live_topology()
        errors.extend(topology_errors(config, topology_observation))
    except (OSError, RuntimeError, asyncio.TimeoutError) as exc:
        errors.append('live topology inspection failed: ' + str(exc))
    profile = read_json(config['profiles'])
    if any(r.get('image_id') != profile.get('engine_image') for r in records):
        errors.append('live image differs from measured profile image')
    if not qualification:
        errors.append('fresh mechanism qualification has not been supplied')
    else:
        proof = read_json(qualification)
        expected = {r['instance_id']: engine_identity(r) for r in records}
        errors.extend(qualification_errors(proof, engine_identities=expected,
                                           profile_sha256=sha256(config['profiles'])))
    return dict(scope='hardware_preflight', host_id=socket.gethostname(), passed=not errors,
                errors=errors, engines=records, profile_sha256=sha256(config['profiles']),
                live_topology=topology_observation)


def create_freeze(config_path, pool_paths, qualification_path, inspection, out):
    if inspection.get('passed') is not True:
        raise ValueError('hardware preflight must pass before freezing formal inputs')
    config = read_json(config_path)
    paths = set(source_files())
    paths.update(Path(p).resolve() for p in [config_path, qualification_path, *pool_paths,
                                          config['profiles'], config['transfer_evidence'], config['interconnect']])
    for name in ('profiles', 'transfer_evidence'):
        paths.update(Path(p).resolve() for p in read_json(config[name])['certification_artifacts'])
    paths.update(Path(p).resolve() for p in config['frequency_evidence'])
    proof = read_json(qualification_path)
    paths.update(Path(p).resolve() for p in proof['artifacts'])
    model_root = Path(config.get('tokenizer', '/root/workspace/models/Qwen2.5-14B-Instruct'))
    index = model_root / 'model.safetensors.index.json'
    weights = set(read_json(index)['weight_map'].values())
    paths.update(model_root / p for p in weights | {'config.json', 'tokenizer.json',
                        'tokenizer_config.json', 'model.safetensors.index.json'})
    for record in inspection['engines']:
        paths.update(Path(p).resolve() for p in record['source_files_at_import'])
    files = {str(p.resolve()): sha256(p) for p in sorted(paths)}
    result = dict(schema='pdblend-scalability-freeze-v1', host_id=inspection['host_id'],
        files=files, config_path=str(Path(config_path).resolve()),
        profile_sha256=sha256(config['profiles']), source_config_sha256=object_hash(config),
        pool_paths=[str(Path(p).resolve()) for p in pool_paths],
        qualification_path=str(Path(qualification_path).resolve()),
        engine_identities={r['instance_id']: engine_identity(r) for r in inspection['engines']})
    datasets = {}
    for path in pool_paths:
        payload = read_json(path)
        dataset = payload.get('dataset') if isinstance(payload,dict) else None
        if dataset not in ('sharegpt','longbench') or dataset in datasets:
            raise ValueError('freeze pools require explicit unique dataset labels')
        datasets[dataset] = str(Path(path).resolve())
    if set(datasets) != {'sharegpt','longbench'}:
        raise ValueError('freeze requires both dataset pools')
    result['pools'] = datasets
    write_json(out, result)
    return result


def verify_freeze(freeze, *, require_local_host=True):
    errors = []
    if freeze.get('schema') != 'pdblend-scalability-freeze-v1' or not freeze.get('files'):
        return ['invalid or empty source freeze']
    if require_local_host and freeze.get('host_id') != socket.gethostname():
        errors.append('freeze belongs to another host')
    for path, digest in freeze['files'].items():
        if not Path(path).is_file() or sha256(path) != digest:
            errors.append('frozen file missing or changed: ' + path)
    files = freeze['files']
    def frozen_json(path, label):
        if not isinstance(path, str) or path not in files or not Path(path).is_file():
            errors.append(label + ' is absent from the frozen file index')
            return None
        if sha256(path) != files[path]:
            return None
        try:
            value = read_json(path)
            if not isinstance(value, dict):
                raise ValueError('expected JSON object')
            return value
        except (ValueError, OSError) as exc:
            errors.append(label + ' is not readable frozen JSON: ' + str(exc))
            return None
    config = frozen_json(freeze.get('config_path'), 'config_path')
    proof = frozen_json(freeze.get('qualification_path'), 'qualification_path')
    identities = freeze.get('engine_identities')
    if not isinstance(identities, dict) or not identities:
        errors.append('freeze has no explicit engine source identities')
    else:
        for instance_id, record in identities.items():
            imports = record.get('source_files_at_import', {}) if isinstance(record, dict) else {}
            if not imports or any(files.get(path) != digest for path, digest in imports.items()):
                errors.append('engine imported source is absent from freeze: ' + str(instance_id))
    pools = freeze.get('pools')
    if not isinstance(pools, dict) or set(pools) != {'sharegpt', 'longbench'}:
        errors.append('freeze requires explicit ShareGPT and LongBench pool paths')
    else:
        if set(freeze.get('pool_paths', [])) != set(pools.values()):
            errors.append('pool_paths differ from the explicit dataset pool map')
        for dataset, path in pools.items():
            payload = frozen_json(path, dataset + ' pool')
            if payload is not None and payload.get('dataset') != dataset:
                errors.append('frozen pool dataset identity differs: ' + dataset)
    if config is not None:
        if freeze.get('source_config_sha256') != object_hash(config):
            errors.append('frozen configuration object hash differs')
        profile_path = config.get('profiles')
        if not isinstance(profile_path, str) or profile_path not in files:
            errors.append('profile path is absent from the frozen file index')
        elif freeze.get('profile_sha256') != files[profile_path]:
            errors.append('freeze profile hash differs from its frozen file')
        for key in ('transfer_evidence', 'interconnect'):
            path = config.get(key)
            if not isinstance(path, str) or path not in files:
                errors.append(key + ' is absent from the frozen file index')
        for key in ('profiles', 'transfer_evidence'):
            document = frozen_json(config.get(key), key)
            if document is not None:
                for path, digest in document.get('certification_artifacts', {}).items():
                    if files.get(path) != digest:
                        errors.append('measured input artifact is absent from freeze: ' + path)
    if proof is not None:
        errors.extend(qualification_errors(proof,
            engine_identities=freeze.get('engine_identities'), profile_sha256=freeze.get('profile_sha256'),
            frozen_files=files))
    if not errors and config is not None:
        errors.extend(measured_source_errors(config, list(freeze['engine_identities'].values())))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--pool', type=Path, action='append', default=[])
    args = parser.parse_args()
    result = asyncio.run(inspect(read_json(args.config), qualification=args.qualification))
    write_json(args.out, result)
    if args.freeze:
        if not args.qualification or not args.pool:
            parser.error('--freeze requires --qualification and at least one --pool')
        create_freeze(args.config, args.pool, args.qualification, result, args.freeze)
    print(json.dumps({'passed': result['passed'], 'errors': len(result['errors']), 'out': str(args.out)}))
    raise SystemExit(0 if result['passed'] else 2)


if __name__ == '__main__':
    main()

"""Independent saved-observer transport and original power-file equivalence audit."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT/'A/isolated-power-v2/manifest.json'
ADAPTER_SHA = 'af64c1ba352fcc72f5bd74f8f6870bd70e347c76185ee404a9d2119b733cca85'
HOOK = ROOT/'A/dynamic-execution-isolated-power-002/sampler_hooks.py'
HOOK_SHA = 'af82b3896bf970238ab62230814166ebcc6e2b650fda9f021d552650cc646eae'
EXECUTOR = ROOT/'A/dynamic-execution-isolated-power-002/manifest.json'
EXECUTOR_SHA = 'cf4e14ab8351d2ccb7384161334f60fc3f6e6481e6e5fecdfbf2bf82205f2dc7'

def need(value, message):
    if not value: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def ref(path): return dict(path=str(path), sha256=sha(path))
def fixed(reference):
    need(sha(reference['path']) == reference['sha256'], 'isolated evidence reference changed')
    return read(reference['path'])

def sources():
    for path, digest in ((ADAPTER, ADAPTER_SHA), (HOOK, HOOK_SHA), (EXECUTOR, EXECUTOR_SHA)):
        need(sha(path) == digest, 'independent sampler contract source changed')
    for manifest in (ADAPTER, EXECUTOR):
        need(all(sha(p) == digest for p, digest in read(manifest)['files'].items()),
             'frozen measurement dependency changed')
    spec = importlib.util.spec_from_file_location('saved_isolated_sampler_hooks', HOOK)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def transport(spec, receipt, raw, host):
    """Rebuild the original worker's byte-stream digest from every retained row."""
    terminal = receipt['terminal']; n = receipt['raw_sample_count']
    need(spec['gpus'] == list(range(8)) and spec['interval'] == .02 and spec['read_only'] is True,
         'isolated observer changed original all8/50Hz scope')
    need(receipt['complete'] is True and receipt['child_exited'] is True
         and receipt['reader_stopped'] is True and receipt['returncode'] == 0
         and receipt['error'] is None and raw['sampling_error'] is None,
         'isolated observer did not exit cleanly')
    counts = dict(samples=len(raw['samples']), metadata=len(raw['metadata']),
                  utilization=len(raw['utilization']), frequency=len(raw['frequency']))
    need(type(n) is int and n >= 2 and counts['samples'] == counts['metadata'] == counts['utilization'] == n
         and counts['frequency'] == (n if spec['sample_clocks'] else 0)
         and terminal['counts'] == counts and terminal['emitted'] == n
         and terminal['pid'] == receipt['pid'] and terminal['sampler_thread_stopped'] is True
         and terminal['error'] is None, 'isolated observer dropped or relabeled raw rows')
    source = raw['power_source']
    need(source.get('mode') == 'instant' and source.get('field_id') == 186
         and source.get('scope_id') == 0 and source.get('unit') == 'W', 'original NVML power source changed')
    digest = hashlib.sha256()
    def append(value):
        digest.update((json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n').encode())
    append(dict(kind='ready', pid=receipt['pid'], source=source, host_manifest=spec['host_manifest'],
                original_sampler_source=host['files']['src/ecopadg/measure/power.py'], read_only=True,
                garbage_collection_disabled_only_in_sampler=True))
    last = None
    for index in range(n):
        power = raw['samples'][index]; metadata = raw['metadata'][index]
        util = raw['utilization'][index]
        frequency = raw['frequency'][index] if spec['sample_clocks'] else None
        need(len(power) == 2 and len(power[1]) == 8 and math.isfinite(power[0])
             and all(math.isfinite(v) and v >= 0 for v in power[1])
             and (last is None or power[0] > last), 'invalid or reordered power row')
        need(metadata['t_s'] == power[0] and metadata['gpus'] == list(range(8))
             and len(util) == 2 and util[0] == power[0] and len(util[1]) == 8
             and (frequency is None or len(frequency) == 2 and frequency[0] == power[0] and len(frequency[1]) == 8),
             'isolated row telemetry alignment changed')
        append(dict(kind='row', index=index, power=power, metadata=metadata, utilization=util, frequency=frequency))
        last = power[0]
    need(digest.hexdigest() == terminal['stream_sha256'], 'independent isolated IPC digest differs')
    return dict(samples=n, ipc_sha256=digest.hexdigest(), stream_reconstructed=True,
                actual_worker_exited=True, first_sample_s=raw['samples'][0][0], last_sample_s=last)

def audit_samplers(references, host_ref, *, artifacts=None):
    hooks = sources(); host = fixed(host_ref)
    host_root = Path(host_ref['path']).parent
    need(all(sha(host_root/name) == digest for name, digest in host['files'].items()), 'actual host source changed')
    need(isinstance(references, list) and references, 'isolated observer references missing')
    roots = {Path(row['directory']) for row in references}
    need(len(roots) == len(references), 'duplicate isolated observer')
    adapter_ref = dict(path=str(ADAPTER), sha256=ADAPTER_SHA)
    expected = hooks.sampler_references(roots)
    need(sorted(references, key=lambda r:r['directory']) == expected, 'isolated observer reference mapping changed')
    sampler_files = hooks.completed_artifacts(roots, host_ref, adapter_ref)
    if artifacts is not None:
        need(all(artifacts.get(p) == digest for p, digest in sampler_files.items()),
             'isolated observer files not frozen in containing artifact set')
    proofs = []; values = {}
    for row in references:
        raw, receipt, spec = fixed(row['raw']), fixed(row['receipt']), fixed(row['spec'])
        proof = transport(spec, receipt, raw, host)
        proofs.append(dict(directory=row['directory'], raw=row['raw'], receipt=row['receipt'], **proof))
        values[row['directory']] = raw
    return dict(artifacts=sampler_files, proofs=proofs, raw_values=values,
                measurement_adapter=adapter_ref, hook_source=dict(path=str(HOOK), sha256=HOOK_SHA))

def match_power_directory(directory, audited, *, artifacts=None):
    directory = Path(directory)
    required = [directory/'power.csv', directory/'power_metadata.jsonl', directory/'power_source.json']
    if artifacts is not None:
        need(all(artifacts.get(str(p)) == sha(p) for p in required), 'original main power files not frozen')
    with required[0].open() as handle:
        reader = csv.DictReader(handle)
        names = ['t_s']+[f'gpu{i}_w' for i in range(8)]+[f'gpu{i}_util_pct' for i in range(8)]
        need(reader.fieldnames == names, 'original eight-GPU CSV columns changed')
        rows = list(reader)
    power = [[float(row['t_s']), [float(row[f'gpu{i}_w']) for i in range(8)]] for row in rows]
    utilization = [[float(row['t_s']), [float(row[f'gpu{i}_util_pct']) for i in range(8)]] for row in rows]
    metadata = [json.loads(line) for line in required[1].read_text().splitlines()]
    source = read(required[2])
    matches = [name for name, raw in audited['raw_values'].items()
               if raw['samples'] == power and raw['metadata'] == metadata
               and raw['power_source'] == source and raw['utilization'] == utilization]
    need(len(matches) == 1, 'main power/metadata/utilization is not exactly one original observer stream')
    name = matches[0]; raw = audited['raw_values'][name]
    clocks = directory/'clocks.json'
    if clocks.exists():
        need(read(clocks) == raw['frequency'], 'transition clock stream differs from isolated raw')
        if artifacts is not None: need(artifacts.get(str(clocks)) == sha(clocks), 'clock artifact not frozen')
    elif (directory/'clocks.csv').exists():
        clocks = directory/'clocks.csv'
        with clocks.open() as handle:
            reader = csv.DictReader(handle)
            clock_rows = [[float(r['t_s']), [float(r[f'gpu{i}_sm_mhz']) for i in range(8)]] for r in reader]
        need(clock_rows == raw['frequency'], 'outer clock stream differs from isolated raw')
        if artifacts is not None: need(artifacts.get(str(clocks)) == sha(clocks), 'clock artifact not frozen')
    return dict(power_directory=str(directory), isolated_directory=name, all_original_rows_exact=True,
                power=ref(required[0]), metadata=ref(required[1]), source=ref(required[2]))

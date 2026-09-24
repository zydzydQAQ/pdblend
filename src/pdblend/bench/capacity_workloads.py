"""Freeze independent capacity workloads; never execute or enqueue GPU jobs.

A family fixes corpus/split, rate anchor, three seeds, SLO and a common service
duration before any rate is measured. Every system at the same rate/repeat gets
the identical trace. Existing evaluation/150s dispatchers cannot execute this
schema: their separate integration and acceptance work remains explicit.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from pathlib import Path
import sys
from types import SimpleNamespace

from . import client
from .client import CONTEXT, MAX_INPUT, MAX_OUTPUT, SLOS, load_split, poisson_trace
from .first_batch import load_anchor
from .resident_session import digest, write_new
from .slo_capacity import CapacityConfig


FAMILY_SCHEMA = 'pdblend-capacity-workload-family/v1'
TRACE_SCHEMA = 'pdblend-independent-capacity-trace/v1'
PROTOCOL = 'native-independent-capacity-workload/v1'
SYSTEMS = ('mixed', 'distserve', 'ecoserve', 'dynamollm', 'pdblend')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, name):
    need(type(value) in (int, float) and math.isfinite(value) and value > 0,
         name + ' must be finite and positive')


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def bound_bytes(ref):
    need(isinstance(ref, dict) and isinstance(ref.get('path'), str)
         and Path(ref['path']).is_absolute() and isinstance(ref.get('sha256'), str),
         'absolute path/SHA256 input binding required')
    raw = Path(ref['path']).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'capacity workload input checksum differs: '+ref['path'])
    return raw


def read_bound(ref):
    return json.loads(bound_bytes(ref))


def replay_source(client_source):
    """Archive the existing pure generator, without importing runtime/GPU code."""
    source = client_source.decode()
    tree = ast.parse(source)
    names = ('Request', 'load_split', 'poisson_trace')
    definitions = {node.name: ast.get_source_segment(source, node) for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names}
    need(set(definitions) == set(names), 'Poisson generator source inventory differs')
    # get_source_segment omits the Request decorator; preserve its dataclass contract.
    limits = next((ast.get_source_segment(source, node) for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Tuple) and
            any(isinstance(item, ast.Name) and item.id == 'MAX_OUTPUT' for item in target.elts)
            for target in node.targets)), None)
    need(limits is not None, 'Poisson output limit source is absent')
    return ('# Frozen pure subset of the bound client.py; original source segments follow.\n'
        'import json\nimport random\nfrom pathlib import Path\nfrom dataclasses import dataclass\n\n'
        + limits+'\n\n@dataclass\n'+definitions['Request']+'\n\n'
        +definitions['load_split']+'\n\n'+definitions['poisson_trace']+'\n').encode()


# This approved v1 source is audit evidence only. Never execute artifact code.
REPLAYER_VERSION = 'pdblend-poisson-stdlib/v1'
REPLAYER_V1_SOURCE = b'# Frozen pure subset of the bound client.py; original source segments follow.\nimport json\nimport random\nfrom pathlib import Path\nfrom dataclasses import dataclass\n\nMAX_INPUT, MAX_OUTPUT, CONTEXT = 7168, 512, 8192\n\n@dataclass\nclass Request:\n    idx: int\n    arrival_s: float\n    prompt: list\n    max_tokens: int\n    source: str = ""\n\n    @property\n    def input_tokens(self) -> int:\n        return len(self.prompt)\n\ndef load_split(corpus_root: Path, dataset: str, split: str = "evaluation") -> list[dict]:\n    data = json.loads((Path(corpus_root) / f"{dataset}.json").read_text())\n    return [r for r in data[split] if r["output_tokens"] >= 2]\n\ndef poisson_trace(records: list[dict], rate_rps: float, duration_s: float, seed: int, source: str = "") -> list[Request]:\n    content, arrival = random.Random(seed * 7919 + 1), random.Random(seed)\n    out, t = [], arrival.expovariate(rate_rps)\n    while t < duration_s:\n        r = content.choice(records)\n        out.append(Request(len(out), t, r["prompt"], min(r["output_tokens"], MAX_OUTPUT), source))\n        t += arrival.expovariate(rate_rps)\n    return out\n'


@dataclass
class _V1Request:
    idx: int
    arrival_s: float
    prompt: list
    max_tokens: int
    source: str = ''


def _v1_load_split(corpus_root, dataset, split):
    data = json.loads((Path(corpus_root)/(dataset+'.json')).read_text())
    return [row for row in data[split] if row['output_tokens'] >= 2]


def _v1_poisson_trace(records, rate_rps, duration_s, seed, source=''):
    # Keep this version stable; an algorithm change requires a new version.
    content, arrival = random.Random(seed * 7919 + 1), random.Random(seed)
    out, time = [], arrival.expovariate(rate_rps)
    while time < duration_s:
        row = content.choice(records)
        out.append(_V1Request(len(out), time, row['prompt'], min(row['output_tokens'], 512), source))
        time += arrival.expovariate(rate_rps)
    return out


def freeze_generator(directory):
    directory = Path(directory)
    directory.mkdir(parents=True)
    sources = {'client.py': Path(client.__file__).read_bytes(),
               'capacity_workloads.py': Path(__file__).read_bytes()}
    need(replay_source(sources['client.py']) == REPLAYER_V1_SOURCE,
         'current Poisson generator needs an explicitly supported replayer version')
    sources['workload_replayer.py'] = REPLAYER_V1_SOURCE
    for name, raw in sources.items():
        with (directory/name).open('xb') as stream:
            stream.write(raw)
    return dict(poisson_module=binding(directory/'client.py'),
        preparer=binding(directory/'capacity_workloads.py'), replayer=binding(directory/'workload_replayer.py'),
        python_version=list(sys.version_info[:3]), algorithm='pdblend.bench.client.poisson_trace',
        version=REPLAYER_VERSION)


def load_replayer(generator):
    """Check frozen known bytes, then call trusted builtins; never exec artifacts."""
    need(generator.get('version') == REPLAYER_VERSION, 'unsupported frozen Poisson replayer version')
    original = bound_bytes(generator['poisson_module'])
    bound_bytes(generator['preparer'])
    source = bound_bytes(generator['replayer'])
    need(source == REPLAYER_V1_SOURCE and replay_source(original) == REPLAYER_V1_SOURCE,
         'frozen replayer is not the approved v1 implementation')
    need(generator.get('algorithm') == 'pdblend.bench.client.poisson_trace', 'generator algorithm differs')
    need(generator.get('python_version') == list(sys.version_info[:3]),
         'frozen generator requires its recorded Python version for stdlib RNG replay')
    # No imports, eval or exec of supplied paths. Archive hashes bind provenance;
    # current mutable workspace source is deliberately irrelevant to replay.
    return SimpleNamespace(load_split=_v1_load_split, poisson_trace=_v1_poisson_trace)


def corpus_inputs(corpus, dataset, split, *, loader=load_split):
    need(dataset in SLOS and split in ('calibration', 'tuning'),
         'capacity workload requires a named calibration/tuning corpus split, never evaluation')
    corpus = Path(corpus).resolve()
    manifest_ref, data_ref = binding(corpus/'manifest.json'), binding(corpus/(dataset+'.json'))
    manifest, data = read_bound(manifest_ref), read_bound(data_ref)
    need(manifest.get('complete') is True and manifest.get('model_name')
         and manifest.get('dataset_sha256', {}).get(dataset) == data_ref['sha256'],
         'complete model-owned prepared corpus with exact dataset checksum required')
    need(data.get('model_name') == manifest['model_name'] and data.get('dataset') == dataset,
         'prepared dataset/model identity differs')
    records = loader(corpus, dataset, split)
    need(binding(data_ref['path']) == data_ref and binding(manifest_ref['path']) == manifest_ref,
         'corpus changed while selecting its split')
    need(records and all(isinstance(row.get('prompt'), list) and 0 < len(row['prompt']) <= MAX_INPUT
         and all(type(token) is int and token >= 0 for token in row['prompt'])
         and type(row.get('output_tokens')) is int and row['output_tokens'] >= 2
         and len(row['prompt']) + min(row['output_tokens'], MAX_OUTPUT) <= CONTEXT for row in records),
         'selected prepared split contains invalid request shapes')
    for key in ('tokenizer_sha256', 'model_config_sha256'):
        need(isinstance(manifest.get(key), str) and manifest[key], 'prepared corpus lacks '+key)
    return records, manifest, dict(corpus_manifest=manifest_ref, corpus_dataset=data_ref,
        selected_split=split, selected_split_sha256=digest(records), selected_split_records=len(records))


def inherited_anchor_requests(value, dataset, row, confirmation_ref, measured):
    """Resolve only the completed recovery's explicit original raw-request link."""
    need(value.get('schema') == 'longbench-anchor-recovery-v1'
         and value.get('scope') == 'longbench_only_anchor_recovery'
         and dataset in ('alpaca', 'sharegpt'),
         'missing local anchor requests require explicit supported inherited provenance')
    inputs = value['prior_inputs']
    refs = inputs['inherited'][dataset]
    original = read_bound(inputs['receipts']['completion'])
    need(original.get('status') == 'failed' and original.get('complete') is False
         and original.get('error') == 'RuntimeError: longbench: no independent tuning confirmation passed'
         and original.get('cleanup_errors') == [] and original.get('hardware_executed') is True
         and original.get('selection_splits') == ['calibration', 'tuning']
         and original.get('evaluation_used_for_selection') is False
         and digest(original.get('anchors', {}).get(dataset)) == digest(row),
         'inherited anchor does not match the clean original partial recovery')
    for key in ('model_id', 'model_hash', 'tokenizer_hash', 'corpus_manifest_sha256',
                'corpus_tokenizer_sha256', 'calibration_seed', 'tuning_seed'):
        need(original.get(key) == value.get(key), 'inherited anchor source identity differs: '+key)
    need(refs['completion']['sha256'] == confirmation_ref['sha256']
         and bound_bytes(refs['completion']) == bound_bytes(confirmation_ref),
         'inherited original confirmation bytes differ from completed recovery')
    need(refs['requests']['sha256'] == measured['trace_sha256'],
         'inherited raw requests do not match the confirmation trace checksum')
    bound_bytes(refs['requests'])
    return refs['requests'], dict(completion=inputs['receipts']['completion'], confirmation=refs['completion'])


def independent_anchor(reference, *, corpus, dataset, manifest, corpus_refs, slo,
                       loader=load_split, generator=poisson_trace):
    """Replay anchor request generation so an evaluation trace cannot be relabelled."""
    value = read_bound(reference)
    model = manifest['model_name']
    # Reuse the existing completed Mixed calibration + independent tuning gate.
    original = load_anchor(reference['path'], model, dataset, corpus_refs['corpus_dataset']['sha256'])
    need(binding(reference['path']) == reference, 'anchor changed during validation')
    need(value.get('hardware_executed') is True
         and value.get('corpus_manifest_sha256') == corpus_refs['corpus_manifest']['sha256']
         and value.get('corpus_tokenizer_sha256') == manifest['tokenizer_sha256'],
         'anchor lacks matching actual corpus/tokenizer evidence')
    seeds = [value.get('calibration_seed'), value.get('tuning_seed')]
    need(all(type(seed) is int and seed >= 0 and seed != 701 for seed in seeds)
         and seeds[0] != seeds[1], 'anchor must use independent non-evaluation calibration/tuning seeds')
    row = value['anchors'][dataset]
    positive(row['base_rate_rps'], 'anchor base_rate_rps')
    need(row.get('model_id') == model, 'anchor row belongs to another model')
    confirmation_path = Path(reference['path']).parent/row['confirmation_path'][len('/output/anchor/'):]
    confirmation_ref = dict(path=str(confirmation_path.resolve()), sha256=row['confirmation_sha256'])
    measured = read_bound(confirmation_ref)
    need(measured.get('dataset') == dataset and measured.get('split') == 'tuning'
         and measured.get('seed') == seeds[1] and measured.get('rate_rps') == row['base_rate_rps'],
         'anchor confirmation dataset/split/seed/rate differs')
    positive(measured.get('duration_s'), 'anchor confirmation duration')
    requests_ref = dict(path=str((confirmation_path.parent/'requests.json').resolve()),
                        sha256=measured['trace_sha256'])
    inherited = None
    if not Path(requests_ref['path']).is_file():
        requests_ref, inherited = inherited_anchor_requests(value, dataset, row, confirmation_ref, measured)
    requests = read_bound(requests_ref)
    need(requests.get('seed') == seeds[1] and requests.get('duration_s') == measured['duration_s'],
         'anchor raw request seed/duration differs')
    expected = [asdict(request) for request in generator(loader(corpus, dataset, 'tuning'),
        row['base_rate_rps'], measured['duration_s'], seeds[1], dataset)]
    need(digest(requests.get('requests')) == digest(expected),
         'anchor raw requests do not reproduce from independent tuning; evaluation/seed701 relabel forbidden')
    need(slo == dict(zip(('ttft_s', 'tpot_s'), SLOS[dataset])),
         'anchor uses the declared dataset SLO; a new SLO needs its own rate anchor')
    need(all(isinstance(value.get(key), str) and value[key] for key in ('model_hash', 'tokenizer_hash')),
         'anchor model/tokenizer identity missing')
    result = dict(completion=reference, confirmation=confirmation_ref, requests=requests_ref,
        base_rate_rps=original['base_rate_rps'], calibration_seed=seeds[0], tuning_seed=seeds[1],
        model_hash=value['model_hash'], tokenizer_hash=value['tokenizer_hash'], capacity_exact=False,
        evaluation_used_for_selection=False, confirmation_requests_replayed=True)
    if inherited is not None:
        result['inherited_provenance'] = inherited
    return result


def common_duration(records, *, minimum_rate_rps, base_duration_s, seeds, minimum_requests, dataset,
                    generator=poisson_trace):
    """Use only declared arrival counts, never performance or evaluation results."""
    positive(minimum_rate_rps, 'minimum rate')
    positive(base_duration_s, 'base duration')
    duration = base_duration_s
    steps = []
    for _ in range(64):
        counts = [len(generator(records, minimum_rate_rps, duration, seed, dataset)) for seed in seeds]
        steps.append(dict(duration_s=duration, request_counts=counts))
        if min(counts) >= minimum_requests:
            return duration, steps
        duration *= 2
        positive(duration, 'common duration')
    raise ValueError('common duration needs more than 64 doublings; declare a larger base duration')


def prepare_family(out, *, corpus, dataset, split, anchor, seeds, minimum_scale,
                   base_duration_s=150., minimum_requests=100, systems=SYSTEMS):
    """Freeze one model/dataset family, without selecting any measured capacity."""
    out = Path(out).resolve()
    need(not out.exists(), 'new immutable capacity family directory required')
    positive(minimum_scale, 'minimum scale')
    positive(base_duration_s, 'base duration')
    need(isinstance(seeds, (tuple, list)) and len(seeds) == len(set(seeds)) == 3
         and all(type(seed) is int and seed >= 0 and seed != 701 for seed in seeds),
         'exactly three distinct explicit non-evaluation seeds required; seed701 cannot be relabelled')
    need(isinstance(systems, (tuple, list)) and systems and len(set(systems)) == len(systems)
         and all(system in SYSTEMS for system in systems), 'distinct supported comparison systems required')
    slo = dict(zip(('ttft_s', 'tpot_s'), SLOS[dataset])) if dataset in SLOS else {}
    CapacityConfig('workload-preparation', slo.get('ttft_s', 0), slo.get('tpot_s', 0),
                   min_requests_per_trial=minimum_requests)
    records, manifest, inputs = corpus_inputs(corpus, dataset, split)
    anchor_value = independent_anchor(anchor, corpus=corpus, dataset=dataset, manifest=manifest,
                                     corpus_refs=inputs, slo=slo)
    need(not set(seeds) & {anchor_value['calibration_seed'], anchor_value['tuning_seed']},
         'capacity repetition seeds must be independent of the rate-anchor search and confirmation')
    minimum_rate = minimum_scale * anchor_value['base_rate_rps']
    duration, sizing = common_duration(records, minimum_rate_rps=minimum_rate,
        base_duration_s=base_duration_s, seeds=seeds, minimum_requests=minimum_requests, dataset=dataset)
    need(replay_source(Path(client.__file__).read_bytes()) == REPLAYER_V1_SOURCE,
         'current Poisson generator needs an explicitly supported replayer version')
    out.mkdir(parents=True)
    generator = freeze_generator(out/'sources')
    identity = dict(model_id=manifest['model_name'], dataset=dataset, selection_split=split,
        model_hash=anchor_value['model_hash'], tokenizer_hash=anchor_value['tokenizer_hash'],
        corpus_tokenizer_sha256=manifest['tokenizer_sha256'], model_config_sha256=manifest['model_config_sha256'],
        inputs=inputs, anchor=anchor_value, seeds=list(seeds), systems=list(systems), slo=slo,
        minimum_scale=minimum_scale, base_rate_rps=anchor_value['base_rate_rps'], duration_s=duration,
        base_duration_s=base_duration_s, minimum_requests=minimum_requests,
        generator=generator, measurement_protocol_version=PROTOCOL,
        output_workload='prepared model-tokenized reference length capped at 512; prompt unchanged')
    family = dict(schema=FAMILY_SCHEMA, family_id='capacity-'+digest(identity), identity=identity,
        duration_selection=dict(rule='double common base duration until all seeds at declared minimum scale have enough arrivals',
                                uses_performance=False, steps=sizing),
        evaluation_used_for_selection=False, hardware_executed=False, jobs_enqueued=False,
        execution_ready=False, execution_wiring_pending=True,
        execution_blocker='Existing comparison dispatcher/acceptance is fixed to evaluation, seed701, 150 seconds.',
        capacity_claimed=False, formal_eligible=False)
    write_new(out/'family.json', family)
    return dict(manifest=binding(out/'family.json'), family=family)


def load_family(family_ref):
    """Revalidate all immutable family inputs without reading evaluation metrics."""
    family = read_bound(family_ref)
    need(family.get('schema') == FAMILY_SCHEMA, 'explicit capacity family schema required; evaluation trace relabel forbidden')
    identity = family['identity']
    need(family.get('family_id') == 'capacity-'+digest(identity), 'capacity family identity checksum differs')
    seeds = identity['seeds']
    need(identity['selection_split'] in ('calibration', 'tuning')
         and isinstance(seeds, list) and len(seeds) == len(set(seeds)) == 3
         and all(type(seed) is int and seed >= 0 and seed != 701 for seed in seeds),
         'family split/seeds cannot be relabelled from evaluation')
    systems = identity['systems']
    need(isinstance(systems, list) and systems and len(set(systems)) == len(systems)
         and all(system in SYSTEMS for system in systems), 'family comparison systems differ')
    CapacityConfig('workload-preparation', identity['slo']['ttft_s'], identity['slo']['tpot_s'],
                   min_requests_per_trial=identity['minimum_requests'])
    for key in ('minimum_scale', 'base_rate_rps', 'duration_s', 'base_duration_s'):
        positive(identity[key], key)
    need(identity['measurement_protocol_version'] == PROTOCOL
         and identity['output_workload'] == 'prepared model-tokenized reference length capped at 512; prompt unchanged',
         'capacity protocol/output workload differs')
    need(all(family.get(key) is False for key in ('evaluation_used_for_selection', 'execution_ready',
         'hardware_executed', 'jobs_enqueued', 'capacity_claimed', 'formal_eligible'))
         and family.get('execution_wiring_pending') is True,
         'preparation cannot claim evaluation selection or executable measurements')
    replayer = load_replayer(identity['generator'])
    corpus = Path(identity['inputs']['corpus_manifest']['path']).parent
    # Check original bound bytes first, even if someone recomputed manifest fields.
    bound_bytes(identity['inputs']['corpus_manifest'])
    bound_bytes(identity['inputs']['corpus_dataset'])
    records, manifest, inputs = corpus_inputs(corpus, identity['dataset'], identity['selection_split'],
                                             loader=replayer.load_split)
    need(inputs == identity['inputs'], 'family corpus/split inputs changed')
    anchor = independent_anchor(identity['anchor']['completion'], corpus=corpus, dataset=identity['dataset'],
        manifest=manifest, corpus_refs=inputs, slo=identity['slo'], loader=replayer.load_split,
        generator=replayer.poisson_trace)
    need(anchor == identity['anchor'], 'family rate anchor changed')
    need(not set(seeds) & {anchor['calibration_seed'], anchor['tuning_seed']},
         'capacity repetition seeds must be independent of the rate-anchor search and confirmation')
    need(identity['model_id'] == manifest['model_name']
         and identity['model_hash'] == anchor['model_hash']
         and identity['tokenizer_hash'] == anchor['tokenizer_hash']
         and identity['base_rate_rps'] == anchor['base_rate_rps']
         and identity['corpus_tokenizer_sha256'] == manifest['tokenizer_sha256']
         and identity['model_config_sha256'] == manifest['model_config_sha256'],
         'family model/tokenizer/rate identity differs from independent inputs')
    duration, sizing = common_duration(records, minimum_rate_rps=identity['minimum_scale']*anchor['base_rate_rps'],
        base_duration_s=identity['base_duration_s'], seeds=seeds, minimum_requests=identity['minimum_requests'],
        dataset=identity['dataset'], generator=replayer.poisson_trace)
    need(identity['duration_s'] == duration and family['duration_selection'] == dict(
        rule='double common base duration until all seeds at declared minimum scale have enough arrivals',
        uses_performance=False, steps=sizing), 'family common duration does not replay its declared minimum scale')
    return family, records, replayer


def make_trace(family, family_ref, records, replayer, *, seed, rate_scale):
    identity = family['identity']
    positive(rate_scale, 'rate scale')
    need(rate_scale >= identity['minimum_scale'], 'rate below declared minimum scale requires a new family; window cannot change')
    need(type(seed) is int and seed in identity['seeds'], 'trace seed must belong to frozen three-seed family')
    rate = identity['base_rate_rps'] * rate_scale
    positive(rate, 'offered rate')
    requests = [asdict(request) for request in replayer.poisson_trace(records, rate, identity['duration_s'], seed, identity['dataset'])]
    need(len(requests) >= identity['minimum_requests'], 'frozen common duration failed minimum arrivals; new family required')
    cohort = [dict(prompt=row['prompt'], max_tokens=row['max_tokens']) for row in requests]
    inputs = identity['inputs']
    return dict(schema=TRACE_SCHEMA, family_id=family['family_id'], family=family_ref,
        capacity_workload_family=family_ref, model_id=identity['model_id'],
        model_hash=identity['model_hash'], tokenizer_hash=identity['tokenizer_hash'],
        dataset=identity['dataset'], selection_split=identity['selection_split'],
        seed=seed, repeat_id='seed-'+str(seed), scale=rate_scale, rate_rps=rate,
        duration_s=identity['duration_s'], slo=identity['slo'], anchor=identity['anchor'],
        corpus_sha256=inputs['corpus_dataset']['sha256'], corpus_manifest_sha256=inputs['corpus_manifest']['sha256'],
        corpus_tokenizer_sha256=identity['corpus_tokenizer_sha256'], selected_split_sha256=inputs['selected_split_sha256'],
        measurement_protocol_version=PROTOCOL, output_workload=identity['output_workload'],
        requests=requests, requests_sha256=digest(requests), cohort_sha256=digest(cohort),
        output_tokens_total=sum(row['max_tokens'] for row in requests),
        input_tokens_total=sum(len(row['prompt']) for row in requests),
        evaluation_used_for_selection=False, execution_ready=False, execution_wiring_pending=True)


def validate_trace(trace_ref):
    """Return {family, trace, family_ref} after full readonly trusted-v1 replay."""
    trace = read_bound(trace_ref)
    need(trace.get('schema') == TRACE_SCHEMA, 'explicit independent capacity trace schema required')
    need(trace.get('selection_split') in ('calibration', 'tuning')
         and trace.get('evaluation_used_for_selection') is False,
         'evaluation trace cannot be relabelled as capacity selection')
    family_ref = trace['capacity_workload_family']
    need(trace.get('family') == family_ref, 'trace family bindings differ')
    family, records, replayer = load_family(family_ref)
    expected = make_trace(family, family_ref, records, replayer, seed=trace['seed'], rate_scale=trace['scale'])
    need(digest(trace) == digest(expected), 'capacity trace does not reproduce its frozen family/cohort bytes and metadata')
    return dict(family=family, trace=trace, family_ref=family_ref)


def prepare_rate(family_ref, out, *, rate_scale):
    """Freeze three reusable trace cohorts; refuse changing a family's window."""
    out = Path(out).resolve()
    need(not out.exists(), 'new immutable capacity rate directory required')
    family, records, replayer = load_family(family_ref)
    identity = family['identity']
    frozen = [make_trace(family, family_ref, records, replayer, seed=seed, rate_scale=rate_scale)
              for seed in identity['seeds']]
    # Validate everything before creating the immutable output tree.
    out.mkdir(parents=True)
    traces, assignments = [], []
    for trace in frozen:
        path = out/('seed-'+str(trace['seed'])+'.json')
        write_new(path, trace)
        ref = binding(path)
        traces.append(dict(trace=ref, seed=trace['seed'], repeat_id=trace['repeat_id'],
            requests=len(trace['requests']), requests_sha256=trace['requests_sha256'], cohort_sha256=trace['cohort_sha256']))
        for system in identity['systems']:
            assignments.append(dict(name=family['family_id'][:25]+'-'+system+'-x'+str(rate_scale)+'-'+trace['repeat_id'],
                model_id=trace['model_id'], system=system, dataset=trace['dataset'], scale=rate_scale,
                seed=trace['seed'], repeat_id=trace['repeat_id'], selection_split=trace['selection_split'],
                duration_s=trace['duration_s'], rate_rps=trace['rate_rps'], slo=trace['slo'], trace=ref, inputs=dict(trace=ref, capacity_workload_family=family_ref),
                measurement_protocol_version=PROTOCOL, family_id=family['family_id'],
                capacity_workload_family=family_ref, output_workload=trace['output_workload'],
                execution_ready=False, execution_wiring_pending=True))
    result = dict(schema='pdblend-capacity-rate-workloads/v1', family=family_ref, capacity_workload_family=family_ref,
        family_id=family['family_id'], scale=rate_scale, rate_rps=frozen[0]['rate_rps'], duration_s=identity['duration_s'],
        selection_split=identity['selection_split'], seeds=identity['seeds'], traces=traces,
        workload_assignments=assignments, execution_ready=False, execution_wiring_pending=True,
        hardware_executed=False, jobs_enqueued=False, formal_eligible=False,
        no_execution_receipts_generated=True)
    write_new(out/'workloads.json', result)
    return dict(manifest=binding(out/'workloads.json'), workloads=result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare-family')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--dataset', choices=tuple(SLOS), required=True)
    p.add_argument('--split', choices=('calibration', 'tuning'), required=True)
    p.add_argument('--anchor', type=Path, required=True)
    p.add_argument('--seed', type=int, action='append', required=True)
    p.add_argument('--minimum-scale', type=float, required=True)
    p.add_argument('--base-duration', type=float, default=150.)
    p.add_argument('--minimum-requests', type=int, default=100)
    p = commands.add_parser('prepare-rate')
    p.add_argument('--family', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rate-scale', type=float, required=True)
    p = commands.add_parser('validate-trace')
    p.add_argument('--trace', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'validate-trace':
            checked = validate_trace(binding(args.trace))
            print(json.dumps(dict(trace=binding(args.trace), family=checked['family_ref'],
                valid=True, execution_wiring_pending=True), sort_keys=True))
            return
        result = (prepare_family(args.out, corpus=args.corpus, dataset=args.dataset, split=args.split,
            anchor=binding(args.anchor), seeds=args.seed, minimum_scale=args.minimum_scale,
            base_duration_s=args.base_duration, minimum_requests=args.minimum_requests)
            if args.command == 'prepare-family' else prepare_rate(binding(args.family), args.out, rate_scale=args.rate_scale))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result['manifest'], sort_keys=True))


if __name__ == '__main__':
    main()

"""Collect model-generated BERT labels from calibration prompts only.

Public reference lengths are never reused as labels. EOS is enabled and the
512-token execution cap is recorded as right-censoring, preserving the paper's
S/M/L class (>=350 for a capped 512-token result).
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import time

from .deployment import SubprocessLifecycle, save, sha
from .predictor import model_identity, output_class, verify_corpus_identity
from .run_v1 import Journal
from .transport import V1Transport
from .validation import MODELS


def prompt_hash(prompt):
    return hashlib.sha256(json.dumps(prompt, separators=(',', ':')).encode()).hexdigest()


def calibration_selection(path, model_path, count, seed):
    value = json.loads(Path(path).read_text())
    manifest = json.loads((Path(path).parent/'manifest.json').read_text())
    identity = model_identity(model_path)
    verify_corpus_identity(manifest, identity, model_path)
    if manifest['datasets'][value['dataset']]['sha256'] != sha(path):
        raise ValueError('calibration source corpus checksum changed')
    rows = value['calibration']
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    # Freeze an ordering so increasing --samples retains previously valid work.
    return [(index, rows[index]) for index in indices[:count]], identity


def actual_label(source, events):
    tokens = []
    terminal = None
    for event in events:
        tokens.extend(event.get('token_ids', []))
        if event.get('finished'):
            terminal = event
    if not tokens or terminal is None or len(tokens) > 512:
        raise ValueError('actual completed target-model token stream required')
    reason = terminal.get('finish_reason')
    if reason not in ('stop', 'length'):
        raise ValueError('model label requires explicit EOS or cap finish reason')
    if reason == 'length' and len(tokens) != 512:
        raise ValueError('unexpected label truncation limit')
    return dict(prompt=list(source['prompt']), input_tokens=len(source['prompt']),
                output_tokens=len(tokens), generated_token_ids=tokens, finish_reason=reason,
                right_censored=reason == 'length', split='calibration',
                prompt_sha256=prompt_hash(source['prompt']),
                label_kind='target_model_completion', ignore_eos=False, sampling_seed=701,
                source_file=source.get('source_file'), source_index=source.get('source_index'))


async def collect_labels(transport, iid, *, model_path, corpus_root, output, per_dataset=32, seed=701):
    if seed != 701 or type(per_dataset) is not int or per_dataset < 5:
        raise ValueError('seed701 and >=5 calibration examples per dataset required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    capability = await transport.json(iid, '/baseline/capability', method='GET')
    identity = model_identity(model_path)
    if (capability.get('model_id') != identity['model']
            or capability.get('engine_revision') != 'vllm-0.10.1.1'):
        raise ValueError('label engine target model/new-stack identity differs')
    datasets = {}
    counts = [0, 0, 0]
    bindings = {}
    for dataset in ('alpaca', 'sharegpt', 'longbench'):
        path = Path(corpus_root)/(dataset+'.json')
        selected, _ = calibration_selection(path, model_path, per_dataset, seed)
        labels = []
        bindings[dataset] = sha(path)
        for index, source in selected:
            key = prompt_hash(source['prompt'])
            artifact = output/'samples'/f'{dataset}-{key}.json'
            if artifact.is_file():
                saved = json.loads(artifact.read_text())
                if (saved.get('model_identity') != identity or saved.get('source_corpus_sha256') != bindings[dataset]
                        or saved.get('capability', {}).get('image_digest') != capability.get('image_digest')
                        or saved.get('capability', {}).get('source_revision') != capability.get('source_revision')
                        or saved.get('label', {}).get('prompt_sha256') != key):
                    raise ValueError('label resume provenance mismatch')
                label = actual_label(source, saved['events'])
                if label != saved['label']:
                    raise ValueError('label differs from saved actual SSE evidence')
            else:
                events = []
                async for event in transport.stream(iid, dict(prompt=source['prompt'], max_tokens=512,
                        request_id=f'dynamo-label-{dataset}-{index}-{time.time_ns()}',
                        seed=701, temperature=0, ignore_eos=False)):
                    events.append(event)
                label = actual_label(source, events)
                save(artifact, dict(model_identity=identity, capability=capability,
                    source_corpus_sha256=bindings[dataset], calibration_index=index,
                    events=events, label=label, generated_at_s=time.time()))
            row = dict(label, label_source_path=str(artifact.resolve()), label_source_sha256=sha(artifact))
            labels.append(row)
            counts[output_class(label['output_tokens'])] += 1
        result = dict(schema=4, dataset=dataset, model_name=identity['model'],
                      label_kind='target_model_completion', calibration=labels,
                      reference_lengths_reused=False, evaluation_trace_used=False)
        target = output/(dataset+'.json')
        save(target, result)
        datasets[dataset] = dict(sha256=sha(target), calibration_examples=len(labels))
    manifest = dict(schema=4, complete=True, model_name=identity['model'], model_identity=identity,
        tokenizer_sha256=identity['tokenizer_sha256'], datasets=datasets,
        source_corpora_sha256=bindings, label_kind='target_model_completion',
        output_work='EOS-enabled actual model completion, capped at 512 and right-censoring recorded',
        seed=701, evaluation_trace_used=False, reference_lengths_reused=False,
        class_counts=dict(zip(('S', 'M', 'L'), counts)), class_coverage_complete=all(count >= 5 for count in counts),
        formal_eligible=False, predictor_qualified=False)
    save(output/'manifest.json', manifest)
    return manifest


async def run(args):
    identity = model_identity(args.model)
    if args.tp not in MODELS.get(identity['model'], ()) or len(args.gpus) != args.tp:
        raise ValueError('one legal model-specific PP1 GPU group required')
    args.out.mkdir(parents=True, exist_ok=True)
    journal = Journal(args.out/('label-events-'+str(time.time_ns())+'.jsonl'))
    spec = dict(id='dynamo-labels', gpus=args.gpus, tp=args.tp, port=args.base_port,
                url=f'http://127.0.0.1:{args.base_port}', generation=0)
    # Label generation changes no GPU clocks and does not claim power metrics.
    transport = V1Transport([spec], lambda *_: (_ for _ in ()).throw(RuntimeError('labeler does not control clocks')), journal)
    config = dict(model_id=identity['model'], model_path=str(args.model), node_gpus=args.gpus,
                  legal_tp=list(MODELS[identity['model']]), base_port=args.base_port)
    lifecycle = SubprocessLifecycle(config, transport, journal, args.out/'engines')
    result = None
    failure = None
    try:
        await transport.start()
        await lifecycle.start(spec)
        result = await collect_labels(transport, 'dynamo-labels', model_path=args.model,
            corpus_root=args.corpus_root, output=args.out, per_dataset=args.samples, seed=701)
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        cleanup = []
        for name, resource in [('lifecycle', lifecycle), ('transport', transport)]:
            try:
                await resource.close()
            except BaseException as exc:
                cleanup.append(dict(component=name, error=repr(exc)))
        journal.close()
        save(args.out/'completion.json', dict(status='passed' if result and not failure and not cleanup else 'failed',
            error=failure, cleanup_errors=cleanup, model_id=identity['model'], seed=701,
            formal_eligible=False, energy_comparable=False, predictor_qualified=False))
    return result


def main(argv=None):
    from .profile_v1 import integers
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--gpus', type=integers, required=True)
    parser.add_argument('--base-port', type=int, default=18000)
    parser.add_argument('--corpus-root', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=32, help='calibration prompts per dataset, resumable ordering')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(asyncio.run(run(args))))


if __name__ == '__main__':
    main()

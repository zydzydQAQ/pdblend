"""Train a model-bound Dynamo predictor for this campaign's fixed output work.

Each model's calibration corpus was independently tokenized against its own
verified tokenizer. Equal token counts across Qwen vocabularies are allowed;
using another model's prepared corpus or trained checkpoint is not.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from .deployment import save, sha
from .predictor import (calibration_examples, encoder_identity, model_identity,
                        output_class, train, verify_corpus_identity)

OUTPUT_WORK = 'model_tokenized_reference_length_cap512_ignore_eos_true'


def prepare(model_path, corpus_root, encoder_path):
    identity = model_identity(model_path)
    root = Path(corpus_root)
    manifest = json.loads((root/'manifest.json').read_text())
    verify_corpus_identity(manifest, identity, model_path)
    if (manifest.get('schema') != 3 or manifest.get('complete') is not True
            or manifest.get('output_work') != 'min(model-tokenized reference length, output cap); reference text omitted'):
        raise ValueError('this campaign requires independently tokenized fixed-work schema3 corpus')
    sources, counts, totals = {}, [0, 0, 0], {}
    for dataset in ('alpaca', 'sharegpt', 'longbench'):
        path = root/(dataset+'.json')
        value = json.loads(path.read_text())
        if (value.get('model_name') != identity['model'] or value.get('dataset') != dataset
                or value.get('output_cap') != 512
                or manifest['datasets'][dataset]['sha256'] != sha(path)):
            raise ValueError('model-specific calibration source binding differs: ' + dataset)
        rows = value.get('calibration', [])
        if not rows:
            raise ValueError('explicit nonempty calibration split required')
        calibration_prompts = set()
        for row in rows:
            prompt, length = row.get('prompt'), row.get('output_tokens')
            if (not isinstance(prompt, list) or len(prompt) != row.get('input_tokens')
                    or type(length) is not int or not 2 <= length <= 512
                    or row.get('split', 'calibration') != 'calibration'):
                raise ValueError('fixed calibration work shape differs')
            digest = hashlib.sha256(json.dumps(prompt, separators=(',', ':')).encode()).hexdigest()
            calibration_prompts.add(digest)
            counts[output_class(length)] += 1
        # Only input hashes from tuning/evaluation are inspected for leakage;
        # their output labels are never consumed by training or selection.
        for split in ('tuning', 'evaluation'):
            if any(hashlib.sha256(json.dumps(row['prompt'], separators=(',', ':')).encode()).hexdigest()
                   in calibration_prompts for row in value.get(split, [])):
                raise ValueError('calibration prompt overlaps another split')
        sources[str(path.resolve())] = sha(path)
        totals[dataset] = len(rows)
    return dict(model_identity=identity, encoder_identity=encoder_identity(encoder_path),
        corpus_manifest_sha256=sha(root/'manifest.json'), source_corpora=sources,
        class_counts=dict(zip(('S', 'M', 'L'), counts)), calibration_examples=totals,
        output_work=OUTPUT_WORK, ignore_eos=True, max_output_tokens=512,
        input_labels='each target model independently tokenized reference, capped at 512',
        evaluation_labels_used=False, seed=701, formal_eligible=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--corpus-root', type=Path, required=True)
    parser.add_argument('--encoder', type=Path, default=Path('/home/models/bert-base-uncased'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    completion = args.report.parent/'completion.json'
    try:
        plan = prepare(args.model, args.corpus_root, args.encoder)
        if args.prepare_only:
            if args.report.exists():
                raise FileExistsError('refusing to overwrite training preparation report')
            save(args.report, dict(plan, hardware_actions_started=False, planned_device=args.device,
                                   output_checkpoint=str(args.out)))
            print(json.dumps(plan))
            return
        if completion.exists():
            raise FileExistsError('refusing to overwrite predictor training completion')
        result = train([args.corpus_root/(name+'.json') for name in ('alpaca','sharegpt','longbench')],
            args.encoder, args.out, model_tokenizer_dir=args.model, epochs=args.epochs,
            batch_size=args.batch_size, seed=701, device=args.device, report_path=args.report)
        path = args.out/'manifest.json'
        manifest = json.loads(path.read_text())
        manifest.update(output_work=OUTPUT_WORK, ignore_eos=True, max_output_tokens=512,
                        campaign_training_plan=plan, predictor_qualified=False)
        save(path, manifest)
        report = json.loads(args.report.read_text())
        report.update(campaign_training_plan=plan, checkpoint_manifest_sha256=sha(path), manifest_sha256=sha(path),
                      predictor_qualified=False, formal_eligible=False)
        save(args.report, report)
        save(completion, dict(status='passed', complete=True, system='dynamollm',
            model_id=plan['model_identity']['model'], seed=701, scope='predictor_training',
            checkpoint=str(args.out), checkpoint_manifest_sha256=sha(path),
            training_report_sha256=sha(args.report), gpu_training=args.device.startswith('cuda'),
            formal_eligible=False, energy_comparable=False, predictor_qualified=False,
            heldout_passed=False, independent_holdout_passed=False))
        print(json.dumps(dict(model=plan['model_identity']['model'], checkpoint=str(args.out),
                              checkpoint_manifest_sha256=sha(path), training_result=result)))
    except BaseException as exc:
        if not args.prepare_only and not completion.exists():
            save(completion, dict(status='failed', complete=False, error=repr(exc),
                scope='predictor_training', formal_eligible=False, energy_comparable=False,
                predictor_qualified=False, heldout_passed=False, independent_holdout_passed=False))
        raise


if __name__ == '__main__':
    main()

"""Resumable own-BERT coverage cache, independent of arrival times or rate anchors."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

from .deployment import save, sha
from .predictor import BertLengthPredictor, model_identity, verify_corpus_identity


def prompt_sha(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def row_identity(row, index):
    prompt = row.get('prompt')
    if (not isinstance(prompt, list) or not prompt or len(prompt) != row.get('input_tokens')
            or any(type(token) is not int or token < 0 for token in prompt)):
        raise ValueError('prepared tokenized request identity invalid')
    return dict(request_index=index, input_tokens=len(prompt), prompt_tokens_sha256=prompt_sha(prompt),
                request_shape_sha256=row['request_shape_sha256'])


def read_prefix(path, rows, binding):
    if not path.exists():
        return []
    result = []
    with path.open() as handle:
        for line in handle:
            value = json.loads(line)
            index = len(result)
            if (index >= len(rows) or value.get('binding_sha256') != binding
                    or any(value.get(k) != v for k,v in row_identity(rows[index], index).items())
                    or type(value.get('predicted_output')) is not int or value['predicted_output'] < 1):
                raise ValueError('cached prediction identity or contiguous prefix differs')
            result.append(value)
    return result


def collect(*, model, predictor, corpus, out, splits=('evaluation',)):
    model, predictor, corpus, out = map(lambda p: Path(p).resolve(), (model,predictor,corpus,out))
    identity = model_identity(model)
    manifest = json.loads((corpus/'manifest.json').read_text())
    verify_corpus_identity(manifest, identity, model)
    if any(split not in ('calibration','tuning','evaluation') for split in splits):
        raise ValueError('explicit corpus split required')
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    engine = BertLengthPredictor(predictor, expected_model=identity['model'],
                                expected_tokenizer_sha256=identity['tokenizer_sha256'])
    binding = dict(schema='dynamo-prediction-cache-v1', model_identity=identity,
        predictor_manifest_sha256=sha(predictor/'manifest.json'), predictor=str(predictor),
        corpus_manifest_sha256=sha(corpus/'manifest.json'), corpus=str(corpus),
        code_sha256={name:sha(Path(__file__).with_name(name))
                     for name in ('prediction_cache.py','predictor.py')},
        splits=list(splits), prediction_inputs='visible prompt text decoded from this model token IDs',
        evaluation_outputs_used_for_prediction=False, future_arrivals_used=False,
        predictor_fitted_here=False, rate_anchor_status='missing_rate_anchor',
        formal_eligible=False, prediction_accuracy_qualified=False)
    marker = out/'binding.json'
    if marker.exists():
        if json.loads(marker.read_text()) != binding:
            raise ValueError('prediction cache frozen binding changed')
    else:
        if any(out.iterdir()):
            raise ValueError('unbound existing cache artifacts')
        save(marker, binding)
    binding_sha = sha(marker)
    started = time.time()
    artifacts, counts, output_shapes = {}, {}, {}
    for dataset in ('alpaca','sharegpt','longbench'):
        source = corpus/(dataset+'.json')
        expected = manifest['dataset_sha256'][dataset]
        if sha(source) != expected:
            raise ValueError('prepared dataset SHA differs: '+dataset)
        value = json.loads(source.read_text())
        if value.get('model_name') != identity['model']:
            raise ValueError('prepared dataset is for another model')
        for split in splits:
            rows = value[split]
            path = out/(dataset+'-'+split+'.jsonl')
            prefix = read_prefix(path, rows, binding_sha)
            with path.open('a') as handle:
                for index in range(len(prefix), len(rows)):
                    row = rows[index]
                    text = tokenizer.decode(row['prompt'], skip_special_tokens=False)
                    predicted = engine.predict_text(text)
                    record = dict(row_identity(row,index), binding_sha256=binding_sha,
                        dataset=dataset, split=split, corpus_file_sha256=expected,
                        visible_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        predicted_output=predicted, latency_s=engine.last_latency_s,
                        actual_output_limit=row['output_tokens'], output_limit_used_for_prediction=False)
                    handle.write(json.dumps(record, separators=(',', ':'))+'\n'); handle.flush()
                    prefix.append(record)
            key=dataset+'-'+split
            counts[key]=len(prefix)
            output_shapes[key]=sorted({r['predicted_output'] for r in prefix})
            artifacts[str(path)]=sha(path)
            save(out/'progress.json', dict(model_id=identity['model'], counts=counts,
                 output_shapes=output_shapes, started_s=started, updated_s=time.time()))
    completion = dict(schema='dynamo-prediction-cache-completion-v1', complete=True,
        model_id=identity['model'], binding_sha256=binding_sha, binding_path=str(marker),
        counts=counts, output_shapes=output_shapes, artifacts=artifacts,
        arrival_times_bound=False, rate_anchor_status='missing_rate_anchor',
        predictor_fitted_here=False, formal_eligible=False, started_s=started, finished_s=time.time())
    save(out/'completion.json', completion)
    return completion


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','predictor','corpus','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['evaluation'])
    args=parser.parse_args(argv)
    print(json.dumps(collect(**vars(args))))


if __name__=='__main__':
    main()

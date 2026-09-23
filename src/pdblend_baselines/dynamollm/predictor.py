"""Locally trained DynamoLLM paper-architecture output-length classifier.

The authors' checkpoint is not bundled. Training consumes calibration prompts
only; inference consumes visible prompt text, never max_tokens or output labels.
Torch/Transformers are imported lazily so ordinary CPU contracts stay lightweight.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time


ARCHITECTURE='bert-base-cls-2fc-3class'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def model_identity(directory):
    """Match the corpus tokenizer identity, including model config and revision."""
    if directory is None:raise ValueError('model tokenizer directory required to bind predictor identity')
    directory=Path(directory)
    def value_digest(value):
        raw=(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)+"\n").encode()
        return hashlib.sha256(raw).hexdigest()
    manifest_path=directory/'pdblend-model-manifest.json'
    manifest=json.loads(manifest_path.read_text())
    files={name:digest(directory/name) for name in ('tokenizer.json','tokenizer_config.json','config.json')}
    upstream={row['path']:row['sha256'] for row in manifest['files']}
    if any(upstream.get(name)!=sha for name,sha in files.items()):
        raise ValueError('model tokenizer differs from verified model manifest')
    return dict(model=manifest['repo_id'].split('/')[-1],repo_id=manifest['repo_id'],
        revision=manifest['revision'],manifest_sha256=digest(manifest_path),
        tokenizer_files=files,tokenizer_sha256=value_digest(files))


def verify_corpus_identity(manifest, identity, model_directory):
    """Bridge old and schema-3 corpus identities using verified file hashes.

    The old digest includes config.json and a trailing newline. Schema 3 hashes
    the tokenizer inventory separately. Equal-looking aggregate digests are
    neither required nor substituted: every declared tokenizer file is checked.
    """
    if manifest.get('model_name') != identity['model']:
        raise ValueError('calibration corpus model differs from predictor identity')
    if manifest.get('schema') != 3:
        if manifest.get('tokenizer_sha256') != identity['tokenizer_sha256']:
            raise ValueError('calibration corpus tokenizer differs from predictor identity')
        return 'legacy-model-tokenizer-v1'
    directory = Path(model_directory).resolve()
    files = manifest.get('tokenizer_files_sha256')
    if not isinstance(files, dict) or not {'tokenizer.json', 'tokenizer_config.json'} <= set(files):
        raise ValueError('schema-3 tokenizer inventory is incomplete')
    raw = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(raw).hexdigest() != manifest.get('tokenizer_sha256'):
        raise ValueError('schema-3 tokenizer inventory digest differs')
    for name, expected in files.items():
        source = (directory / name).resolve()
        if not source.is_relative_to(directory) or digest(source) != expected:
            raise ValueError('schema-3 tokenizer file differs: ' + name)
    if manifest.get('model_config_sha256') != identity['tokenizer_files']['config.json']:
        raise ValueError('schema-3 target model configuration differs')
    return 'corpus-schema3-file-verified-to-model-tokenizer-v1'


def encoder_identity(directory):
    directory=Path(directory);path=directory/'pdblend-encoder-manifest.json'
    manifest=json.loads(path.read_text())
    if manifest.get('repo_id')!='google-bert/bert-base-uncased' or manifest.get('verified') is not True:
        raise ValueError('verified official BERT encoder manifest required')
    for row in manifest['files']:
        source=(directory/row['path']).resolve()
        if not source.is_relative_to(directory.resolve()) or digest(source)!=row['sha256']:
            raise ValueError('BERT encoder source hash mismatch')
    return dict(repo_id=manifest['repo_id'],revision=manifest['revision'],manifest_sha256=digest(path),
                source_manifest_sha256=manifest['source_manifest_sha256'])


def output_class(tokens):
    if type(tokens) is not int or tokens<1:
        raise ValueError('positive integer output length required')
    return 0 if tokens<100 else 1 if tokens<350 else 2


def calibration_examples(corpus, model_tokenizer=None):
    if not isinstance(corpus,dict) or not corpus.get('calibration'):
        raise ValueError('an explicit nonempty calibration split is required')
    examples=[]
    for row in corpus['calibration']:
        if row.get('split',row.get('fixture_split','calibration'))!='calibration':
            raise ValueError('non-calibration row in predictor training')
        text=row.get('text',row.get('prompt_text'))
        if text is None:
            prompt=row.get('prompt')
            if isinstance(prompt,str):text=prompt
            elif isinstance(prompt,list) and model_tokenizer is not None:
                text=model_tokenizer.decode(prompt,skip_special_tokens=False)
        if not isinstance(text,str) or not text.strip():
            raise ValueError('calibration prompt text or its local Qwen tokenizer is required')
        length=row.get('output_tokens')
        output_class(length)
        examples.append((text,length))
    return examples


def split_examples(examples, seed=1701):
    """Keep identical visible prompts in one split, irrespective of labels."""
    groups=defaultdict(list)
    for text,length in examples:
        groups[hashlib.sha256(text.encode()).hexdigest()].append((text,length))
    keys=sorted(groups)
    if len(keys)<5:
        raise ValueError('at least five distinct calibration prompts required for held-out training')
    random.Random(seed).shuffle(keys)
    cut=max(1,min(len(keys)-1,int(.8*len(keys))))
    return ([row for key in keys[:cut] for row in groups[key]],
            [row for key in keys[cut:] for row in groups[key]])


def _network(encoder, hidden_size=256):
    import torch
    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=encoder
            self.hidden=torch.nn.Linear(encoder.config.hidden_size,hidden_size)
            self.output=torch.nn.Linear(hidden_size,3)

        def forward(self, **inputs):
            cls=self.encoder(**inputs).last_hidden_state[:,0]
            return self.output(torch.relu(self.hidden(cls)))
    return Classifier()


def _encode(tokenizer,texts,max_tokens,device):
    # Preserve both the task instruction and a long prompt's final question.
    prepared=[]
    budget=max_tokens-2
    for text in texts:
        ids=tokenizer.encode(text,add_special_tokens=False)
        if len(ids)>budget:
            left=budget//2
            ids=ids[:left]+ids[-(budget-left):]
        ids=tokenizer.build_inputs_with_special_tokens(ids)
        prepared.append(dict(input_ids=ids,attention_mask=[1]*len(ids)))
    return {key:value.to(device) for key,value in tokenizer.pad(prepared,padding=True,
                                                               return_tensors='pt').items()}


def _metrics(labels,predictions):
    confusion=[[0]*3 for _ in range(3)]
    for actual,predicted in zip(labels,predictions):confusion[actual][predicted]+=1
    f1=[]
    for c in range(3):
        tp=confusion[c][c];fp=sum(row[c] for row in confusion)-tp;fn=sum(confusion[c])-tp
        f1.append(2*tp/max(1,2*tp+fp+fn))
    return dict(confusion=confusion,accuracy=sum(a==p for a,p in zip(labels,predictions))/len(labels),
                macro_f1=sum(f1)/3,underprediction_rate=sum(p<a for a,p in zip(labels,predictions))/len(labels))


def train(corpora, encoder_dir, output_dir, *, model_tokenizer_dir=None,
          epochs=3,batch_size=16,learning_rate=2e-5,seed=1701,device='cpu',report_path=None):
    """Train from local assets, save best held-out epoch and a hash-bound manifest."""
    model_source=model_identity(model_tokenizer_dir)
    encoder_source=encoder_identity(encoder_dir)
    import torch
    from transformers import AutoTokenizer,BertModel
    if (type(epochs) is not int or epochs<1 or type(batch_size) is not int or batch_size<1
            or not math.isfinite(learning_rate) or learning_rate<=0):
        raise ValueError('positive training settings required')
    output_dir=Path(output_dir)
    if output_dir.exists():raise FileExistsError('refusing to overwrite predictor checkpoint')
    if report_path is not None and Path(report_path).exists():raise FileExistsError('refusing to overwrite predictor report')
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(str(encoder_dir),local_files_only=True)
    qwen=(AutoTokenizer.from_pretrained(str(model_tokenizer_dir),local_files_only=True)
          if model_tokenizer_dir is not None else None)
    examples=[];sources={}
    for path in corpora:
        path=Path(path)
        corpus=json.loads(path.read_text())
        corpus_manifest_path=path.parent/'manifest.json'
        if not corpus_manifest_path.is_file():
            raise ValueError('corpus model/tokenizer manifest required')
        corpus_manifest=json.loads(corpus_manifest_path.read_text())
        verify_corpus_identity(corpus_manifest, model_source, model_tokenizer_dir)
        expected=corpus_manifest.get('datasets',{}).get(corpus.get('dataset'),{}).get('sha256')
        if expected!=digest(path):raise ValueError('calibration corpus differs from frozen manifest')
        examples.extend(calibration_examples(corpus,qwen))
        sources[str(path.resolve())]=digest(path)
    training,validation=split_examples(examples,seed)
    encoder=BertModel.from_pretrained(str(encoder_dir),local_files_only=True)
    if encoder.config.hidden_size!=768 or encoder.config.num_hidden_layers!=12:
        raise ValueError('the paper-architecture predictor requires BERT-base')
    model=_network(encoder).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    criterion=torch.nn.CrossEntropyLoss()
    best=None;best_state=None;history=[];started=time.monotonic()
    rng=random.Random(seed)
    for epoch in range(epochs):
        model.train();order=list(training);rng.shuffle(order);loss_sum=0.
        for index in range(0,len(order),batch_size):
            batch=order[index:index+batch_size]
            inputs=_encode(tokenizer,[r[0] for r in batch],512,device)
            labels=torch.tensor([output_class(r[1]) for r in batch],device=device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(**inputs),labels);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();loss_sum+=float(loss.detach())*len(batch)
        model.eval();predictions=[];labels=[]
        with torch.inference_mode():
            for index in range(0,len(validation),batch_size):
                batch=validation[index:index+batch_size]
                predictions.extend(model(**_encode(tokenizer,[r[0] for r in batch],512,device)).argmax(-1).cpu().tolist())
                labels.extend(output_class(r[1]) for r in batch)
        metrics=_metrics(labels,predictions)
        history.append(dict(epoch=epoch+1,training_loss=loss_sum/len(training),**metrics))
        score=(metrics['macro_f1'],-metrics['underprediction_rate'])
        if best is None or score>best:
            best=score;best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    output_dir.mkdir(parents=True,exist_ok=False)
    tokenizer.save_pretrained(output_dir/'tokenizer')
    encoder.config.to_json_file(output_dir/'encoder_config.json')
    torch.save(best_state,output_dir/'classifier.pt')
    representatives=[]
    defaults=(99,349,512)
    for c in range(3):
        lengths=sorted(length for _,length in training if output_class(length)==c)
        representatives.append(lengths[min(len(lengths)-1,math.ceil(.9*len(lengths))-1)] if lengths else defaults[c])
    manifest=dict(schema=1,architecture=ARCHITECTURE,origin='locally trained paper architecture; not authors checkpoint',
        split='calibration',source_corpora=sources,encoder_dir=str(Path(encoder_dir).resolve()),
        model_identity=model_source,encoder_identity=encoder_source,
        encoder_config_sha256=digest(output_dir/'encoder_config.json'),hidden_size=256,max_tokens=512,
        text_truncation='head-tail',classes=['S','M','L'],output_cuts=[99,349],representatives=representatives,
        seed=seed,training_examples=len(training),validation_examples=len(validation),
        prompt_group_split=True,evaluation_trace_used=False,device=device,
        training_settings=dict(epochs=epochs,batch_size=batch_size,learning_rate=learning_rate),
        epoch_metrics=history,best_macro_f1=best[0],training_elapsed_s=time.monotonic()-started,
        files={str(p.relative_to(output_dir)):digest(p) for p in output_dir.rglob('*') if p.is_file()})
    (output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    if report_path is not None:
        report_path=Path(report_path);report_path.parent.mkdir(parents=True,exist_ok=True)
        report_path.write_text(json.dumps(dict(manifest,checkpoint_dir=str(output_dir.resolve()),
                            manifest_sha256=digest(output_dir/'manifest.json')),indent=2,allow_nan=False))
    return manifest


def verify_checkpoint(directory, *, expected_model=None, expected_tokenizer_sha256=None):
    directory=Path(directory).resolve()
    if not (directory/'manifest.json').is_file():
        raise ValueError('Dynamo BERT checkpoint missing: train and verify calibration-only weights before measurement')
    manifest=json.loads((directory/'manifest.json').read_text())
    if (manifest.get('schema')!=1 or manifest.get('architecture')!=ARCHITECTURE
            or manifest.get('split')!='calibration' or manifest.get('evaluation_trace_used') is not False
            or manifest.get('output_cuts')!=[99,349] or not manifest.get('source_corpora')
            or not manifest.get('files') or 'classifier.pt' not in manifest['files']
            or 'encoder_config.json' not in manifest['files']):
        raise ValueError('invalid calibration-only BERT checkpoint manifest')
    model=manifest.get('model_identity',{});encoder=manifest.get('encoder_identity',{})
    if (not model.get('model') or not re.fullmatch('[0-9a-f]{64}',str(model.get('tokenizer_sha256','')))
            or encoder.get('repo_id')!='google-bert/bert-base-uncased'
            or not re.fullmatch('[0-9a-f]{40}',str(encoder.get('revision','')))
            or not re.fullmatch('[0-9a-f]{64}',str(encoder.get('manifest_sha256','')))):
        raise ValueError('missing BERT encoder or output model/tokenizer identity')
    if expected_model is not None and model['model']!=expected_model:
        raise ValueError('Dynamo checkpoint output model mismatch')
    if expected_tokenizer_sha256 is not None and model['tokenizer_sha256']!=expected_tokenizer_sha256:
        raise ValueError('Dynamo checkpoint output tokenizer mismatch')
    for relative,expected in manifest['files'].items():
        path=(directory/relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or digest(path)!=expected:
            raise ValueError('BERT checkpoint hash mismatch: '+relative)
    reps=manifest.get('representatives',[])
    if len(reps)!=3 or [output_class(n) for n in reps]!=[0,1,2]:
        raise ValueError('BERT class representatives change the paper boundaries')
    return manifest


class BertLengthPredictor:
    def __init__(self, checkpoint_dir, *, device='cpu',expected_model=None,expected_tokenizer_sha256=None):
        self.directory=Path(checkpoint_dir)
        self.manifest=verify_checkpoint(self.directory,expected_model=expected_model,
                                        expected_tokenizer_sha256=expected_tokenizer_sha256)
        import torch
        from transformers import AutoTokenizer,BertConfig,BertModel
        self.torch=torch;self.device=device
        if device=='cpu':torch.set_num_threads(4)
        self.tokenizer=AutoTokenizer.from_pretrained(str(self.directory/'tokenizer'),local_files_only=True)
        encoder=BertModel(BertConfig.from_json_file(self.directory/'encoder_config.json'))
        self.model=_network(encoder,self.manifest['hidden_size'])
        self.model.load_state_dict(torch.load(self.directory/'classifier.pt',map_location='cpu',weights_only=True))
        self.model.to(device).eval()
        self.last_latency_s=0.

    def predict_text(self, text):
        if not isinstance(text,str) or not text.strip():raise ValueError('visible prompt text required')
        started=time.perf_counter()
        with self.torch.inference_mode():
            inputs=_encode(self.tokenizer,[text],self.manifest['max_tokens'],self.device)
            prediction=int(self.model(**inputs).argmax(-1).item())
        self.last_latency_s=time.perf_counter()-started
        return self.manifest['representatives'][prediction]


def benchmark(checkpoint_dir, corpora, model_tokenizer_dir, report_path, *, samples=32,warmup=3):
    """Measure actual CPU prediction cost on calibration prompts, outside GPU runs."""
    if type(samples) is not int or samples<1 or type(warmup) is not int or warmup<0:
        raise ValueError('positive sample count and nonnegative warmup required')
    report_path=Path(report_path)
    if report_path.exists():raise FileExistsError('refusing to overwrite prediction benchmark')
    identity=model_identity(model_tokenizer_dir)
    predictor=BertLengthPredictor(checkpoint_dir,device='cpu',expected_model=identity['model'],
                                 expected_tokenizer_sha256=identity['tokenizer_sha256'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(model_tokenizer_dir),local_files_only=True)
    chosen=[];sources={}
    for path in corpora:
        path=Path(path);source=json.loads(path.read_text());sources[str(path.resolve())]=digest(path)
        rows=calibration_examples(source,tokenizer)
        # Equal deterministic sample budget per dataset, without future outputs.
        chosen.extend((source.get('dataset',path.stem),text) for text,_ in rows[:samples])
    for index in range(warmup):predictor.predict_text(chosen[index%len(chosen)][1])
    records=[];started=time.perf_counter();cpu_started=time.process_time()
    for dataset,text in chosen:
        predicted=predictor.predict_text(text)
        records.append(dict(dataset=dataset,prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                            latency_s=predictor.last_latency_s,predicted_output=predicted))
    values=sorted(row['latency_s'] for row in records)
    quantile=lambda q:values[min(len(values)-1,math.ceil(q*len(values))-1)]
    report=dict(schema=1,measurement='hardware-cpu',kind='Dynamo BERT inference',device='cpu',threads=4,
        checkpoint_manifest_sha256=digest(Path(checkpoint_dir)/'manifest.json'),
        model_identity=identity,source_corpora=sources,split='calibration',warmup=warmup,
        samples=len(records),p50_s=quantile(.5),p95_s=quantile(.95),p99_s=quantile(.99),
        cpu_time_s=time.process_time()-cpu_started,elapsed_s=time.perf_counter()-started,records=records,
        accounting='model/tokenizer loading and warmup excluded; tokenization and forward pass included; serving also records thread wait in TTFT')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def benchmark_main(argv=None):
    parser=argparse.ArgumentParser(description='Measure trained Dynamo predictor CPU latency on calibration prompts')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,nargs='+',required=True)
    parser.add_argument('--model-tokenizer',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=32)
    parser.add_argument('--warmup',type=int,default=3)
    args=parser.parse_args(argv)
    result=benchmark(args.checkpoint,args.corpus,args.model_tokenizer,args.report,samples=args.samples,warmup=args.warmup)
    print(json.dumps({key:result[key] for key in ('samples','p50_s','p95_s','p99_s','cpu_time_s')}))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,nargs='+',required=True)
    parser.add_argument('--encoder',type=Path,required=True)
    parser.add_argument('--model-tokenizer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--seed',type=int,default=1701)
    args=parser.parse_args(argv)
    report=train(args.corpus,args.encoder,args.output,model_tokenizer_dir=args.model_tokenizer,
                 epochs=args.epochs,batch_size=args.batch_size,seed=args.seed,device=args.device,report_path=args.report)
    print(json.dumps(dict(checkpoint=str(args.output),best_macro_f1=report['best_macro_f1'])))


if __name__=='__main__':main()

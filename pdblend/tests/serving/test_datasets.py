import json
import pytest
from ecopadg.serving.datasets import raw_examples,encode_workload,make_trace,make_dynamic_trace


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        return list(range(20))
    def encode(self,text,**kwargs):
        return list(range(len(text)))


def test_longbench_keeps_task_question_and_avoids_duplicate_e_subset(tmp_path):
    root=tmp_path/'longbench';root.mkdir()
    sample=dict(context='evidence',input='actual question?',answers=['answer'])
    for name in ('qasper','qasper_e'):
        (root/(name+'.jsonl')).write_text(json.dumps(sample)+'\n')
    rows=list(raw_examples('longbench',tmp_path,{'qasper':'Read {context}; answer {input}'}))
    assert len(rows)==1
    assert rows[0][2]==[dict(role='user',content='Read evidence; answer actual question?')]


def test_context_budget_preserves_both_prompt_ends_and_fixed_output_work():
    row=encode_workload(Tokenizer(),[],reference='answer',max_input=8,max_output=3)
    assert row['prompt']==[0,1,2,3,16,17,18,19]
    assert row['output_tokens']==3 and row['input_tokens']==8
    assert row['original_input_tokens']==20 and row['truncated']
    assert not any(k in row for k in ('reference','answer'))
    assert encode_workload(Tokenizer(),[],' ') is None
    with pytest.raises(ValueError): encode_workload(Tokenizer(),[],'answer',0)


def test_trace_pairing_is_seeded_and_never_exposes_reference():
    rows=[dict(prompt=[i],input_tokens=1,output_tokens=i+1,request_shape_sha256=str(i),
               answer='private reference') for i in range(5)]
    args=dict(dataset='test',split='development',load='low')
    a=make_trace(rows,.3,11,**args)
    assert a==make_trace(rows,.3,11,**args)
    assert a['requests']!=make_trace(rows,.3,22,**args)['requests']
    assert a['requests'][0]['arrival_s']==0
    assert [q['output_len'] for q in a['requests']]==[1,2,3,4,5]
    assert 'private reference' not in json.dumps(a)


def test_dynamic_trace_covers_true_slow_cycles_with_shared_fixed_work():
    names=('alpaca','sharegpt','longbench')
    corpora={d:[dict(prompt=[i+1],input_tokens=1,output_tokens=2,request_shape_sha256=d)]
             for i,d in enumerate(names)}
    phases=[dict(start_s=0,end_s=1800,length_mix=[1,0,0],capacity_fraction=.3),
            dict(start_s=1800,end_s=3600,length_mix=[0,0,1],capacity_fraction=.6)]
    trace=make_dynamic_trace(corpora,dict.fromkeys(names,.1),phases,101)
    assert trace==make_dynamic_trace(corpora,dict.fromkeys(names,.1),phases,101)
    assert trace['requests'][0]['arrival_s']==0 and trace['requests'][-1]['arrival_s']==3600
    assert trace['phases'][0]['rate']==pytest.approx(.03)
    assert trace['phases'][1]['rate']==pytest.approx(.06)
    assert set(d for d,q in zip(trace['source_datasets'],trace['requests']) if q['arrival_s']<1800)=={'alpaca'}
    assert set(d for d,q in zip(trace['source_datasets'],trace['requests']) if q['arrival_s']>=1800)=={'longbench'}

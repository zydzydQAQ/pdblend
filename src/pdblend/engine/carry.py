"""Strict first-token carry protocol for the controlled OpenAI benchmark path.

Token IDs come from vLLM logprobs, never from retokenizing output text. Full
decode IDs are an optional functional diagnostic with additional engine work.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass


class CarryProtocolError(ValueError):
    pass


# The three registered Qwen2.5 tokenizer_config.json files mark these IDs as
# special. Other model families must supply their own verified special IDs.
QWEN25_SPECIAL_TOKEN_IDS = tuple(range(151643,151657))


def validate_request(body):
    prompt=body.get('prompt')
    if not isinstance(prompt,(list,tuple)) or not prompt or any(type(t) is not int or t<0 for t in prompt):
        raise CarryProtocolError('carry_first_token requires a nonempty flat token-ID prompt; string retokenization is unsupported')
    if type(body.get('max_tokens')) is not int or body['max_tokens']<1:
        raise CarryProtocolError('max_tokens must be a positive integer')
    if body.get('ignore_eos') is not True:
        raise CarryProtocolError('carry_first_token currently requires ignore_eos=true')
    if body.get('temperature',0)!=0:
        raise CarryProtocolError('carry_first_token currently supports greedy sampling only')
    for key,default in [('n',1),('best_of',1),('echo',False),('use_beam_search',False),
                        ('frequency_penalty',0),('presence_penalty',0),('repetition_penalty',1),('min_tokens',0)]:
        if body.get(key,default) not in (None,default):
            raise CarryProtocolError(f'unsupported carry sampling option: {key}')
    for key in ('stop','stop_token_ids','logits_processors','allowed_token_ids','logit_bias','guided_json',
                'guided_regex','guided_choice','guided_grammar','structured_outputs','response_format','suffix'):
        if body.get(key):raise CarryProtocolError(f'unsupported carry sampling option: {key}')
    if body.get('logprobs') is not None or body.get('prompt_logprobs') is not None or body.get('return_tokens_as_token_ids'):
        raise CarryProtocolError('caller-supplied logprobs are unsupported in carry mode; use explicit token diagnostics')


def diagnostic_body(body):
    return dict(body,logprobs=0,return_tokens_as_token_ids=True)


def token_ids(choice):
    logprobs=choice.get('logprobs')
    if not isinstance(logprobs,dict) or not isinstance(logprobs.get('tokens'),list):
        raise CarryProtocolError('missing authoritative logprobs token IDs')
    ids=[]
    for token in logprobs['tokens']:
        if not isinstance(token,str) or re.fullmatch(r'token_id:(0|[1-9][0-9]*)',token) is None:
            raise CarryProtocolError('noncanonical token ID; text retokenization is forbidden')
        ids.append(int(token.split(':',1)[1]))
    return ids


@dataclass(frozen=True)
class FirstToken:
    token_id:int
    text:str
    choice:dict
    received_s:float


def extract_first(data, *, prompt_tokens, received_s, allow_unsafe_text=False,
                  special_token_ids=QWEN25_SPECIAL_TOKEN_IDS):
    choices=data.get('choices')
    if not isinstance(choices,list) or len(choices)!=1 or choices[0].get('index',0)!=0:
        raise CarryProtocolError('prefill must return exactly one choice')
    choice=choices[0];ids=token_ids(choice);usage=data.get('usage') or {}
    if len(ids)!=1 or usage.get('completion_tokens')!=1 or usage.get('prompt_tokens')!=prompt_tokens:
        raise CarryProtocolError('prefill token IDs and usage must describe exactly one generated token')
    if usage.get('total_tokens',prompt_tokens+1)!=prompt_tokens+1:
        raise CarryProtocolError('prefill total usage is inconsistent')
    if choice.get('finish_reason')!='length':
        raise CarryProtocolError('unexpected prefill EOS/stop; carry requires a length-limited first token')
    text=choice.get('text')
    if not isinstance(text,str):
        raise CarryProtocolError('prefill text must be a string')
    if ('\ufffd' in text or (not text and ids[0] not in special_token_ids)) and not allow_unsafe_text:
        raise CarryProtocolError('first-token text has an unsupported empty/partial-Unicode boundary; no silent text reconstruction')
    return FirstToken(ids[0],text,copy.deepcopy(choice),received_s)


def decode_body(body, first, *, diagnostics=False):
    validate_request(body)
    if body['max_tokens']==1:
        raise CarryProtocolError('one-token requests must bypass remote KV before prefill')
    result=dict(body,prompt=[*body['prompt'],first.token_id],max_tokens=body['max_tokens']-1)
    return diagnostic_body(result) if diagnostics else result


def combined_usage(usage, *, original_prompt_tokens, max_tokens):
    if not isinstance(usage,dict) or usage.get('prompt_tokens')!=original_prompt_tokens+1:
        raise CarryProtocolError('decode usage does not include the carried prompt token')
    n=usage.get('completion_tokens')
    if type(n) is not int or not 0<=n<=max_tokens-1:
        raise CarryProtocolError('decode output usage exceeds the remaining token budget')
    if usage.get('total_tokens',original_prompt_tokens+1+n)!=original_prompt_tokens+1+n:
        raise CarryProtocolError('decode total usage is inconsistent')
    return dict(prompt_tokens=original_prompt_tokens,completion_tokens=n+1,total_tokens=original_prompt_tokens+n+1)


def first_event(first, *, request_id, model, created, diagnostics=False):
    choice=copy.deepcopy(first.choice)
    choice.update(index=0,finish_reason=None)
    choice.pop('stop_reason',None)
    if not diagnostics:choice['logprobs']=None
    event=dict(id=request_id,object='text_completion',created=created,model=model,choices=[choice])
    if not first.text:
        # A skipped special token is still a generated token and its arrival
        # determines TTFT. Nonempty text keeps the ordinary SSE counting path.
        event['pdblend_generated_tokens']=1
    return event


def encode_event(event):
    return b'data: '+json.dumps(event,separators=(',',':'),ensure_ascii=False).encode()+b'\n\n'


class SSEEvents:
    """Parse complete JSON data frames across arbitrary network boundaries."""
    def __init__(self):self.buffer=b'';self.done=False
    def feed(self,chunk):
        self.buffer+=chunk;events=[]
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1);line=line.rstrip(b'\r')
            if not line.startswith(b'data:'):continue
            payload=line[5:].strip()
            if payload==b'[DONE]':self.done=True;break
            if self.done:raise CarryProtocolError('data after DONE')
            try:event=json.loads(payload)
            except (ValueError,UnicodeDecodeError) as exc:raise CarryProtocolError('invalid upstream SSE JSON') from exc
            if not isinstance(event,dict):raise CarryProtocolError('upstream SSE event must be an object')
            events.append(event)
        return events


def golden(reference, repeat, combined):
    """No text matching or partial-prefix acceptance can replace exact IDs."""
    for row in (reference,repeat,combined):
        if not row or any(type(t) is not int or t<0 for t in row):
            raise CarryProtocolError('golden requires nonempty authoritative token-ID sequences')
    mismatch=next((i for i,(a,b) in enumerate(zip(reference,combined)) if a!=b),None)
    if mismatch is None and len(reference)!=len(combined):mismatch=min(len(reference),len(combined))
    return dict(reference_stable=list(reference)==list(repeat),tokens_match=list(reference)==list(combined),
        first_mismatch_index=mismatch,reference_token_id=reference[mismatch] if mismatch is not None and mismatch<len(reference) else None,
        pd_token_id=combined[mismatch] if mismatch is not None and mismatch<len(combined) else None,
        passed=list(reference)==list(repeat)==list(combined),energy_comparable=False)

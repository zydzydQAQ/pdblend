"""Lossless JSON encoding of explicitly unavailable development predictions."""
from copy import deepcopy
from dataclasses import asdict
import math


def encode_plan(plan):
    """Keep unknown positive-infinite estimates unknown, never invent a number."""
    missing = []
    def visit(value, path):
        if isinstance(value, dict):
            return {key:visit(item, path+'/'+str(key).replace('~','~0').replace('/','~1'))
                    for key,item in value.items()}
        if isinstance(value, (list, tuple)):
            return [visit(item, path+'/'+str(i)) for i,item in enumerate(value)]
        if type(value) is float and not math.isfinite(value):
            if value != float('inf') or not (path in ('/power_w','/ttft_s','/tpot_s') or path.startswith('/detail/')):
                raise ValueError('only unavailable positive-infinite plan estimates can be encoded')
            missing.append(dict(path=path, kind='positive_infinity'))
            return None
        return value
    values = visit(asdict(plan), '')
    return dict(plan=values, unavailable_estimates=sorted(missing, key=lambda row:row['path']))


def decode_plan(choice, *, observation=False):
    """Formal callers never accept unavailable estimate annotations."""
    from pdblend.planner.pool import Plan
    values = deepcopy(choice.get('plan'))
    if not isinstance(values, dict):
        raise ValueError('offline choice requires a plan object')
    unavailable = choice.get('unavailable_estimates', [])
    if not isinstance(unavailable, list) or (unavailable and not observation):
        raise ValueError('unavailable plan estimates require explicit development observation scope')
    seen = set()
    for entry in unavailable:
        if not isinstance(entry, dict) or set(entry) != {'path','kind'}:
            raise ValueError('invalid unavailable plan estimate annotation')
        path = entry['path']
        if (not isinstance(path,str) or path in seen or entry['kind'] != 'positive_infinity'
                or not (path in ('/power_w','/ttft_s','/tpot_s') or path.startswith('/detail/'))):
            raise ValueError('invalid unavailable plan estimate path')
        parts = [part.replace('~1','/').replace('~0','~') for part in path[1:].split('/')]
        node = values
        try:
            for part in parts[:-1]:
                node = node[int(part)] if isinstance(node,list) else node[part]
            key = int(parts[-1]) if isinstance(node,list) else parts[-1]
            if node[key] is not None:
                raise ValueError('annotated estimate is not an explicit null')
            node[key] = float('inf')
        except (KeyError,IndexError,TypeError) as exc:
            raise ValueError('unavailable plan estimate path is absent') from exc
        seen.add(path)
    if any(type(values.get(key)) not in (int,float) or values[key] < 0
           or math.isnan(values[key]) for key in ('power_w','ttft_s','tpot_s')):
        raise ValueError('invalid offline Plan estimate or missing explicit unavailable annotation')
    try:
        result = Plan(**values)
    except TypeError as exc:
        raise ValueError('invalid offline plan fields') from exc
    # Re-encoding proves that every nonfinite field was explicitly annotated
    # and rejects NaN/-infinity hidden anywhere in the plan detail.
    encoded = encode_plan(result)
    # Historical development choices used Python's Infinity JSON extension.
    # Reading those original bytes remains supported only in the pre-existing
    # explicit observation scope; all newly written choices use strict JSON.
    if observation and 'unavailable_estimates' not in choice:
        return result
    if encoded['plan'] != choice['plan'] or encoded['unavailable_estimates'] != sorted(unavailable, key=lambda row:row['path']):
        raise ValueError('offline plan unavailable annotations do not round-trip')
    return result

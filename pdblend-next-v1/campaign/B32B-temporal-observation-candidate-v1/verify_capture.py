"""Offline bounded log verification. Complete observation is not KV correctness."""
import hashlib
import json
import math
from pathlib import Path
from pdblend_diagnostics import validate_spec, STEPS, MAX_BYTES, MAX_RECORDS


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def verify(spec_path, log_dir, full_outputs_path):
    spec_path, root = Path(spec_path), Path(log_dir)
    raw = spec_path.read_bytes()
    spec = validate_spec(json.loads(raw))
    digest = hashlib.sha256(raw).hexdigest()
    full_outputs_path = Path(full_outputs_path)
    full_raw = full_outputs_path.read_bytes()
    full_outputs = json.loads(full_raw)['token_ids_by_request_uuid']
    expected = {(rid, step) for rid in spec['request_ids'] for step in STEPS}
    ranks, files = {}, {str(spec_path): hashlib.sha256(raw).hexdigest(),
        str(full_outputs_path): hashlib.sha256(full_raw).hexdigest()}
    require(set(full_outputs) >= set(spec['request_ids']), 'explicit full HTTP UUID outputs')
    for rid in spec['request_ids']:
        require(len(full_outputs[rid]) == 64 and all(type(x) is int for x in full_outputs[rid]),
                'complete original64 outputs required')
    for rank in (0, 1):
        logs = list(root.glob('rank%d-pid*.jsonl' % rank))
        require(len(logs) == 1, 'one real process per TP rank')
        p = logs[0]
        data = p.read_bytes()
        require(len(data) <= MAX_RECORDS * MAX_BYTES, 'bounded log')
        rows = [json.loads(x) for x in data.splitlines()]
        require(len(rows) == MAX_RECORDS, 'all fourteen captures')
        status_path = p.with_suffix('.status.json')
        status = json.loads(status_path.read_text())
        require(status['rank'] == rank and status['complete'] is True and
                status['failed'] is False and status['error'] is None and status['written'] == 14,
                'writer terminal completeness')
        keyed = {}
        for r in rows:
            key = (r['request_id'], r['output_index'])
            require(key in expected and key not in keyed, 'UUID/step completeness')
            require(r['spec_sha256'] == digest and r['rank'] == rank and r['pid'] == status['pid']
                    and r['observer_error'] is None and r['tp'] == 2 and r['pp'] == 1 and r['eager'] is True
                    and r['tensor_device'] == 'cuda:%d' % rank,
                    'source/spec/process identity')
            require(math.isfinite(r['host_readback_s']) and r['host_readback_s'] >= 0, 'readback time')
            a = r['actual']
            for k in ('input_token', 'position', 'slot', 'sequence_len', 'context_len'):
                require(len(a[k]) == 1 and type(a[k][0]) is int, 'integer scalar metadata')
            require(all(type(x) is int for x in a['block_table']) and 0 < len(a['block_table']) <= 64,
                    'bounded blocks')
            if rank == 0:
                require(r['sampler_parent_seq_id'] == r['seq_id'] and
                        r['sampler_output_token'] == full_outputs[key[0]][key[1]-1],
                        'actual sampler output equals completed HTTP token by UUID')
                l = r['raw_model_logits_pre_sampler']
                require(len(l['top2_values']) == len(l['top2_token_ids']) == 2 and
                        all(math.isfinite(v) for v in l['top2_values'] + [l['token_2776'],l['token_4172']]),
                        'finite actual logits')
            else:
                require(r['raw_model_logits_pre_sampler'] is None and r['sampler_output_token'] is None,
                        'rank1 no invented logits/sampling')
            keyed[key] = r
        require(set(keyed) == expected, 'seven selected steps per UUID per rank')
        ranks[rank] = keyed
        for q in (p, status_path):
            files[str(q)] = hashlib.sha256(q.read_bytes()).hexdigest()
    require(next(iter(ranks[0].values()))['pid'] != next(iter(ranks[1].values()))['pid'],
            'two distinct actual TP worker processes')
    comparisons = []
    for key in sorted(expected):
        a, b = ranks[0][key], ranks[1][key]
        checks = {}
        for rank, r in ((0, a), (1, b)):
            m = r['actual']
            pos, block_size = m['position'][0], r['block_size']
            table = m['block_table']
            physical_slot = table[pos // block_size] * block_size + pos % block_size if 0 <= pos // block_size < len(table) else None
            checks[str(rank)] = dict(
                uuid_seq_binding=[r['request_id'],r['seq_id']] in r['ordered_request_seq_ids'],
                position_equals_computed_before=pos == r['computed_before'],
                input_equals_driver=m['input_token'][0] == r['last_input_token'],
                attention_len_equals_sequence=m['sequence_len'][0] == r['sequence_len'],
                query_offsets_match=m['query_start_end'] == [r['input_token_offset'],r['input_token_offset']+1],
                slot_matches_actual_block_table=m['slot'][0] == physical_slot,
                actual_blocks_match_driver=table[:len(r['driver_block_table'])] == r['driver_block_table'])
        comparisons.append(dict(request_id=key[0],output_index=key[1],
            rank_metadata_equal=a['actual'] == b['actual'], checks=checks,
            raw_logits=a['raw_model_logits_pre_sampler'],sampled_token=a['sampler_output_token'],
            measured_host_readback_s=[a['host_readback_s'],b['host_readback_s']]))
    return dict(schema=1,capture_complete=True,hardware_correctness_proven=False,
                kv_contents_observed=False,performance_evidence=False,records=comparisons,inputs=files)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    p.add_argument('--logs', required=True)
    p.add_argument('--full-outputs', required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.spec, a.logs, a.full_outputs), indent=2, allow_nan=False))

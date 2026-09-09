"""Validate A's preregistered EcoServe original-rate group before final scope."""
from pathlib import Path
from baseline_row_metadata_v1 import metadata

DECLARATION_SHA = 'd8bb8fb072658a9e28c489647130082bb44f80a48b3394af8d25bce18b9f67b2'


def validate(p, reference, d, originals, sources):
    p.need(reference['sha256'] == DECLARATION_SHA
           and d['schema'] == 'A-Eco-whole-source-logical31-v1'
           and d['logical_declared_count'] == 31
           and d['actual_final_pdb_terminal_required'] is True
           and d['new_rate_rows_require_explicit_append_declaration'] is True,
           'unrecognized A EcoServe logical group')
    for ref in (reference, d['source'], d['original_snapshot'], d['source_cpu'], d['restore_cpu']):
        p.checked(ref)
        sources[ref['path']] = ref['sha256']
    host_root = Path(d['source']['path']).parent
    for name, digest in p.checked(d['source'])['files'].items():
        p.need(p.sha(host_root / name) == digest, 'A EcoServe source changed')
    p.need(d['original_policy_profiles_unchanged'] is True and d['window_s'] == 100
           and d['request_timeout_s'] == 120 and d['seed'] == 701
           and d['all8gpu_power'] is True and d['deadline_s'] is None,
           'A EcoServe changed scientific budgets')
    old = {x['cell_id']: x for x in originals if x['model'] == '14b' and x['system'] == 'ecoserve'}
    rows = {x['cell_id']: x for x in d['cells']}
    maps = {x['new_cell_id']: x for x in d['replacement_mapping']}
    p.need(len(old) == 30 and len(rows) == len(d['cells']) == len(maps)
           == len(d['replacement_mapping']) == 31 and set(rows) == set(maps),
           'A whole original30 and one extra repetition required')
    fresh, retired = {}, {}
    for cid, row in rows.items():
        m = maps[cid]
        lid = m['old_cell_id']
        p.need(lid in old and row['repeat'] == m['repeat']
               and row['model'] == '14b' and row['system'] == 'ecoserve'
               and row['baseline_controller_host'] == str(host_root), 'A mapping changed')
        cp = p.checked(m['checkpoint'])
        sources[m['checkpoint']['path']] = m['checkpoint']['sha256']
        p.need(cp['row']['cell_id'] == lid, 'A original checkpoint mismatch')
        a, b = metadata(p, row), metadata(p, cp['row'])
        for key in (*p.PAIR_FIELDS, 'system', 'trace_path', 'arrival_window_s', 'slo_scale'):
            p.need(a[key] == b[key], 'A exact scientific row changed: ' + key)
        sources[row['trace_path']] = row['trace_sha256']
        if row['repeat'] == 2:
            p.need(lid == '14b-alpaca-r12-s701-w100-ecoserve-slo1', 'A undeclared extra repeat')
        else:
            p.need(row['repeat'] == 1 and lid not in retired, 'A repeated original observation')
            retired[lid] = cid
        fresh[cid] = dict(row=row, declaration=reference, host_release=str(host_root),
            host_manifest=d['source'], existing_original_rate=True,
            original_logical_baseline_id=lid, execution_scope_pending_final_pdb=True,
            reuse_original_rate_first_repeat=row['repeat'] == 1 and row['rate_rps'] != 12)
    p.need(set(retired) == set(old), 'A original group not completely replaced')
    return fresh, retired

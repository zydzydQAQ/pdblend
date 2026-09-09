"""Audit B's complete EcoServe group and its two pre-existing boundary exclusions."""
from pathlib import Path
from baseline_row_metadata_v1 import metadata

DECLARATION_SHA = 'b71286d6423ca7195c99fe3040bec9ebe59887c33b1708d5e3aa36fd5364332d'


def validate(p, reference, d, originals, sources):
    p.need(reference['sha256'] == DECLARATION_SHA
           and d['schema'] == 'B-Eco-window-drain-whole-group-v1'
           and d['old_raw_unchanged'] is True, 'unrecognized B EcoServe replacement')
    refs = [reference, d['host_manifest'], d['cpu_validation'], d['parent_snapshot'],
            d['newrate_parent'], d['original_AB_window_audit'],
            d['completed_previous_cooperative'], d['pdb_boundary_evidence']]
    for ref in refs:
        p.checked(ref)
        sources[ref['path']] = ref['sha256']
    host = p.checked(d['host_manifest'])
    host_root = Path(d['host_release'])
    p.need(d['host_manifest']['path'] == str(host_root / 'manifest.json')
           and d['declared_count'] == 37 and d['execution_count'] == 35
           and d['arrival_window_s'] == 100 and d['request_hard_timeout_s'] == 120
           and d['cleanup_local_budget_s'] == 90 and d['deadline_s'] is None
           and d['energy_gpu_indices'] == list(range(8)), 'B replacement changed source or measurement budget')
    for name, digest in host['files'].items():
        p.need(p.sha(host_root / name) == digest, 'B EcoServe source changed')
    old = {x['cell_id']: x for x in originals if x['model'] == '32b' and x['system'] == 'ecoserve'}
    parent = p.checked(d['newrate_parent'])
    new_rates = {x['cell_id']: x for x in parent['cells']
                 if x['system'] == 'ecoserve'
                 and (x['dataset'], x['rate_rps']) in {('alpaca', 5.), ('sharegpt', 1.25), ('sharegpt', 1.5)}}
    p.need(len(old) == 30 and len(new_rates) == 6, 'B whole original30 and new six required')
    declared = {x['cell_id']: x for x in d['declared_cells']}
    active = {x['cell_id']: x for x in d['cells']}
    excluded = {x['row']['cell_id']: x['row'] for x in d['excluded_cells']}
    p.need(len(declared) == len(d['declared_cells']) == 37
           and len(active) == len(d['cells']) == 35 and len(excluded) == 2
           and not set(active) & set(excluded) and {**active, **excluded} == declared,
           'B active and excluded group membership differs')
    expected_excluded = {'32b-sharegpt-r2-s701-w100-ecoserve-slo1',
                         '32b-longbench-r1-s701-w100-ecoserve-slo1'}
    p.need({x['logical_cell_id'] for x in excluded.values()} == expected_excluded,
           'B may exclude only the original two above-boundary points')
    status = p.checked(d['pdb_boundary_evidence'])
    p.need(status['complete'] is True and status['phase'] == 'complete'
           and not status['failed'] and status['node_lease_held'] is False
           and status['first_complete_breach']['sharegpt'] == 1.5
           and status['first_complete_breach']['longbench'] == .3,
           'B boundary exclusion authority differs')
    for row in excluded.values():
        p.need(row['rate_rps'] > status['first_complete_breach'][row['dataset']],
               'B excluded a point below the declared boundary')
    fresh, retired = {}, {}
    for cid, row in declared.items():
        lid = row['logical_cell_id']
        original = lid in old
        p.need(lid in old or lid in new_rates, 'B replacement introduced another logical point')
        p.need(row['model'] == '32b' and row['system'] == 'ecoserve', 'B replacement changed system/model')
        if original:
            cp = p.checked(row['original_checkpoint'])
            expected = cp['row']
            sources[row['original_checkpoint']['path']] = row['original_checkpoint']['sha256']
            p.need(expected['cell_id'] == lid, 'B original checkpoint ID differs')
        else:
            expected = new_rates[lid]
        p.need(row['source_row'] == expected, 'B replacement changed original scientific row')
        actual_meta, expected_meta = metadata(p, row), metadata(p, expected)
        for field in (*p.PAIR_FIELDS, 'system', 'trace_path', 'arrival_window_s', 'slo_scale'):
            p.need(actual_meta[field] == expected_meta[field], 'B exact pair changed: ' + field)
        sources[row['trace_path']] = row['trace_sha256']
        special = row['replacement_scope'] == 'authorized-adjacent-repair-control'
        if special:
            p.need(lid == '32b-longbench-r0.25-s701-w100-ecoserve-slo1' and row['repeat'] == 2,
                   'only declared LB0.25 second control is allowed')
        else:
            p.need(row['repeat'] == (1 if original else expected['repeat']) and lid not in retired,
                   'B changed or duplicated repetition')
            retired[lid] = cid
        if cid in active:
            fresh[cid] = dict(row=row, declaration=reference, host_release=str(host_root),
                host_manifest=d['host_manifest'], existing_original_rate=original,
                original_logical_baseline_id=lid,
                reuse_original_rate_first_repeat=original and row['repeat'] == 1)
    p.need(set(retired) == set(old) | set(new_rates) and len(fresh) == 35,
           'B did not replace the complete required baseline group')
    second = {p.pair_identity(metadata(p, x['row'])) for x in fresh.values() if x['row']['repeat'] == 2}
    for item in fresh.values():
        if p.pair_identity(metadata(p, item['row'])) in second:
            item['reuse_original_rate_first_repeat'] = False
    return fresh, retired

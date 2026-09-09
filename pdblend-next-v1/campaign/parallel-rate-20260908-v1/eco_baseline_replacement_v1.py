"""Validate the complete declared C EcoServe source replacement, without outcomes."""
from pathlib import Path
from baseline_row_metadata_v1 import metadata

DECLARATION_SHA = '5caeffe98efdb9b0a4fdaf8a6d134f9b13851d6c3dcbe3171b4819237b9d0cb9'


def validate(p, reference, document, originals, sources):
    d = document
    p.need(reference['sha256'] == DECLARATION_SHA
           and d['schema'] == 'C-EcoServe-whole-source-replacement-37-v1'
           and d['authorized'] is True and d['pointwise_best_selection_forbidden'] is True,
           'unrecognized EcoServe whole-source replacement')
    for path, digest in d['files'].items():
        p.need(p.sha(path) == digest, 'EcoServe declaration dependency changed')
        sources[path] = digest
    parent = p.checked(d['parent_declaration'])
    host = p.checked(d['source'])
    p.checked(d['original_snapshot'])
    host_root = Path(d['source']['path']).parent
    p.need(d['baseline_controller_hosts'] == {'ecoserve': str(host_root)}
           and d['whole_group_required'] == 37 and d['original_policy_profiles_unchanged'] is True
           and d['request_timeout_s'] == 120 and d['window_s'] == 100 and d['seed'] == 701
           and d['deadline_s'] is None and d['all8gpu_power'] is True,
           'EcoServe replacement changed source, trace or budgets')
    for name, digest in host['files'].items():
        p.need(p.sha(host_root / name) == digest, 'EcoServe replacement serving code changed')
    for ref in (reference, d['parent_declaration'], d['source'], d['original_snapshot']):
        sources[ref['path']] = ref['sha256']
    old_original = {x['cell_id']: x for x in originals
                    if x['model'] == '7b' and x['system'] == 'ecoserve'}
    old_new = {x['cell_id']: x for x in parent['cells'] if x['system'] == 'ecoserve'}
    p.need(len(old_original) == 30 and len(old_new) == 6, 'complete old EcoServe group required')
    expected_old = set(old_original) | set(old_new)
    p.need(len(d['retired_group']) == 36
           and {x['old_cell_id'] for x in d['retired_group']} == expected_old,
           'EcoServe replacement must retire the whole old group')
    rows = {x['cell_id']: x for x in d['cells']}
    mappings = {x['new_cell_id']: x for x in d['replacement_mapping']}
    p.need(len(rows) == len(d['cells']) == len(mappings) == len(d['replacement_mapping']) == 37
           and set(rows) == set(mappings)
           and len(d['priority_first8']) == 8 and len(d['remaining29']) == 29
           and d['priority_first8'] + d['remaining29'] == [x['cell_id'] for x in d['cells']],
           'EcoServe declared execution membership or priority differs')
    fresh, retired = {}, {}
    for cid, row in rows.items():
        mapping = mappings[cid]
        origin = mapping['origin']
        previous_id = origin['old_cell_id']
        original_rate = previous_id in old_original
        p.need(previous_id in expected_old and row['model'] == '7b' and row['system'] == 'ecoserve'
               and row['repeat'] == mapping['repeat']
               and row['baseline_controller_host'] == str(host_root), 'EcoServe mapping changes model, strategy or repeat')
        if original_rate:
            previous = old_original[previous_id]
            cp = p.checked(origin['checkpoint'])
            p.need(cp['row']['cell_id'] == previous_id, 'EcoServe original checkpoint differs')
            sources[origin['checkpoint']['path']] = origin['checkpoint']['sha256']
            expected = cp['row']
        else:
            p.need(origin['declaration'] == d['parent_declaration'], 'EcoServe new-rate origin differs')
            expected = old_new[previous_id]
        actual_metadata, expected_metadata = metadata(p, row), metadata(p, expected)
        for field in (*p.PAIR_FIELDS, 'system', 'trace_path', 'arrival_window_s', 'slo_scale'):
            a, b = actual_metadata[field], expected_metadata[field]
            p.need(a == b, 'EcoServe changed exact trace/SLO/work denominator: ' + field)
        p.need(p.sha(row['trace_path']) == row['trace_sha256'], 'EcoServe trace bytes changed')
        sources[row['trace_path']] = row['trace_sha256']
        special = origin['kind'] == 'authorized_original_alp12_second_repeat'
        if special:
            p.need(previous_id == '7b-alpaca-r12-s701-w100-ecoserve-slo1' and row['repeat'] == 2,
                   'only the declared original Alpaca12 control has an extra repetition')
        else:
            p.need(origin['kind'] == ('original_main30' if original_rate else 'original_new_rate6')
                   and row['repeat'] == (1 if original_rate else expected['repeat'])
                   and previous_id not in retired, 'EcoServe changed or repeated a source mapping')
            retired[previous_id] = cid
        fresh[cid] = dict(row=row, declaration=reference, host_release=str(host_root),
                          host_manifest=d['source'], existing_original_rate=original_rate,
                          original_logical_baseline_id=previous_id,
                          reuse_original_rate_first_repeat=original_rate and row['repeat'] == 1)
    p.need(set(retired) == expected_old, 'EcoServe original group replacement is incomplete')
    # The existing protocol reuses original-rate baseline R1 for PDB R2.
    # Where a baseline R2 is explicitly declared, wait for that R2 instead.
    dedicated_second = {p.pair_identity(metadata(p, x['row']))
                        for x in fresh.values() if x['row']['repeat'] == 2}
    for item in fresh.values():
        if p.pair_identity(metadata(p, item['row'])) in dedicated_second:
            item['reuse_original_rate_first_repeat'] = False
    return fresh, retired

"""Read-only exact main references; never installs CPs in a new output tree."""
import copy
from pathlib import Path
import contract as c


def package_check():
    manifest=c.read(c.HERE/'manifest.json')
    for name,h in manifest['files'].items():c.require(c.sha(c.HERE/name)==h,'scale adapter source changed: '+name)
    for name,h in manifest['dependencies'].items():c.require(c.sha(name)==h,'scale adapter dependency changed: '+name)


def mappings(checked):
    byid={r['row']['cell_id']:r for r in checked['main_records']}
    return {r['cell_id']:dict(scale_row=copy.deepcopy(r),main=copy.deepcopy(byid[r['reuse_main_cell_id']]))
        for r in checked['rows']}


def binding_metadata(binding,source_ref,release_path,release_sha,datasets):
    """Metadata for a future fresh binder; never writes or refreshes identity.

    Caller must write a NEW binding, include its real inventory, and pass the
    complete check_spec/actual engine gates. This helper cannot qualify a PID.
    """
    package_check();c.released.verify_release(release_path,release_sha,expected_model=binding['model'])
    proof=c.read(release_path)['models'][binding['model']]
    c.require(source_ref==dict(path=proof['source_manifest'],sha256=proof['source_sha256']),
        'scale binding metadata needs the original full source')
    records,_=c.released_main_records(dict(system=binding['system'],datasets=datasets),proof,c.reference(source_ref))
    refs=list({(r['execution']['execution_manifest']['path'],r['execution']['execution_manifest']['sha256']):
        r['execution']['execution_manifest'] for r in records}.values())
    result=copy.deepcopy(binding)
    for ref in [source_ref,*refs]:
        c.require(c.sha(ref['path'])==ref['sha256'],'actual historical execution manifest changed')
        c.require(result['files'].get(ref['path'],ref['sha256'])==ref['sha256'],'binding has conflicting source evidence')
        result['files'][ref['path']]=ref['sha256']
    result['execution_manifest']=copy.deepcopy(source_ref)
    result['main_execution_manifests']=copy.deepcopy(refs)
    return result


def build(spec_path,spec_sha,release_path,release_sha,group_id):
    spec=c.reference(dict(path=str(spec_path),sha256=spec_sha))
    checked=c.check_spec(spec,release_path,release_sha,[group_id])['groups'][0]
    c.require(checked['current'] is not None,'reuse-only group cannot create a measurement map')
    return dict(schema=2,kind='scale-main-reference-map',protocol_id=c.PROTOCOL,deadline_s=c.DEADLINE,
        spec=dict(path=str(spec_path),sha256=spec_sha),release=dict(path=str(release_path),sha256=release_sha),
        group_id=group_id,source=copy.deepcopy(spec['source']),binding=copy.deepcopy(checked['group']['scale_binding']),
        model=spec['model'],system=checked['group']['system'],datasets=copy.deepcopy(checked['group']['datasets']),
        references=mappings(checked),scope='Each main CP remains in its actual producer output; no copied checkpoint is authorized.')


def verify(path,expected_sha,manifest_path,binding_path,system,datasets,row=None):
    package_check()
    value=c.reference(dict(path=str(path),sha256=expected_sha))
    c.require(value.get('schema')==2 and value.get('kind')=='scale-main-reference-map'
        and value.get('protocol_id')==c.PROTOCOL and value.get('deadline_s')==c.DEADLINE,'wrong reference-map protocol')
    spec=c.reference(value['spec'])
    c.require(value['source']==spec['source'] and str(Path(manifest_path).resolve())==value['source']['path']
        and c.sha(manifest_path)==value['source']['sha256'],'scale driver source differs from released original declaration')
    c.require(str(Path(binding_path).resolve())==value['binding']['path'] and c.sha(binding_path)==value['binding']['sha256'],
        'scale driver did not use its fixed current binding')
    c.require(value['model']==spec['model'] and value['system']==system and set(value['datasets'])==set(datasets),
        'scale driver model/system/datasets differ from reference map')
    checked=c.check_spec(spec,value['release']['path'],value['release']['sha256'],[value['group_id']])['groups'][0]
    c.require(checked['group']['scale_binding']==value['binding'] and checked['group']['system']==system
        and checked['group']['datasets']==value['datasets'],'map selected another physical/source group')
    expected=mappings(checked)
    c.require(value['references']==expected,'map main row/source/config/invocation/process evidence differs from actual released producer')
    if row is not None:
        c.require(row.get('phase')=='scale' and row.get('cell_id') in expected
            and row==expected[row['cell_id']]['scale_row'],'requested scale row/SLO/work differs from original declaration')
    c.require(c.sha(path)==expected_sha,'reference map changed during raw review')
    return value

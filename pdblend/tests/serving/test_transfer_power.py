import json
from pathlib import Path

import pytest
from ecopadg.serving.prefill_batch import build, transfer_energy_components, TRANSFER_ENERGY_ACCOUNTING
from ecopadg.serving.transfer_validation import merge
from ecopadg.serving.interconnect import InterconnectTopology
from ecopadg.serving.evidence import sha256


def instant(power):
    source=dict(mode='instant',source_id='nvml:field:186:scope:0:mW',field_id=186,scope_id=0,unit='W')
    metadata=[dict(t_s=t,gpus=list(range(8)),mode=['instant']*8,source_id=[source['source_id']]*8,
        field_id=[186]*8,scope_id=[0]*8,value_type=[1]*8,return_code=[0]*8,
        nvml_timestamp_us=[int(t*1e6)]*8,nvml_latency_us=[10]*8,
        read_started_s=[t]*8,read_finished_s=[t]*8) for t,_ in power]
    return dict(power_samples=power,power_source=source,power_metadata=metadata)


def raw(frequency=2520):
    return dict(instant([(1,[20]*8),(2,[100]*8),(3,[200]*8)]),complete=True,sampling_error=None,
        topology=dict(prefill=dict(tp=1,gpus=[0]),decode=dict(tp=1,gpus=[1])),
        frequency_samples=[[1,[2520,frequency]+[2520]*6]],commanded_frequencies={'0':2520,'1':frequency},
        decode_frequency_mhz=frequency,power_limit_w=[350]*8,idle_start_s=1,idle_end_s=1.5,
        engine_provenance=[dict(image_id='sha256:'+'a'*64,source_files_at_import={'/serving/engine.py':'engine-fixture'})],
        runs=[dict(skipped=False,input_tokens=128,batch=1,step=dict(started_s=2),
            send=[dict(started_s=2.1,finished_s=2.2)],receive=[dict(started_s=2.2,finished_s=2.3)],output_matches=True)])


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def test_prefill_batch_build_requires_instant_metadata_and_retains_destination_clock():
    topology=InterconnectTopology.parse('GPU0 X PIX\nGPU1 PIX X\n')
    value=raw(900)
    profiles,links=build(value,'raw-fixture',topology)
    assert profiles['power_source_verified'] and links[0]['decode_frequency_mhz']==900
    assert profiles['transfer_energy_accounting']==TRANSFER_ENERGY_ACCOUNTING
    value['power_source']['mode']='average'
    with pytest.raises(ValueError,match='instant power evidence'): build(value,'raw-fixture',topology)
    value=raw();value['power_metadata'].pop()
    with pytest.raises(ValueError,match='instant power evidence'): build(value,'raw-fixture',topology)


def cost_manifest(tmp_path,average=False):
    topo=tmp_path/'topology.txt';topo.write_text('GPU0 X PIX\nGPU1 PIX X\n')
    topology=InterconnectTopology.parse(topo.read_text())
    diagnostic=save(tmp_path/'diagnostic.json',dict(complete=True,passed=True,
        engine_provenance=raw()['engine_provenance'],cases=[dict(source_tp=1,target_tp=1,passed=True)]))
    directories=[]
    for frequency in (900,2520):
        value=raw(frequency);directory=tmp_path/str(frequency)
        profiles,links=build(value,'pending',topology)
        if average and frequency==900: value['power_source']['mode']='average'
        path=save(directory/'raw.json',value)
        for link in links: link['source_sha256']=sha256(path)
        profiles.update(source_sha256=sha256(path),frequency_samples_source_sha256=sha256(path))
        save(directory/'profiles.json',profiles)
        save(directory/'transfers.json',links);directories.append(str(directory))
    return dict(interconnect=str(topo),engine_image='sha256:'+'a'*64,
        diagnostics=[str(diagnostic)],cost_directories=directories)


@pytest.mark.parametrize('average',[False,True])
def test_transfer_certification_keeps_correctness_separate_but_rejects_old_average_costs(tmp_path,average):
    manifest=cost_manifest(tmp_path,average)
    if average:
        with pytest.raises(ValueError,match='instant power evidence'): merge(manifest)
    else:
        result=merge(manifest)
        assert result['certified'] and result['instant_power_costs_verified']
        assert {link['decode_frequency_mhz'] for link in result['links']}=={900,2520}
        assert result['schema']==3 and result['receiver_transfer_energy_included']
        assert all(str(Path(directory)/'profiles.json') in result['certification_artifacts']
                   for directory in manifest['cost_directories'])


@pytest.mark.parametrize('batch',[1,4])
def test_builder_charges_receiver_during_send_with_its_own_idle_and_ignores_other_gpus(batch):
    value=raw(900)
    # P idle=20 W, D idle=50 W. Both are active throughout send and import.
    # Six unrelated resident GPUs draw 300 W each and must not be charged.
    value.update(instant([(t,([20,50] if t<=2 else [120,250])+[300]*6) for t in range(1,7)]))
    value.update(idle_start_s=1,idle_end_s=2)
    value['runs']=[dict(skipped=False,input_tokens=128,batch=batch,step=dict(started_s=2.5),
        send=[dict(started_s=3,finished_s=4)],receive=[dict(started_s=4,finished_s=5)],output_matches=True)]
    profiles,links=build(value,'unchanged-raw',InterconnectTopology.parse('GPU0 X PIX\nGPU1 PIX X\n'))
    components=profiles['transfer_energy_components'][0]
    assert components['source_send_incremental_j']==pytest.approx(100)
    assert components['target_send_incremental_j']==pytest.approx(200)
    assert components['target_import_nonoverlap_incremental_j']==pytest.approx(200)
    assert components['source_idle_w']==20 and components['target_idle_w']==50
    assert components['import_nonoverlap_windows_s']==[[4,5]]
    assert links[0]['incremental_j']==pytest.approx(500)
    assert links[0]['profile_batch']==batch  # Group cost is neither divided nor multiplied by batch.
    assert links[0]['seconds_upper']==pytest.approx(2.2)
    assert links[0]['import_seconds_upper']==pytest.approx(1.1)


@pytest.mark.parametrize('import_window,remaining,expected',[
    ((4,6),[[5,6]],800),       # Overlap on the right.
    ((2,4),[[2,3]],800),       # Overlap on the left.
    ((3.5,4.5),[],600),        # Import lies entirely within send.
    ((2,6),[[2,3],[5,6]],1000),# Import contains send.
    ((3,5),[],600),           # Exact overlap.
    ((1,2),[[1,2]],800),      # Disjoint before send.
    ((6,7),[[6,7]],800),      # Disjoint after send.
    ((5,6),[[5,6]],800),      # Touching boundaries have no overlap.
])
def test_receiver_send_import_union_never_charges_overlap_twice(import_window,remaining,expected):
    calls=[]
    def watts(instance,start,end):
        calls.append((instance,start,end))
        return {'P':120,'D':250}[instance]
    result=transfer_energy_components(watts,'P','D',20,50,(3,5),import_window)
    assert result['import_nonoverlap_windows_s']==remaining
    assert result['incremental_j']==pytest.approx(expected)
    assert calls==[('P',3,5),('D',3,5),*(('D',a,b) for a,b in remaining)]


def test_endpoint_idle_clamps_do_not_offset_each_other():
    result=transfer_energy_components(lambda instance,a,b:120 if instance=='P' else 40,
        'P','D',20,50,(3,4),(4,5))
    assert result['source_send_incremental_j']==100
    assert result['target_send_incremental_j']==0
    assert result['target_import_nonoverlap_incremental_j']==0
    assert result['incremental_j']==100


@pytest.mark.parametrize('send,receive',[
    ((3,3),(4,5)),((4,3),(4,5)),((3,float('nan')),(4,5)),
    ((3,4),(5,5)),((3,4),(5,4)),((3,4),(4,float('inf'))),
])
def test_transfer_energy_rejects_invalid_windows(send,receive):
    with pytest.raises(ValueError,match='positive transfer energy windows'):
        transfer_energy_components(lambda *args:100,'P','D',20,50,send,receive)


@pytest.mark.parametrize('tamper',['omit_receiver_send','change_incremental_energy'])
def test_certificate_rejects_costs_that_disagree_with_raw_rebuild(tmp_path,tamper):
    manifest=cost_manifest(tmp_path)
    directory=Path(manifest['cost_directories'][0])
    links=json.loads((directory/'transfers.json').read_text())
    components=json.loads((directory/'profiles.json').read_text())['transfer_energy_components'][0]
    if tamper=='omit_receiver_send':
        assert components['target_send_incremental_j']>0
        links[0]['incremental_j']-=components['target_send_incremental_j']
    else:
        links[0]['incremental_j']+=1
    save(directory/'transfers.json',links)
    with pytest.raises(ValueError):merge(manifest)

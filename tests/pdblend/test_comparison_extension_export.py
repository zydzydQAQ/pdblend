import asyncio
import csv
import json
from pathlib import Path

import pytest

from pdblend.bench import comparison_campaign as campaign
from pdblend.bench.comparison_extension_export import extension_watch_paths
from pdblend.bench.comparison_recorded import POLICY
from pdblend.bench import single_observation_slo_boundary as boundary
from pdblend.bench.single_observation_slo_boundary import SCHEMA
from pdblend.bench.resident_session import ResidentGroupSession
from test_comparison_recorded import write
from test_single_observation_slo_boundary import fixture, fake_factory, Adapter


def extension_fixture(tmp_path, monkeypatch):
    group = fixture(tmp_path/'prepared')
    for point in group['points']:
        point.update(status='prepared', run_id='round-new')

    def factory(ref, dataset, scale, out):
        point, _ = fake_factory(ref, dataset, scale, out)
        point.update(status='prepared', run_id='round-new')
        path = Path(out)/'export-point.json'; write(path, point)
        return point, campaign.binding(path)

    monkeypatch.setattr(boundary, 'make_extension_point', factory)
    session = tmp_path/'session'
    report = asyncio.run(ResidentGroupSession(group, Adapter(), session).run())
    assert report['complete']
    manifest_path = Path(report['extension_manifest']['path'])
    manifest = json.loads(manifest_path.read_text())
    source, output = tmp_path/'campaign.json', tmp_path/'compare.csv'
    write(source, dict(campaign_id='new-run', points=group['points'],
                       extension_policy_refs=[group['extension_policy']]))
    return (session, source, output, session/'extensions/latest.json', manifest_path,
            Path(manifest['points'][0]['point']['path']), None)


def test_authorized_extension_is_discovered_without_replacing_base_inventory(tmp_path, monkeypatch):
    session, source, output, pointer, manifest, point, _ = extension_fixture(tmp_path, monkeypatch)
    original = source.read_bytes()
    predecessors = sorted((session/'extensions').glob('manifest-*.json'))[:3]
    campaign.export(source, output, session_roots=[session], analysis_policy=POLICY,
                    extension_manifests=[*predecessors, pointer, manifest])
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == len({r['receipt_path'] for r in rows}) == 6
    extension = next(r for r in rows if r['point_id'] == '7b-pdblend-alpaca-x1.25-seed701')
    assert extension['run_id'] == 'round-new' and extension['rate_scale'] == '1.25'
    assert extension['boundary_scope'] == SCHEMA and extension['receipt_path']
    assert extension['boundary_role'] == 'L' and extension['status'] == 'measured'
    assert extension['energy_service_j'] == '' and extension['energy_rank'] == ''
    assert extension['boundary_manifest_sha256'] == campaign.file_sha(manifest)
    assert source.read_bytes() == original
    assert pointer in extension_watch_paths(source, session_roots=[session])


def test_unlisted_policy_and_changed_extension_point_cannot_replace_export(tmp_path, monkeypatch):
    session, source, output, pointer, manifest, point, _ = extension_fixture(tmp_path, monkeypatch)
    output.write_text('prior frozen CSV')
    spec = json.loads(source.read_text()); spec['extension_policy_refs'] = []; write(source, spec)
    with pytest.raises(ValueError, match='not authorized'):
        campaign.export(source, output, extension_manifests=[pointer], analysis_policy=POLICY)
    assert output.read_text() == 'prior frozen CSV'
    # Automatic discovery ignores a different run's authorized manifest.
    campaign.export(source, output, session_roots=[session], analysis_policy=POLICY)
    assert len(list(csv.DictReader(output.open()))) == 4
    old = output.read_bytes()
    value = json.loads(point.read_text()); value['rate_rps'] = 999.; write(point, value)
    with pytest.raises(ValueError, match='checksum|binding'):
        campaign.export(source, output, extension_manifests=[manifest], analysis_policy=POLICY)
    assert output.read_bytes() == old


def test_expanded_scale_order_does_not_change_original_four_window_order():
    base = dict(model_id=campaign.MODELS[0], system='mixed', dataset='alpaca')
    points = [dict(base, scale=scale) for scale in (2., 1., .25, 1.25, .5, .75)]
    assert [p['scale'] for p in sorted(points, key=campaign.point_order)] == [.5, .25, .75, 1., 1.25, 2.]
    with pytest.raises(ValueError, match='positive'):
        campaign.point_order(dict(base, scale=0.))

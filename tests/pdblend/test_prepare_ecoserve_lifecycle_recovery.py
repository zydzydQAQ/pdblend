"""Builder opt-in is scoped to the bound unattempted selection."""
import json
from test_prepare_ecoserve_recovery import recovery, terminal_inventory
from pathlib import Path


def test_legacy_builder_does_not_add_lifecycle_mode_or_review(recovery):
    x = recovery
    queue, _ = terminal_inventory(x)
    x['argv'] += ['--unattempted-from', str(x['prior']), '--queue', str(queue)]
    x['module'].main()
    campaign = json.loads((x['out']/'campaign.json').read_text())
    point = next(p for p in campaign['points'] if p['name'] == 'never-executed')
    config = json.loads(Path(point['inputs']['system_config']['path']).read_text())
    assert 'eco_comparison_lifecycle' not in point
    assert 'eco_comparison_lifecycle' not in point['engine_identity']
    assert 'eco_comparison_lifecycle' not in config
    assert 'eco_lifecycle_review' not in point['inputs']

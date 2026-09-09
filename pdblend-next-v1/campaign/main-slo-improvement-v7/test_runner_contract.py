import json
from pathlib import Path
import subprocess
import sys
import pytest
import runner
import report

def test_existing_output_is_never_rewritten(tmp_path):
    out = tmp_path / 'existing'; out.mkdir()
    marker = out / 'status.json'; marker.write_text('{"complete": false, "preserve": true}')
    before = marker.read_bytes()
    result = subprocess.run([sys.executable, str(Path(runner.__file__)), '--release', str(tmp_path/'missing'),
        '--out', str(out), '--stage', 'screen_fixed2', '--run'], capture_output=True, text=True)
    assert result.returncode != 0 and 'existing evidence is immutable' in result.stderr
    assert marker.read_bytes() == before

def test_all_eight_board_interpolated_energy():
    rows = [dict(t_s=0, **{f'g{i}': i for i in range(8)}),
            dict(t_s=2, **{f'g{i}': i+2 for i in range(8)})]
    actual = report.integrate(rows, .5, 1.5, [f'g{i}' for i in range(8)])
    assert actual == pytest.approx([i+1 for i in range(8)])

def test_measurement_window_cannot_be_extrapolated():
    with pytest.raises(ValueError, match='coverage'):
        report.integrate([dict(t_s=0, g=1), dict(t_s=1, g=1)], 0, 2, ['g'])

def test_regressed_power_time_refused():
    with pytest.raises(ValueError, match='time order'):
        report.integrate([dict(t_s=0, g=1), dict(t_s=0, g=1), dict(t_s=2, g=1)], 0, 1, ['g'])

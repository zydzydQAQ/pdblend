"""The lookup contract is constant work, exact floats and bounded coverage."""
import builtins
import copy
import json
import math
from pathlib import Path
import pickle
import random

import pytest

from pdblend.profile.query.index import AxisIndex, CurveIndex, FrequencyIndex, IndexQualificationError
from pdblend.profile.query.power_table import CompiledPowerTable, PowerCoverageError, predict as reference_power
from pdblend.profile.query.long_context import CompiledLongTable, KIND as LONG_KIND, predict as reference_long
from pdblend.profile.query.runtime import RuntimeQualificationError
from pdblend.profile.query.versions import BoundedVersionModel
from pdblend.planner.pool import PoolPlanner, PlannerConfig, SLO
from test_power_override import model


def test_index_preserves_fractional_breakpoints_and_inputs():
    axis = AxisIndex([10.1, 10.2, 10.7, 10.9, 11.5])
    for x, expected in [(10.1,0), (10.15,0), (10.2,1), (10.65,1), (10.8,2), (10.9,3), (11.49,3), (11.5,4)]:
        assert axis.lower(x) == expected
    curve = CurveIndex([10.1, 10.2, 10.7, 10.9, 11.5], [1, 8, 2, 7, 5])
    assert curve.predict(10.15) == pytest.approx(4.5)
    assert curve.predict(10.15) != curve.predict(10.1)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), None, 10., 11.51])
def test_invalid_and_outside_values_cannot_index(value):
    with pytest.raises(ValueError, match='missing_profile'):
        AxisIndex([10.1,11.5]).lower(value)


def test_dense_or_large_directory_has_no_scan_fallback():
    with pytest.raises(IndexQualificationError, match='four breakpoints'):
        AxisIndex([1.1,1.2,1.3,1.4,1.5])
    with pytest.raises(IndexQualificationError, match='memory limit'):
        AxisIndex([0, 10**9])


class CountedKnots:
    def __init__(self, values): self.values, self.reads = values, 0
    def __len__(self): return len(self.values)
    def __getitem__(self, index):
        self.reads += 1
        return self.values[index]
    def __iter__(self): raise AssertionError('query scanned knots')


def test_lookup_access_bound_does_not_grow_from_16_to_1024():
    counts = []
    for size in (16,64,256,1024):
        axis = AxisIndex(range(size))
        counter = CountedKnots(axis.knots)
        axis.knots = counter
        assert axis.lower(size / 2 + .5) == size // 2
        counts.append(counter.reads)
    assert counts == [4,4,4,4]


def test_original_power_equations_and_coverage_on_fractional_batches():
    instance = model()
    spec = copy.deepcopy(instance.decode_power_overrides[900])
    table = CompiledPowerTable(spec)
    queries = [(b,c) for b in (1,4,4.5,7.999,8,16,1.5,2,3,16.1)
               for c in (499.99,500,503.3,510,777.77,1000,4000,4010,4010.1)]
    rng = random.Random(701)
    queries += [(rng.uniform(4,16),rng.uniform(500,4010)) for _ in range(100)]
    for batch, context in queries:
        try: expected = reference_power(spec,batch,context)
        except PowerCoverageError:
            with pytest.raises(PowerCoverageError): table.predict(batch,context)
        else:
            assert table.predict(batch,context) == pytest.approx(expected,rel=1e-14)


def long_spec():
    return dict(kind=LONG_KIND,exact_batches=[1,4,8],nodes={
        f'{f}/{b}':[dict(context=c,step_seconds=.01+b*c/1e7,power_w=150+b+c/100)
                    for c in (6000.1,6000.4,7168.5)] for f in (900,1500) for b in (1,4,8)})


def test_long_exact_batch_accepts_integer_float_but_no_new_domain():
    spec = long_spec(); table = CompiledLongTable(spec)
    for b in (1,4,8):
        for c in (6000.1,6000.3,7000.123,7168.5):
            for metric in ('step_seconds','power_w'):
                assert table.predict(metric,1500.0,float(b),c) == reference_long(spec,metric,1500,b,c)
    for b,c,f in [(4.5,7000,1500),(4,5999,1500),(4,7169,1500),(4,7000,1200),(math.nan,7000,1500),(4,math.inf,1500)]:
        with pytest.raises(ValueError): table.predict('step_seconds',f,b,c)


def test_frequency_nearest_is_exact_for_float_queries_and_original_ties():
    frequencies = (2520,900,1500)
    index = FrequencyIndex(frequencies)
    for frequency in (0,900,900.01,1200,1200.001,1499.9,2010,3000):
        assert index.nearest(frequency) == min(frequencies,key=lambda f:abs(f-frequency))
    assert not index.contains(1500.1)


def test_warm_queries_need_no_files_or_compile(monkeypatch):
    instance = model(); table = CompiledLongTable(long_spec())
    def forbidden(*args,**kwargs): raise AssertionError('hot path performed I/O or compilation')
    monkeypatch.setattr(builtins,'open',forbidden)
    monkeypatch.setattr(Path,'read_text',forbidden)
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    monkeypatch.setattr(CompiledPowerTable,'__init__',forbidden)
    for _ in range(100):
        assert instance.prefill_seconds(950,900)>0
        assert instance.step_seconds(4.5,1000.1,900)>0
        assert instance.decode_power_w(4.5,900,ctx=1000.1)>0
        assert table.predict('step_seconds',1500,4.,7000.123)>0


def test_mutating_calibration_input_recompiles_before_query():
    instance = model()
    spec = instance.decode_power_overrides[900]
    before = instance.decode_power_w(4,900,ctx=500)
    spec['nodes'][3]['power_w'] += 50
    assert instance.decode_power_w(4,900,ctx=500) == before+50
    spec['nodes'] = [dict(batch=4,context_min=500,context_max=510,power_w=333.)]
    assert instance.decode_power_w(4,900,ctx=505) == 333
    with pytest.raises(PowerCoverageError): instance.decode_power_w(8,900,ctx=1000)
    with pytest.raises(ValueError): spec['nodes'].append(dict(spec['nodes'][0]))
    with pytest.raises(PowerCoverageError): instance.decode_power_w(4,900,ctx=505)


def test_json_and_pickle_keep_compatibility_and_compiled_queries():
    instance = model()
    from pdblend.profile.model import PerfModel as old_type
    from pdblend.profile.query.model import PerfModel as new_type
    assert old_type is new_type
    for loaded in (old_type.from_json(instance.to_json()),pickle.loads(pickle.dumps(instance))):
        assert loaded.decode_power_w(4.5,900,ctx=777.77) == instance.decode_power_w(4.5,900,ctx=777.77)
        assert loaded.query_qualification['qualified'] is True
    old_path_pickle = pickle.dumps(instance,protocol=0).replace(
        b'pdblend.profile.query.model\n',b'pdblend.profile.model\n')
    assert pickle.loads(old_path_pickle).decode_power_w(4,900,ctx=1000) == 204


@pytest.mark.parametrize('invalid',[float('nan'),float('inf'),-float('inf')])
def test_scalar_model_rejects_nonfinite_values(invalid):
    instance = model()
    for query in (lambda:instance.prefill_seconds(invalid,900),
                  lambda:instance.prefill_power_w(invalid,900),
                  lambda:instance.step_seconds(invalid,1000,900),
                  lambda:instance.step_seconds(4,invalid,900),
                  lambda:instance.decode_power_w(4,900,ctx=invalid)):
        with pytest.raises(ValueError): query()
    assert not instance.decode_supported(4,invalid,900)
    assert not instance.decode_power_supported(4,invalid,900)


def test_frozen_identity_checks_actual_source_and_not_live_alias(tmp_path):
    import hashlib
    from pdblend.profile.query.compatibility import numerical_sources
    root = tmp_path/'pdblend/profile';root.mkdir(parents=True)
    old = root/'model.py'; old.write_text('from pdblend.profile.query.model import PerfModel\n')
    moved = root/'query/model.py';moved.parent.mkdir();moved.write_text('class PerfModel: pass\n')
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'pdblend/profile/model.py':sha(old),'pdblend/profile/query/model.py':sha(moved)}
    assert numerical_sources(root,manifest,['model.py']) == {'model.py':moved}
    moved.write_text('class PerfModel: changed=True\n')
    with pytest.raises(ValueError,match='checksum mismatch'):
        numerical_sources(root,manifest,['model.py'])
    moved.unlink()
    with pytest.raises(ValueError,match='alias is insufficient'):
        numerical_sources(root,manifest,['model.py'])


def test_component_profile_is_rejected_at_planner_entry_without_runtime_costs():
    instance = model(); instance.bounded_coverage = dict(prefill_tokens=[128,7168])
    bounded = BoundedVersionModel(instance,dict(version_id='fixture'))
    assert bounded.step_seconds(4,1000,900) > 0
    with pytest.raises(RuntimeQualificationError,match='capacity.*static.*transfer.*clock_transition'):
        PoolPlanner(bounded,PlannerConfig(8,SLO(5,.15)))
    for query in (lambda:bounded.kv_capacity_tokens,lambda:bounded.static_power_w('parked'),
                  lambda:bounded.transfer_seconds(512),lambda:bounded.freq_switch_s):
        with pytest.raises(RuntimeQualificationError): query()

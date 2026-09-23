from types import SimpleNamespace

from pdblend_runtime.cleanup import cleanup_owned


def test_cleanup_faults_do_not_skip_other_owned_gpu_or_sampler():
    seen=[]
    def operation(name,fail=False):
        def call(*args):
            seen.append((name,*args))
            if fail:raise RuntimeError(name)
        return call
    fleet=SimpleNamespace(instances={'p':SimpleNamespace(stop=operation('p',True)),
                                     'd':SimpleNamespace(stop=operation('d'))})
    meter=SimpleNamespace(gpus=[1,3],unpark=operation('unpark',True),reset_clock=operation('reset'))
    sampler=SimpleNamespace(stop=operation('sampler'))
    failures=cleanup_owned(fleet,meter,sampler)
    assert [r['component'] for r in failures]==['stop:p','unpark:1','unpark:3']
    assert seen==[('p',),('d',),('unpark',1),('reset',1),('unpark',3),('reset',3),('sampler',)]

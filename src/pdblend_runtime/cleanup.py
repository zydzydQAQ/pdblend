"""Best-effort cleanup of exactly one probe's owned instances and GPU group."""


def cleanup_owned(fleet, meter, sampler):
    failures=[]
    operations=[('stop:'+name,instance.stop) for name,instance in fleet.instances.items()]
    for gpu in meter.gpus:
        operations.extend([(f'unpark:{gpu}',lambda gpu=gpu:meter.unpark(gpu)),
                           (f'reset_clock:{gpu}',lambda gpu=gpu:meter.reset_clock(gpu))])
    operations.append(('stop_sampler',sampler.stop))
    for component,operation in operations:
        try:
            operation()
        except Exception as exc:
            failures.append(dict(component=component,error=repr(exc)))
    return failures

"""Create an auditable adaptation of the frozen original fixed100 executor."""
from pathlib import Path
import hashlib
import ast
import json
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / 'campaign/five-system-execution-v3/run.py'
PIN = '7c7dbe217243b42a8f93b57476ed457a6111e8f90c71ac4269130c9b46420f92'

def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('frozen adaptation target differs: '+old[:120])
    return source.replace(old, new)

def build(out):
    if out.exists():
        raise ValueError('fresh dynamic executor source required')
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        raise ValueError('original measurement source changed')
    source = raw.decode().split('\nasync def sweep(')[0]
    source = once(source, 'import time\n', 'import time\nimport dynamic_ownership as ownership\n')
    source = once(source, "    config = binding['configs'][row['dataset']]\n", """    config = binding['configs'][row['dataset']]
    dynamic_config = read(config)
    require(dynamic_config.get('capacity_integration_v1') is True, 'actual dynamic arm required')
    capacity = read(dynamic_config['capacity_binding_path'])
    require(sha(dynamic_config['capacity_binding_path']) == dynamic_config['capacity_binding_sha256'],
            'physical capacity binding changed')
    inventory_path = Path(dynamic_config['capacity_inventory_path'])
    require(not inventory_path.exists(), 'fresh per-cell capacity inventory required')
    lease = ownership.inherited_lease()
""")
    source = once(source, "        engine_ports=[i['port'] for i in binding['instances']])", """        engine_ports=[i['port'] for i in binding['instances']], lease=lease,
        controller_port=dynamic_config.get('port', 18080), inventory_path=str(inventory_path),
        capacity_identity=capacity['identity'], initial_instances=binding['instances'])""")
    source = once(source, "str(Path(__file__).with_name('child.py'))", "str(Path(__file__).with_name('dynamic_child.py'))")
    source = once(source, 'stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)',
                  "stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True, pass_fds=(lease['fd'],))")
    source = once(source, "        require(summary['measurement_end_s'] <= job['execution_deadline_s'], 'measurement exceeded reserved tail')", """        require(summary['measurement_end_s'] <= job['execution_deadline_s'], 'measurement exceeded reserved tail')
        value = ownership.inventory(inventory_path, binding['instances'], child_pid=child.pid,
                                    identity=capacity['identity'])
        ownership.validate_terminal(value, binding['instances'])
        receipt['dynamic_artifacts'] = ownership.transition_artifacts(value)
        receipt['dynamic_inventory_verified'] = True
""")
    source = once(source, "        receipt['child_stopped'] = child_stopped\n", """        receipt['child_stopped'] = child_stopped
        all_known = list(binding['instances'])
        live_inventory = None
        try:
            require(child_stopped, 'inventory cleanup requires actual child exit')
            if inventory_path.exists():
                live_inventory = ownership.inventory(inventory_path, binding['instances'],
                    child_pid=child.pid, identity=capacity['identity'])
                all_known = list(live_inventory['known_instances'].values())
                write(operation / 'inventory.final.json', live_inventory)
        except BaseException as exc:
            errors.append('dynamic ownership audit: ' + repr(exc))
""")
    source = once(source, "                    for i in binding['instances'] for port, rid in owned if i['port'] == port]",
                  "                    for i in all_known for port, rid in owned if i['port'] == port\n                    and (i.get('owner_kind') != 'created_for_cell' or i.get('state') not in ('stopped', 'stopped_after_failure'))]")
    source = once(source, "        receipt['restoration'] = {}\n", """        try:
            require(child_stopped, 'physical cleanup requires actual child exit')
            if live_inventory is not None:
                receipt['dynamic_outer_cleanup'] = await bounded(ownership.remove_owned_extras(
                    sys.modules[__name__], session, live_inventory, capacity, cleanup_end, hardware), 60)
        except BaseException as exc:
            errors.append('dynamic physical cleanup: ' + repr(exc))
        receipt['restoration'] = {}
""")
    source = once(source, "        receipt['measurement_valid'] = bool(failure is None and not errors\n",
                  "        receipt['measurement_valid'] = bool(failure is None and not errors and receipt.get('dynamic_inventory_verified') is True\n")
    source = source.rstrip()+'\n'
    ast.parse(source)
    out.write_text(source)
    manifest = dict(schema=1, original_source=dict(path=str(SOURCE), sha256=PIN),
        adapted_source=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        protocol_unchanged=True, eight_gpu_energy_unchanged=True,
        changes=['inherited real parent FLOCK', 'durable dynamic endpoint dispatch',
                 'require measured return to original two', 'verified owned-extra failure cleanup'],
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest

if __name__ == '__main__':
    print(build(ROOT/'dynamic_measurement.py'))

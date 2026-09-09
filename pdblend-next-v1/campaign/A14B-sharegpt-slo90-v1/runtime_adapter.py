"""Freeze the inherited serving implementation for the new ShareGPT SLO scope.

Preparation is CPU-only. This module never starts/stops engines or takes a GPU
lease. A caller must provide the actual engine binding and own the node lease
before using the returned executor's unchanged ``run_one`` implementation.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL = 'a14b-sharegpt-slo90-v1'
OLD_PROTOCOL = 'per-dataset-slo-five-system-fixed-window-v1'
LIFECYCLE = 'until_declared_complete_v1'
COMMON_PARENT = REPO / 'campaign/parallel-rate-20260908-v1/common/execution-until-complete-v1'
PDB_PARENT = REPO / 'campaign/parallel-rate-20260908-v1/hosts/14b-fixed-p4'
BASELINE_PARENT = REPO / 'releases/five-system100-A14B-baseline-v1-runtime'
PDB_RELEASE = REPO / 'campaign/parallel-rate-20260908-v1/A/p4-minimal/fixed-release-001/release.json'
SYSTEMS = ('pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve')
CELL = 'src/ecopadg/serving/cell.py'
TOPOLOGY = 'src/ecopadg/serving/topology.py'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def once(source, old, new):
    require(source.count(old) == 1, 'parent source shape changed: ' + old)
    return source.replace(old, new)


def transform_cell(source):
    """Change scope validation only; all controller/measurement bodies survive."""
    source = once(source, "FIXED_WINDOW_PROTOCOL='" + OLD_PROTOCOL + "'",
                  "FIXED_WINDOW_PROTOCOL='" + PROTOCOL + "'")
    source = once(source,
        "DATASET_SLOS={'alpaca':(1.,.1),'sharegpt':(5.,.15),'longbench':(15.,.2)}",
        "DATASET_SLOS={'sharegpt':(5.,.15)}")
    source = once(source, "    requests=trace.get('requests',[])",
        "    if (trace.get('model') != '14b'\n"
        "            or type(trace.get('rate_rps')) not in (int,float)\n"
        "            or not math.isfinite(trace['rate_rps']) or trace['rate_rps'] <= 0):\n"
        "        raise ValueError('only 14B ShareGPT with a positive finite rate is declared')\n"
        "    requests=trace.get('requests',[])")
    source = once(source, "scale not in (.5,1.,2.)", "scale not in (.5,2.)")
    source = once(source, "SLO scale must be one of 0.5, 1, 2", "SLO scale must be 0.5 or 2")
    ast.parse(source)
    return source


def transform_common(source):
    source = once(source, "PROTOCOL = '" + OLD_PROTOCOL + "'", "PROTOCOL = '" + PROTOCOL + "'")
    require('GLOBAL_DEADLINE = None' in source and LIFECYCLE in source,
            'until-complete parent required')
    source = once(source, '    for path, digest in binding[\'files\'].items():',
        "    require(binding.get('model') == '14b' and set(binding['configs']) == {'sharegpt'},\n"
        "            'only the declared 14B ShareGPT binding is allowed')\n"
        "    for path, digest in binding['files'].items():")
    # The caller uses run_one. Keeping the historical sweep would accidentally
    # reintroduce its main-reference requirement and independent old queue.
    source = once(source, 'async def sweep(args, binding):\n',
        "async def sweep(args, binding):\n"
        "    raise RuntimeError('use the new SLO90 queue and run_one; legacy sweep is disabled')\n")
    ast.parse(source)
    return source


def transform_topology(source):
    """Bind dynamic engine imports to the same prepared native source tree."""
    source = once(source, "'-e','PYTHONPATH=/root/workspace/pdblend/src'",
                  "'-e','PYTHONPATH='+self.template.get('observation_engine_pythonpath','/root/workspace/pdblend/src')")
    source = once(source, '        self.slots=asyncio.Semaphore(parallelism)',
        "        self.container_names={}\n"
        "        observed_prefix=self.template.get('observation_container_prefix')\n"
        "        if observed_prefix is not None:\n"
        "            if (not isinstance(observed_prefix,str) or not observed_prefix.startswith('slo90-')\n"
        "                    or not observed_prefix.endswith('-')\n"
        "                    or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in observed_prefix)):\n"
        "                raise ValueError('observation lifecycle needs an independent task container prefix')\n"
        "            names=self.template.get('observation_container_names')\n"
        "            if (not isinstance(names,dict) or not names or len(set(names.values()))!=len(names)\n"
        "                    or any(not isinstance(iid,str) or not iid or not isinstance(name,str)\n"
        "                        or not name.startswith(observed_prefix)\n"
        "                        or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name)\n"
        "                        for iid,name in names.items())):\n"
        "                raise ValueError('observed container mapping escapes this task prefix')\n"
        "            self.prefix=observed_prefix;self.container_names=dict(names)\n"
        "        elif self.template.get('observation_container_names'):\n"
        "            raise ValueError('observed container names require their task prefix')\n"
        "        self.slots=asyncio.Semaphore(parallelism)\n"
        "\n"
        "    def container_name(self,spec):\n"
        "        iid=spec.instance_id\n"
        "        if (not isinstance(iid,str) or not iid\n"
        "                or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in iid)):\n"
        "            raise ValueError('invalid lifecycle instance identity')\n"
        "        return self.container_names.get(iid,self.prefix+iid)")
    source = once(source, "'--name',self.prefix+spec.instance_id", "'--name',self.container_name(spec)")
    source = once(source, '        name=self.prefix+spec.instance_id', '        name=self.container_name(spec)')
    ast.parse(source)
    return source


def checked_manifest(directory):
    directory = Path(directory).resolve()
    manifest = read(directory / 'manifest.json')
    for name, digest in manifest['files'].items():
        path = Path(name) if Path(name).is_absolute() else directory / name
        require(sha(path) == digest, 'frozen runtime changed: ' + str(path))
    return manifest


def prepare_host(parent_host, out):
    parent_host, out = Path(parent_host).resolve(), Path(out).resolve()
    require(parent_host in (PDB_PARENT, BASELINE_PARENT), 'explicit qualified parent runtime required')
    require(not out.exists(), 'fresh runtime destination required')
    parent = checked_manifest(parent_host)
    out.mkdir(parents=True)
    for name in parent['files']:
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'relative runtime files required')
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == CELL:
            target.write_text(transform_cell((parent_host / name).read_text()))
        elif name == TOPOLOGY and parent_host == BASELINE_PARENT:
            target.write_text(transform_topology((parent_host / name).read_text()))
        else:
            shutil.copyfile(parent_host / name, target)
    manifest = dict(schema=1, protocol_id=PROTOCOL, parent=ref(parent_host / 'manifest.json'),
        parent_release=str(parent_host), changed_runtime_files=[CELL] + ([TOPOLOGY] if parent_host == BASELINE_PARENT else []),
        implementation_policy_and_measurement_unchanged=True,
        source_selection='latest passed ShareGPT p4; experimental p6 capacity and 2400 profiles are not qualified for ShareGPT',
        files={name: sha(out / name) for name in parent['files']})
    write(out / 'manifest.json', manifest)
    return manifest


def prepare_common(out, parent=COMMON_PARENT):
    parent, out = Path(parent).resolve(), Path(out).resolve()
    require(parent == COMMON_PARENT, 'declared until-complete executor required')
    checked_manifest(parent)
    require(not out.exists(), 'fresh common destination required')
    out.mkdir(parents=True)
    (out / 'run.py').write_text(transform_common((parent / 'run.py').read_text()))
    shutil.copyfile(parent / 'child.py', out / 'child.py')
    manifest = dict(schema=1, protocol_id=PROTOCOL, deadline_s=None, campaign_lifecycle=LIFECYCLE,
        parent=ref(parent / 'manifest.json'), unchanged_child=True,
        unchanged_single_cell_measurement=True, legacy_sweep_disabled=True,
        files={name: sha(out / name) for name in ('run.py', 'child.py')})
    write(out / 'manifest.json', manifest)
    return manifest


def prepare_config(parent_config, out, host_release=None):
    parent_config = Path(parent_config).resolve()
    config = read(parent_config)
    canonical = ('pdblend' if config['strategy'].startswith('pdblend') else
                 'dynamollm' if config['strategy'] == 'dynamollm-resident' else config['strategy'])
    require(canonical in SYSTEMS and config.get('evaluation_protocol') == 'evaluation-v3',
            'frozen five-system evaluation-v3 strategy required')
    require(config.get('measurement_window_protocol') == OLD_PROTOCOL and
            config.get('arrival_window_s') == 100 and
            config.get('request_timeout_s', 120 if canonical != 'pdblend' else None) == 120,
            'original 100/120-second protocol required')
    require(config.get('node_gpus') == list(range(8)), 'all eight GPUs must remain measured')
    require(not config.get('capacity_control'), 'unqualified dynamic capacity is not declared for ShareGPT')
    config['measurement_window_protocol'] = PROTOCOL
    # A row supplies its effective SLO to configure_fixed_window; retain the
    # inherited initial placeholder exactly, as all other policy parameters.
    write(out, config)
    return ref(out)


def stat_identity(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino, device=stat.st_dev)


def configuration_inputs(config):
    """Actual file-valued configuration inputs, excluding output journals."""
    found = set()
    def visit(value, key=None):
        if key in ('journal', 'capacity_inventory_path', 'capacity_job_path'):
            return
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and value.startswith('/') and Path(value).is_file():
            found.add(Path(value).resolve())
    visit(config)
    return found


def make_binding(base, *, host_release, config, common_dir, output, evidence=(),
                 frozen_inputs=(), large_input_paths=(), retain_parent_files=False):
    """Bind frozen host source to actual engines; never manufacture live identity."""
    base_path = Path(base).resolve() if isinstance(base, (str, Path)) else None
    binding = copy.deepcopy(read(base_path) if base_path else base)
    require(binding['model'] == '14b' and binding['system'] in SYSTEMS, '14B five-system base required')
    host_release, common_dir = Path(host_release).resolve(), Path(common_dir).resolve()
    host_manifest = checked_manifest(host_release)
    common_manifest = checked_manifest(common_dir)
    require(host_manifest['protocol_id'] == common_manifest['protocol_id'] == PROTOCOL, 'new runtime scope missing')
    config = Path(config['path'] if isinstance(config, dict) else config).resolve()
    cfg = read(config)
    require(cfg.get('measurement_window_protocol') == PROTOCOL, 'new configuration scope missing')
    expected_ids = [(i['id'], i['tp'], i['gpus'], i['url']) for i in binding['instances']]
    actual_ids = [(i['id'], i['tp'], i['gpus'], i['url']) for i in cfg['instances']]
    require(actual_ids == expected_ids, 'controller and actual bound engine layout differ')
    # Do not copy another host's inode/stat identities or its whole historical
    # campaign into the new freeze. Provenance is explicitly retained below;
    # optional large inputs are restatted on the actual execution host.
    files = copy.deepcopy(binding.get('files', {})) if retain_parent_files else {}
    binding['files'] = files
    binding['large_inputs'] = {str(Path(path).resolve()): dict(stat=stat_identity(path))
                               for path in large_input_paths}
    for directory, manifest in ((host_release, host_manifest), (common_dir, common_manifest)):
        files.update({str(directory / name): digest for name, digest in manifest['files'].items()})
        files[str(directory / 'manifest.json')] = sha(directory / 'manifest.json')
    required = configuration_inputs(cfg) | {Path(path).resolve() for path in frozen_inputs}
    for instance in binding['instances']:
        if instance.get('engine_config'):
            required.add(Path(instance['engine_config']).resolve())
        for path, digest in instance.get('provenance', {}).get('source_files_at_import', {}).items():
            require(sha(path) == digest, 'actual imported engine source changed: ' + path)
            required.add(Path(path).resolve())
    for path in (config, Path(__file__).resolve(), *required, *evidence,
                 *((base_path,) if base_path else ())):
        path = Path(path).resolve()
        files[str(path)] = sha(path)
    binding.update(protocol_id=PROTOCOL, experiment_scope=PROTOCOL, deadline_s=None,
        campaign_lifecycle=LIFECYCLE, host_release=str(host_release),
        configs={'sharegpt': str(config)}, output=str(Path(output).resolve()),
        executor=str(common_dir / 'run.py'), formal_eligible=False,
        baseline_or_pdb_policy_unchanged=True, main_reference_required=False)
    return binding


def load_runtime(host_release, common_dir):
    host_release, common_dir = Path(host_release).resolve(), Path(common_dir).resolve()
    checked_manifest(host_release)
    checked_manifest(common_dir)
    # Switching a loaded ecopadg module in-process would splice source versions.
    for name, module in tuple(sys.modules.items()):
        if name == 'ecopadg' or name.startswith('ecopadg.') or name.startswith('benchmarks.'):
            path = getattr(module, '__file__', None)
            if path:
                require(Path(path).resolve().is_relative_to(host_release),
                        'different serving runtime already imported; use a fresh process: ' + name)
    paths = [str(host_release / 'src'), str(host_release), '/root/workspace/pdblend/.runtime-deps']
    sys.path[:0] = paths
    os.environ['PYTHONPATH'] = ':'.join(paths)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    path = common_dir / 'run.py'
    spec = importlib.util.spec_from_file_location('slo90_common_' + sha(path)[:12], path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

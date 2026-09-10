"""Reuse an independently computed qualification only while every input is unchanged."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import contextlib
import re
import stat
import sys
import time
import types

HERE = Path(__file__).resolve().parent
WORKSPACE = Path('/root/workspace')


def need(value, why):
    if not value:
        raise ValueError(why)


def metadata(path):
    return stat_metadata(os.stat(path))


def stat_metadata(value):
    return {key: getattr(value, 'st_' + key) for key in ('dev', 'ino', 'mode', 'size', 'mtime_ns', 'ctime_ns')}


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked_bytes(reference):
    path = reference['path']; before = metadata(path)
    with open(path, 'rb') as stream:
        need(stat_metadata(os.fstat(stream.fileno())) == before, 'input replaced while opening: ' + path)
        data = stream.read()
        need(stat_metadata(os.fstat(stream.fileno())) == before, 'input changed while reading: ' + path)
    need(hashlib.sha256(data).hexdigest() == reference['sha256'] and metadata(path) == before,
         'qualification cache dependency changed: ' + path)
    return data


def checked(reference):
    return json.loads(checked_bytes(reference))


def load(reference, name):
    source = checked_bytes(reference)
    module = types.ModuleType(name)
    module.__file__ = reference['path']
    exec(compile(source, reference['path'], 'exec'), module.__dict__)
    return module


class ReadCapture:
    """Capture supported Python file, stat and directory reads in a fresh worker.

    Process creation, networking, ctypes and mmap escapes are rejected. This is
    a dependency tracker for trusted Python verifiers, not a native-code sandbox.
    """
    def __init__(self, workspace=WORKSPACE):
        self.workspace = str(Path(workspace).resolve()) + os.sep
        self.reads = set()
        self.stats = {}
        self.writes = set()
        self.failures = []
        self.volatile_terminal_observations = set()
        self.directory_identities = {}
        self.symlinks = {}
        self.directories = {}
        self.absent = set()
        self.active = False
        self.original_stat = os.stat
        self.original_path_stat = Path.stat
        self.original_lstat = os.lstat
        self.original_readlink = os.readlink

    def fail(self, why):
        self.failures.append(why)
        raise RuntimeError(why)

    def path(self, value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return None
        path = os.path.abspath(os.fsdecode(value))
        if path.startswith('/proc/'):
            if re.fullmatch(r'/proc/[0-9]+(?:/stat)?', path):
                if self.active: self.volatile_terminal_observations.add(path)
                return None
            if self.active: self.fail('unsupported volatile qualification read: ' + path)
            return None
        if path.startswith(('/sys/', '/dev/')):
            if self.active: self.fail('unsupported volatile qualification read: ' + path)
            return None
        return path

    def record(self, path, value):
        if stat.S_ISDIR(value.st_mode):
            current = {key: getattr(value, 'st_' + key) for key in ('dev', 'ino', 'mode')}
            if path in self.directory_identities and self.directory_identities[path] != current:
                self.fail('directory identity changed during full qualification: ' + path)
            self.directory_identities[path] = current
        elif stat.S_ISLNK(value.st_mode):
            current = dict(metadata=stat_metadata(value), target=self.original_readlink(path))
            if path in self.symlinks and self.symlinks[path] != current:
                self.fail('symbolic link changed during full qualification: ' + path)
            self.symlinks[path] = current
        elif stat.S_ISREG(value.st_mode):
            current = stat_metadata(value)
            need(path not in self.stats or self.stats[path] == current, 'input changed during full qualification: ' + path)
            self.stats[path] = current

    def stat(self, path, *args, **kwargs):
        need(not self.active or kwargs.get('dir_fd') is None, 'untracked relative stat descriptor')
        if self.active:
            canonical = self.path(path)
            if canonical:
                try:
                    entry = self.original_lstat(path)
                    if stat.S_ISLNK(entry.st_mode): self.record(canonical, entry)
                except FileNotFoundError:
                    self.absent.add(canonical)
        try:
            value = self.original_stat(path, *args, **kwargs)
        except FileNotFoundError:
            canonical = self.path(path)
            if self.active and canonical and kwargs.get('dir_fd') is None:
                self.absent.add(canonical)
            raise
        if self.active and kwargs.get('dir_fd') is None:
            canonical = self.path(path)
            if canonical:
                self.record(canonical, value)
        return value

    def readlink(self, path, *, dir_fd=None):
        need(not self.active or dir_fd is None, 'untracked relative readlink descriptor')
        result = self.original_readlink(path, dir_fd=dir_fd)
        canonical = self.path(path)
        if self.active and canonical:
            self.record(canonical, self.original_lstat(path))
            need(self.symlinks[canonical]['target'] == result, 'symbolic link changed while reading target')
        return result

    def hook(self, event, args):
        if not self.active:
            return
        if (event.startswith(('subprocess.', 'socket.', 'ctypes.', 'winreg.'))
                or event in ('os.system', 'os.exec', 'os.posix_spawn', 'os.fork',
                             'os.forkpty', 'os.spawn', 'pty.spawn', 'mmap.__new__')):
            self.fail('unsupported qualification input channel: ' + event)
        if event in ('os.remove', 'os.rename', 'os.rmdir', 'os.mkdir', 'os.link', 'os.symlink', 'os.truncate', 'os.chmod', 'os.chown', 'os.utime'):
            if any(self.path(arg) for arg in args if not isinstance(arg, int)):
                self.fail('qualification verifier must not modify workspace inputs: ' + event)
        if event in ('os.listdir', 'os.scandir'):
            need(not isinstance(args[0], int), 'untracked directory descriptor')
            path = self.path(args[0])
            if path:
                self.active = False
                try:
                    names = sorted(os.listdir(path))
                    self.record(path, self.original_stat(path))
                finally:
                    self.active = True
                need(path not in self.directories or self.directories[path] == names,
                     'directory changed during full qualification: ' + path)
                self.directories[path] = names
            return
        if event != 'open':
            return
        need(not isinstance(args[0], int), 'untracked file descriptor open')
        path = self.path(args[0])
        if not path:
            return
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(k in mode for k in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR)
        if writing:
            self.writes.add(path)
            self.fail('qualification verifier must not write files: ' + path)
        try:
            value = self.original_stat(path)
        except FileNotFoundError:
            self.absent.add(path)
            return
        except OSError:
            return
        if stat.S_ISREG(value.st_mode):
            self.reads.add(path)
            self.record(path, value)

    def __enter__(self):
        sys.addaudithook(self.hook)
        os.stat = self.stat
        os.lstat = lambda path, **kwargs: self.stat(path, follow_symlinks=False, **kwargs)
        os.readlink = self.readlink
        capture = self
        Path.stat = lambda path, **kwargs: capture.stat(path, **kwargs)
        self.active = True
        return self

    def __exit__(self, *args):
        self.active = False
        os.stat = self.original_stat
        os.lstat = self.original_lstat
        os.readlink = self.original_readlink
        Path.stat = self.original_path_stat
        if not args[0] and self.failures:
            raise ValueError('qualification verifier suppressed a forbidden input channel: ' + self.failures[0])


def declared_refs(value, base):
    if isinstance(value, list):
        for item in value:
            yield from declared_refs(item, base)
    elif isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            yield value
        for key in ('files', 'source_files', 'artifacts'):
            for name, digest in (value.get(key) or {}).items() if isinstance(value.get(key), dict) else []:
                if isinstance(digest, str) and len(digest) == 64:
                    path = Path(name)
                    yield dict(path=str(path if path.is_absolute() else base / path), sha256=digest)
        for key, item in value.items():
            if key not in ('files', 'source_files', 'artifacts'):
                yield from declared_refs(item, base)


def build(qualification, validator, out, *, workspace=WORKSPACE):
    """Run the original verifier in a new interpreter with bytecode writes off."""
    request = dict(qualification=qualification, validator=validator,
                   out=str(Path(out).resolve()), workspace=str(Path(workspace).resolve()))
    process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                              '--build-request', json.dumps(request)],
                             text=True, capture_output=True)
    need(process.returncode == 0, 'fresh qualification cache worker failed: ' + process.stderr.strip())
    return json.loads(process.stdout)


def build_worker(qualification, validator, out, *, workspace=WORKSPACE):
    need(sys.dont_write_bytecode, 'qualification cache worker requires python -B')
    own_source = Path(__file__).resolve()
    for module in tuple(sys.modules.values()):
        source = getattr(module, '__file__', None)
        if source:
            source = Path(source).resolve()
            need(not source.is_relative_to(Path(workspace).resolve()) or source == own_source,
                 'workspace module preloaded before qualification: ' + str(source))
    out = Path(out).resolve()
    need(not out.exists(), 'fresh immutable qualification cache directory required')
    started = time.time()
    helper_metadata = metadata(own_source)
    with ReadCapture(workspace) as capture:
        qualified = checked(qualification)
        module = load(validator, 'captured_full_qualification')
        proof = module.verify(qualification)
        need(proof.get('passed') is True and proof.get('independently_recomputed') is True,
             'full independent qualification did not pass')
        binding = checked(proof['binding'])
    need(not capture.writes.intersection(capture.reads), 'full verifier reads its own mutable output; cache needs immutable inputs')
    inputs = set(capture.reads)
    inputs.update((qualification['path'], validator['path'], str(Path(__file__).resolve()), str(Path(sys.executable).resolve())))
    for imported in tuple(sys.modules.values()):
        path = getattr(imported, '__file__', None)
        canonical = capture.path(path)
        if canonical and Path(canonical).is_file():
            inputs.add(canonical)
            if canonical.endswith('.pyc'):
                try:
                    source = importlib.util.source_from_cache(canonical)
                    if Path(source).is_file():
                        inputs.add(source)
                except ValueError:
                    pass
    declared = list(declared_refs(qualified, Path(qualification['path']).parent))
    declared += list(declared_refs(binding, Path(proof['binding']['path']).parent))
    for reference in declared:
        need(sha(reference['path']) == reference['sha256'], 'declared qualification input changed: ' + reference['path'])
        inputs.add(reference['path'])
    files, file_metadata = {}, {}
    for path in sorted(inputs):
        before = metadata(path)
        if path in capture.stats:
            need(before == capture.stats[path], 'input changed after full qualification: ' + path)
        if Path(path) == own_source:
            need(before == helper_metadata, 'cache helper changed during full qualification')
        digest = sha(path)
        need(metadata(path) == before, 'input changed while freezing qualification: ' + path)
        files[path] = digest
        file_metadata[path] = before
    stat_only = {path: expected for path, expected in capture.stats.items() if path not in files}
    for path, expected in stat_only.items():
        need(metadata(path) == expected, 'stat-only qualification input changed: ' + path)
    for path, names in capture.directories.items():
        need(sorted(os.listdir(path)) == names, 'qualification directory changed after full audit: ' + path)
        need(not out.is_relative_to(path), 'cache destination must be outside scanned qualification directories')
    for path in capture.absent:
        need(not os.path.exists(path), 'previously absent qualification input appeared: ' + path)
    verify_path_kinds(capture.directory_identities, capture.symlinks)
    value = dict(schema='immutable-independently-recomputed-qualification-cache-v2',
        hostname=socket.gethostname(), python_version=sys.version, qualification=qualification,
        original_validator=validator, proof=proof, files=files, stat_only_inputs=stat_only,
        file_metadata=file_metadata, directory_membership=capture.directories,
        directory_identities=capture.directory_identities, symbolic_links=capture.symlinks,
        absent_inputs=sorted(capture.absent),
        volatile_terminal_observations=sorted(capture.volatile_terminal_observations),
        volatile_terminal_semantics='Only /proc/<numeric-pid>/stat observations are omitted from static file hashes; original terminal proof requires the recorded pid/startticks owner to have exited, a monotonic identity condition. Executor live engine/GPU checks are never cached.',
        started_s=started, finished_s=time.time(), full_verifier_elapsed_s=time.time()-started,
        full_validator_ran=True, evidence_semantics='Original full independent proof retained; all read/source SHA and stat-only inputs rechecked on every reuse; live engine/GPU checks remain in the executor.')
    out.mkdir(parents=True)
    path = out / 'cache.json'
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    cache_ref, helper_ref = ref(path), ref(__file__)
    wrapper = out / 'verify.py'
    wrapper.write_text('"""Frozen qualification cache; live executor identity checks remain mandatory."""\n'
        'import hashlib, types\n'
        f'CACHE = {cache_ref!r}\nHELPER = {helper_ref!r}\n'
        'def verify(reference):\n'
        '    with open(HELPER["path"], "rb") as stream:\n'
        '        source = stream.read()\n'
        '    if hashlib.sha256(source).hexdigest() != HELPER["sha256"]: raise ValueError("cache verifier changed")\n'
        '    module = types.ModuleType("immutable_qualification_cache"); module.__file__ = HELPER["path"]\n'
        '    exec(compile(source, HELPER["path"], "exec"), module.__dict__)\n'
        '    return module.verify_cached(reference, CACHE)\n')
    return dict(qualification=qualification, qualification_validator=ref(wrapper), cache=cache_ref,
                helper=helper_ref, files=len(files), stat_only_inputs=len(stat_only), elapsed_s=time.time()-started)


def verify_path_kinds(directories, symlinks):
    for path, expected in directories.items():
        current = os.stat(path)
        need({key: getattr(current, 'st_' + key) for key in ('dev', 'ino', 'mode')} == expected,
             'cached qualification directory identity changed: ' + path)
    for path, expected in symlinks.items():
        current = os.lstat(path)
        need(stat_metadata(current) == expected['metadata'] and os.readlink(path) == expected['target'],
             'cached qualification symbolic link changed: ' + path)


def verify_cached(reference, cache_reference):
    value = checked(cache_reference)
    need(value['schema'] == 'immutable-independently-recomputed-qualification-cache-v2'
         and value['full_validator_ran'] is True, 'unrecognized qualification cache')
    need(reference == value['qualification'], 'cache belongs to another qualification')
    need(socket.gethostname() == value['hostname'] and sys.version == value['python_version'],
         'qualification cache belongs to another host or Python runtime')
    for path, digest in value['files'].items():
        before = metadata(path)
        need(before == value['file_metadata'][path], 'cached qualification file identity changed: ' + path)
        need(sha(path) == digest and metadata(path) == before, 'cached qualification dependency changed: ' + path)
    for path, expected in value['stat_only_inputs'].items():
        need(metadata(path) == expected, 'cached qualification stat-only input changed: ' + path)
    for path, names in value['directory_membership'].items():
        need(sorted(os.listdir(path)) == names, 'cached qualification directory membership changed: ' + path)
    for path in value['absent_inputs']:
        need(not os.path.exists(path), 'cached qualification previously absent input appeared: ' + path)
    # A late dependency replacement during a different file's SHA scan must
    # invalidate this verification, including replacements with identical bytes.
    for path, expected in {**value['file_metadata'], **value['stat_only_inputs']}.items():
        need(metadata(path) == expected, 'cached qualification final metadata barrier changed: ' + path)
    verify_path_kinds(value['directory_identities'], value['symbolic_links'])
    proof = copy.deepcopy(value['proof'])
    need(proof.get('passed') is True and proof.get('independently_recomputed') is True,
         'original qualification proof did not pass')
    helper_path = str(Path(__file__).resolve())
    need(helper_path in value['files'], 'cache helper absent from frozen closure')
    proof.setdefault('files', {}).update({cache_reference['path']: cache_reference['sha256'],
                                        helper_path: value['files'][helper_path]})
    proof['qualification_cache'] = cache_reference
    proof['qualification_cache_helper'] = dict(path=helper_path, sha256=value['files'][helper_path])
    proof['full_independent_audit_reused'] = True
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-request', help=argparse.SUPPRESS)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--validator', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.build_request:
        with contextlib.redirect_stdout(sys.stderr):
            result = build_worker(**json.loads(args.build_request))
    else:
        parser.error('--qualification, --validator and --out are required') if not all((args.qualification, args.validator, args.out)) else None
        result = build(ref(args.qualification), ref(args.validator), args.out)
    print(json.dumps(result))

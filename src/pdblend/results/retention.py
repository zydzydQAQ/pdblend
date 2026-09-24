"""Export identities before pruning explicitly retired evidence.

Candidates are explicit paths, never age/status rules. Current non-candidate
metadata, source and process references conservatively protect the dependency
closure. A CSV tombstone with a full checksum precedes each unlink.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .catalog import collect, write_csv, sha, FIELDS

TEXT = {'.py', '.sh', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.csv', '.txt'}
PATH = re.compile(r'(?:/home/pdblend4/)?(?:results|datasets|scripts)/[^\s\"\'<>`,;(){}\[\]]+')


def regular(path):
    return path.is_file() and not path.is_symlink() and not any(p.is_symlink() for p in path.parents)


def identity(path):
    s = path.stat()
    return dict(size=s.st_size, inode=s.st_ino, device=s.st_dev, mtime_ns=s.st_mtime_ns)


def below(path, roots):
    return any(path == r or path.is_relative_to(r) for r in roots)


def own_ancestors():
    pids=set(); pid=os.getpid()
    while pid > 0 and pid not in pids:
        pids.add(pid)
        try:
            status=(Path('/proc')/str(pid)/'status').read_text()
            pid=int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
        except (OSError, StopIteration): break
    return pids


def open_inodes():
    opened=set()
    for proc in Path('/proc').glob('[0-9]*'):
        try: fds=list((proc/'fd').iterdir())
        except OSError: continue
        for fd in fds:
            try:
                st=fd.stat(); opened.add((st.st_dev,st.st_ino))
            except OSError: pass
    return opened


def references(path, root):
    """Stream large logs; resolve explicit paths and JSON-relative identities."""
    found = set()
    try:
        with path.open(errors='replace') as stream:
            for line in stream:
                if not any(marker in line for marker in ('results/', 'datasets/', 'scripts/')):
                    continue
                for name in PATH.findall(line):
                    p = Path(name.rstrip('.:'))
                    p = p if p.is_absolute() else root / p
                    if p.exists():
                        found.add(p.resolve())
        if path.suffix == '.json' and path.stat().st_size < 16 * 1024 * 1024:
            def walk(value):
                if isinstance(value, dict):
                    for k, v in value.items():
                        yield from walk(k); yield from walk(v)
                elif isinstance(value, list):
                    for v in value: yield from walk(v)
                elif isinstance(value, str): yield value
            for value in walk(json.loads(path.read_text())):
                if not value or len(value) > 4096 or '\n' in value or '{' in value:
                    continue
                if '/' not in value and Path(value).suffix not in TEXT | {'.pt', '.bin', '.pkl', '.safetensors'}:
                    continue
                # Docker bind mounts retain the host side, including directories.
                name = value.split(':', 1)[0] if value.startswith('/') else value
                for base in (path.parent, root):
                    try:
                        p = base / name
                        if p.exists(): found.add(p.resolve())
                    except OSError: pass
    except (OSError, ValueError):
        pass  # actively written JSON still has streamed explicit-path references
    return found


def dependency_closure(root, candidates, audit, active_roots=None):
    roots = ([Path(p) for p in active_roots] if active_roots else
             [root / p for p in ('src', 'tests', 'scripts', 'datasets', 'results')])
    exclude = [audit, root/'results/maintenance', root/'results/archive', root/'results/runs.csv', root/'results/profile_points.csv']
    scanned = set(); todo = []
    for entry in roots:
        for p in entry.rglob('*'):
            if regular(p) and p.suffix in TEXT and not below(p, candidates+exclude):
                todo.append(p)
    # Narrative historical mentions are not executable/data dependencies.
    # Explicitly referenced documents are retained by their referring manifest.
    protected = {}; scanned_bytes = 0
    while todo:
        p = todo.pop()
        if p in scanned: continue
        scanned.add(p); scanned_bytes += p.stat().st_size
        for ref in references(p, root):
            if not below(ref, candidates):
                if ref.is_relative_to(root) and not below(ref, exclude):
                    if regular(ref) and ref.suffix in TEXT:
                        todo.append(ref)
                    elif ref.is_dir() and ref != root:
                        # An output namespace such as results/ is not a live
                        # dependency on every retired sibling. Explicit active
                        # roots enumerate its current branches. Direct inputs
                        # inside a candidate are handled below and stay protected.
                        if active_roots and any(c.is_relative_to(ref) for c in candidates):
                            continue
                        todo.extend(x for x in ref.rglob('*') if regular(x) and x.suffix in TEXT
                                    and not below(x, candidates+exclude))
                continue
            members = [x for x in ref.rglob('*') if regular(x)] if ref.is_dir() else [ref]
            for member in members:
                if member in protected: continue
                protected[member] = str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
                if member.suffix in TEXT: todo.append(member)
    # Open fds and command-line inputs are roots even when there is no manifest.
    self_pids=own_ancestors()
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    target = fd.resolve()
                    if below(target, candidates): protected[target] = str(fd)
                except OSError: pass
            if int(proc.name) in self_pids: continue
            command = (proc/'cmdline').read_bytes().decode(errors='replace')
            for name in PATH.findall(command.replace('\0', ' ')):
                target = Path(name); target = target if target.is_absolute() else root/target
                if target.exists() and below(target, candidates):
                    for item in ([x for x in target.rglob('*') if regular(x)] if target.is_dir() else [target]):
                        protected[item] = str(proc/'cmdline')
        except (OSError, ProcessLookupError): pass
    return protected, dict(scanned_files=len(scanned), scanned_bytes=scanned_bytes)


def prepare(root, candidates, output, active_roots=None):
    root = root.resolve(); output = output.resolve()
    candidates = [p.resolve() for p in candidates]
    if any(not p.is_relative_to(root) or p == root or '.git' in p.parts for p in candidates):
        raise ValueError('candidate must be an explicit workspace data/script path')
    output.mkdir(parents=True, exist_ok=False)
    rows, errors = collect(root/'results')
    if errors: raise ValueError(f'CSV export errors: {errors}')
    write_csv(root/'results/runs.csv', rows)
    historical = [r for r in rows if below(Path(r['artifact_path']), candidates)]
    write_csv(output/'historical_runs.csv', historical)
    profile_csv=root/'results/profile_points.csv'
    if profile_csv.exists():
        with profile_csv.open(newline='') as stream:
            reader=csv.DictReader(stream); fields=reader.fieldnames
            old_points=[r for r in reader if r.get('raw_path') and below(Path(r['raw_path']),candidates)]
        write_csv(output/'historical_profile_points.csv',old_points,fields)
    protected, scan = dependency_closure(root, candidates, output, active_roots)
    entries = []
    for c in candidates:
        for p in sorted(c.rglob('*') if c.is_dir() else [c]):
            if not regular(p): continue
            entries.append(dict(path=str(p.relative_to(root)), experiment=str(c.relative_to(root)),
                **identity(p), sha256=sha(p), replacement=str(output/'historical_runs.csv'),
                reference_check=protected.get(p, 'no_current_reference'),
                decision='retain_csv' if p.suffix == '.csv' else ('retain_dependency' if p in protected else 'delete')))
    write_csv(output/'identities.csv', entries, ('path','experiment','size','inode','device','mtime_ns','sha256','replacement','reference_check','decision'))
    plan=dict(schema='historical-retention-v1', root=str(root), candidates=[str(p) for p in candidates],
        output=str(output), status='prepared', prepared_at=time.time(), dependency_scan=scan,
        dependency_roots=[str(Path(p).resolve()) for p in active_roots] if active_roots else None,
        runs_sha256=sha(root/'results/runs.csv'), historical_rows=len(historical),
        entries=entries, candidate_bytes=sum(e['size'] for e in entries if e['decision']=='delete'))
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    return plan


def apply(plan_path):
    plan_path=Path(plan_path); plan=json.loads(plan_path.read_text())
    if plan['status'] != 'prepared': raise ValueError('plan already applied')
    root=Path(plan['root']); output=Path(plan['output']); candidates=list(map(Path, plan['candidates']))
    # Rescan roots immediately before applying: new jobs may have been prepared.
    protected, scan=dependency_closure(root,candidates,output,plan.get('dependency_roots'))
    if sha(root/'results/runs.csv') != plan['runs_sha256']:
        raise ValueError('CSV changed after prepare; prepare a fresh deletion plan')
    entries=plan['entries']; ready=[]
    for e in entries:
        if e['decision'] != 'delete': continue
        p=root/e['path']
        if p in protected: e['skip_reason']='new_current_reference'; continue
        if not regular(p) or identity(p) != {k:e[k] for k in ('size','inode','device','mtime_ns')}:
            e['skip_reason']='file_changed'; continue
        if sha(p) != e['sha256']: e['skip_reason']='checksum_changed'; continue
        ready.append(e)
    opened=open_inodes()
    for e in ready:
        if (e['device'],e['inode']) in opened: e['skip_reason']='open_fd'
    ready=[e for e in ready if not e.get('skip_reason')]
    # Mark affected runs before deleting, including partial raw pruning.
    with (root/'results/runs.csv').open(newline='') as stream: rows=list(csv.DictReader(stream))
    affected={root/e['path'] for e in ready}
    # Test each row's containing directory once, rather than comparing every
    # profile point with every deleted file (millions of Path constructions).
    affected_directories={parent for path in affected for parent in path.parents}
    for row in rows:
        if Path(row['artifact_path']).parent in affected_directories:
            row.update(evidence_status='raw_pruned', formal_eligible=False, purpose='historical',
                       retention_manifest=str(plan_path.resolve()))
    write_csv(root/'results/runs.csv', rows)
    write_csv(output/'historical_runs.csv',[r for r in rows if below(Path(r['artifact_path']),candidates)])
    profile_csv=root/'results/profile_points.csv'
    if profile_csv.exists():
        with profile_csv.open(newline='') as stream:
            reader=csv.DictReader(stream); fields=reader.fieldnames; points=list(reader)
        for row in points:
            if row.get('raw_path') and Path(row['raw_path']).parent in affected_directories:
                row.update(status='raw_pruned',formal_eligible=False)
        write_csv(profile_csv,points,fields)
        write_csv(output/'historical_profile_points.csv',
                  [r for r in points if r.get('raw_path') and below(Path(r['raw_path']),candidates)],fields)
    plan.update(status='applying', dependency_recheck=scan)
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    deleted=0; allocated=0; count=0
    for e in ready:
        p=root/e['path']
        if not regular(p) or identity(p) != {k:e[k] for k in ('size','inode','device','mtime_ns')}:
            e['skip_reason']='changed_at_delete'; continue
        # O_NOFOLLOW plus matching inode prevents unlinking a substituted target.
        fd=os.open(p, os.O_RDONLY|os.O_NOFOLLOW)
        try:
            st=os.fstat(fd)
            if (st.st_ino,st.st_dev,st.st_mtime_ns)!=(e['inode'],e['device'],e['mtime_ns']):
                e['skip_reason']='changed_at_open'; continue
            blocks=st.st_blocks*512 if st.st_nlink==1 else 0
            p.unlink(); e['deleted']=True; deleted+=e['size']; allocated+=blocks; count+=1
        finally: os.close(fd)
    plan.update(status='applied',finished_at=time.time(),deleted_files=count,deleted_bytes=deleted,
        freed_allocated_bytes=allocated,retained_files=sum(not e.get('deleted') for e in entries))
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    write_csv(output/'identities.csv',entries,('path','experiment','size','inode','device','mtime_ns','sha256','replacement','reference_check','decision','deleted','skip_reason'))
    return plan


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare');a.add_argument('--root',type=Path,default=Path('.'));a.add_argument('--output',type=Path,required=True);a.add_argument('--candidate',type=Path,action='append',required=True);a.add_argument('--active-root',type=Path,action='append')
    a=s.add_parser('apply');a.add_argument('plan',type=Path)
    args=p.parse_args(argv)
    d=prepare(args.root,args.candidate,args.output,args.active_root) if args.action=='prepare' else apply(args.plan)
    print(json.dumps({k:v for k,v in d.items() if k!='entries'}))

if __name__=='__main__':main()

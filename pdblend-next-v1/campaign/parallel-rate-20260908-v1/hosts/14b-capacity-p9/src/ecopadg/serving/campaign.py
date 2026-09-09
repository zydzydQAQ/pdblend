"""Exclusive-node, deadline-bounded campaign runner with prerequisite gates."""
import argparse
import hashlib
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .evidence import baseline_gaps, formal_freeze_gaps
from .budget import read_budget


@contextmanager
def node_lease():
    """Share a campaign's inherited lease, or exclusively acquire the node."""
    path=Path('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock')
    path.parent.mkdir(parents=True,exist_ok=True)
    inherited=os.environ.get('PDBLEND_NODE_LOCK_FD')
    with os.fdopen(os.dup(int(inherited)),'a') if inherited else path.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield handle


class Campaign:
    def __init__(self,root,limit_s=86400):
        import math
        if type(limit_s) not in (int,float) or not math.isfinite(limit_s) or limit_s<=0:
            raise ValueError('finite positive campaign budget required')
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock=open(self.root.parent/'node-experiment.lock','a')
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.state_path=self.root/'budget.json'
            self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else dict(
                limit_s=limit_s,started_s=None,consumed_s=0,formal_started=False)
            effective=read_budget(self.root,state=self.state)
            if effective['revision_seq']:
                if limit_s>effective['limit_s']:
                    raise ValueError('manifest budget exceeds the authorized ledger')
                # Historical 24-hour manifests are execution descriptions, not
                # permission to undo a later explicit user-authorized revision.
                self.state=effective
            else:
                if limit_s>86400:raise ValueError('first-round budget is at most 24 hours without authorization')
                self.state['limit_s']=min(self.state.get('limit_s',limit_s),limit_s)
                self.state=read_budget(self.root,state=self.state)
            self.active=None
            self.outcomes=[]
        except BaseException:
            self.lock.close()
            raise

    def persist(self):
        now=time.time()
        self.state=read_budget(self.root,state=self.state,now=now)
        elapsed=self.state['elapsed_s']
        self.state.update(updated_s=now,elapsed_s=elapsed,
                          remaining_s=max(0,self.state['limit_s']-elapsed))
        temp=self.state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.state,indent=2))
        temp.replace(self.state_path)

    @property
    def remaining_s(self):
        # Keep dict identity: callers may be evaluating self.state.update(...,
        # remaining_s=self.remaining_s) before committing their stage outcome.
        self.state.update(read_budget(self.root,state=self.state))
        return self.state['remaining_s']

    def formal_gate(self,mechanisms,freeze):
        missing=baseline_gaps(mechanisms)
        changed=formal_freeze_gaps(freeze)
        if missing or changed or not freeze:
            raise RuntimeError('formal gate closed: '+json.dumps(dict(missing=missing,changed=changed)))
        self.state['formal_started']=True
        self.persist()

    def run(self,name,argv,stage_limit_s,*,gpu=True):
        if not argv or not all(isinstance(s,str) for s in argv):
            raise ValueError('commands must be argv arrays, never shell snippets')
        if gpu and self.state.get('started_s') is None:
            self.state['started_s']=time.time()
        # Reserve the final minute for stopping this experiment's engines and
        # restoring hardware clocks even if its last child hits the deadline.
        # CPU stages inside an active node lease still leave resident engines
        # allocated. They must not postpone the node's absolute cleanup time.
        timeout=(min(float(stage_limit_s),self.remaining_s-60)
                 if self.state.get('started_s') else float(stage_limit_s))
        if timeout<=0:
            raise TimeoutError('node budget exhausted')
        started=time.time()
        self.state.update(stage=name,last_started_s=started,last_exit_code=None,
                          last_error=None,last_interrupted=False)
        self.persist()
        code=None;error=None;interrupted=False
        try:
            with (self.root/(name+'.log')).open('wb') as log:
                environment=dict(os.environ,PDBLEND_NODE_LOCK_FD=str(self.lock.fileno()))
                try:
                    self.active=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                        env=environment,pass_fds=(self.lock.fileno(),))
                    code=self.active.wait(timeout=timeout)
                except BaseException:
                    if self.active is not None:
                        if self.active.poll() is None:
                            try:os.killpg(self.active.pid,signal.SIGTERM)
                            except ProcessLookupError:pass
                        try:code=self.active.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(self.active.pid,signal.SIGKILL)
                            code=self.active.wait()
                    raise
                if code:
                    raise RuntimeError(f'{name} failed with exit code {code}')
        except BaseException as exc:
            error=repr(exc)
            interrupted=isinstance(exc,(KeyboardInterrupt,TimeoutError,subprocess.TimeoutExpired))
            raise
        finally:
            if self.active is not None and getattr(self.active,'returncode',None) is not None:
                code=self.active.returncode
            self.active=None
            finished=time.time()
            self.state.update(last_finished_s=finished,last_elapsed_s=finished-started,
                last_exit_code=code,last_error=error,last_interrupted=interrupted,remaining_s=self.remaining_s)
            self.outcomes.append(dict(name=name,argv=argv,started_s=started,finished_s=finished,
                declared_limit_s=float(stage_limit_s),effective_timeout_s=timeout,exit_code=code,
                error=error,interrupted=interrupted,budget_revision_seq=self.state['revision_seq'],
                effective_deadline_s=self.state['deadline_s']))
            self.persist()

    def close(self):
        try:
            if self.state.get('started_s') and self.remaining_s<=60:
                self.restore_node_at_deadline()
            self.persist()
        finally:self.lock.close()

    def restore_node_at_deadline(self):
        """Only this experiment's containers, under the exclusive node lease."""
        budget=read_budget(self.root,state=getattr(self,'state',None))
        result=dict(at_s=time.time(),reason=('authorized campaign budget deadline' if budget['revision_seq']
            else '24-hour campaign deadline'),stopped=[],errors=[],
            original_deadline_s=budget['original_deadline_s'],effective_deadline_s=budget['deadline_s'],
            budget_revision_seq=budget['revision_seq'],budget_revision_sha256=budget['revision_sha256'],
            authorization_sha256=budget['authorization_sha256'])
        try:
            listed=subprocess.run(['docker','ps','-a','--filter','name=^pdb-v2-',
                '--format','{{.Names}}'],capture_output=True,text=True,check=True,timeout=5)
            names=[n for n in listed.stdout.splitlines() if n.startswith('pdb-v2-')
                   and all(c.isalnum() or c in '-_' for c in n)]
            if names:
                subprocess.run(['docker','rm','-f',*names],capture_output=True,check=True,timeout=35)
                result['stopped']=names
        except Exception as exc: result['errors'].append(repr(exc))
        try:
            from ecopadg.measure.backends import PynvmlBackend
            backend=PynvmlBackend()
            for gpu in range(8):
                try: backend.reset_clock(gpu)
                except Exception as exc: result['errors'].append(f'GPU {gpu}: {exc!r}')
        except Exception as exc: result['errors'].append(repr(exc))
        (self.root/'deadline_cleanup.json').write_text(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    args=parser.parse_args()
    manifest=Path(args.manifest).resolve()
    receipt=manifest.with_suffix('.execution-result.json')
    if receipt.exists():raise ValueError('campaign already has an actual execution receipt; use a new manifest')
    manifest_bytes=manifest.read_bytes()
    manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest()
    spec=json.loads(manifest_bytes)
    campaign=None;passed=set();started=time.time();exit_code=1;error=None
    def terminate(signum,_frame):
        raise KeyboardInterrupt('campaign received signal '+str(signum))
    old_term=signal.signal(signal.SIGTERM,terminate)
    try:
        campaign=Campaign(spec['output'],spec.get('budget_s',86400))
        for stage in spec['stages']:
            if not set(stage.get('requires',())).issubset(passed):
                raise RuntimeError('unmet stage prerequisites: '+stage['name'])
            if stage.get('formal'):
                campaign.formal_gate(json.loads(Path(spec['mechanisms']).read_text()),
                                     json.loads(Path(spec['freeze']).read_text()))
            campaign.run(stage['name'],stage['argv'],stage['limit_s'],gpu=stage.get('gpu',True))
            passed.add(stage['name'])
        exit_code=0
    except BaseException as exc:
        error=repr(exc);exit_code=130 if isinstance(exc,KeyboardInterrupt) else 1
    finally:
        try:
            if campaign is not None:campaign.close()
        except BaseException as exc:
            error=(error+'; ' if error else '')+'close: '+repr(exc);exit_code=1
        finally:
            signal.signal(signal.SIGTERM,old_term)
            result=dict(manifest=str(manifest),manifest_sha256=manifest_sha256,started_s=started,
                finished_s=time.time(),exit_code=exit_code,error=error,
                complete=exit_code==0,stage_outcomes=campaign.outcomes if campaign else [],
                passed_stages=sorted(passed),formal_eligible=False)
            with receipt.open('x') as stream:json.dump(result,stream,indent=2)
    return exit_code


if __name__=='__main__':
    raise SystemExit(main())

"""Private profile startup port reservations. No GPU, policy or measurement changes.

All cohort members reserve their complete static listening-port set before any
model is loaded. Reservations are handed to an engine only under the existing
model-load lock. This is cooperative coordination, not FD inheritance: an
unrelated process can still race the brief release-to-engine-bind interval.
Such a race fails with raw socket/PID evidence; no unrelated process is killed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


CONTRACT='distserve-complete-static-port-reservation/v1'


def need(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+str(os.getpid()))
    try:
        with temporary.open('x') as stream:
            json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def inventory(specs):
    """Pinned NativeSpec and fixed P2P image: HTTP + kv_port + every TP rank."""
    rows=[]
    for spec in specs:
        need(spec.pp==1 and spec.tp in (1,2) and spec.kv_connector=='P2pNcclConnector',
             'port contract only covers this PP1 native P2P stage pair')
        command=spec.command()
        need(command.count('--host')==1 and command.count('--port')==1
             and command[command.index('--host')+1] in ('0.0.0.0','127.0.0.1')
             and int(command[command.index('--port')+1])==spec.port,
             'actual HTTP launch address/port differs from NativeSpec')
        need(spec.side_channel_port is None,'unreviewed side-channel listener is not covered')
        kv=json.loads(command[command.index('--kv-transfer-config')+1])
        need(kv['kv_connector']=='P2pNcclConnector' and int(kv['kv_port'])==int(spec.zmq_address.rsplit(':',1)[1])
             and spec.environment().get('VLLM_HOST_IP')=='127.0.0.1'
             and str(kv['kv_connector_extra_config']['http_port'])==str(spec.port)
             and not kv['kv_connector_extra_config'].get('proxy_port'),
             'actual P2P/control/proxy launch differs from reviewed port contract')
        for kind,rank,port,address in [('http',None,spec.port,command[command.index('--host')+1]),
            *[('p2p_rank',rank,int(kv['kv_port'])+rank,'127.0.0.1') for rank in range(spec.tp)]]:
            need(type(port) is int and 0<port<65536,'invalid listening port')
            rows.append(dict(instance_id=spec.instance_id,kind=kind,rank=rank,port=port,
                engine_bind_address=address,reservation_bind_address='0.0.0.0'))
    need(len({r['port'] for r in rows})==len(rows),'actual instance/TP-rank listening ports overlap')
    return rows


def socket_snapshot(ports,*,proc_root='/proc'):
    """Raw host-network sockets plus inode-to-visible-process joins, never PID inference."""
    ports=set(ports);proc=Path(proc_root);observed=[];errors=[]
    for family in ('tcp','tcp6'):
        try:
            for line in (proc/'net'/family).read_text().splitlines()[1:]:
                fields=line.split();port=int(fields[1].split(':')[1],16)
                if port in ports:observed.append(dict(family=family,local_address=fields[1],
                    port=port,state_hex=fields[3],inode=fields[9],raw=line))
        except (OSError,ValueError,IndexError) as error:errors.append(family+': '+repr(error))
    inodes={r['inode'] for r in observed if r['inode']!='0'};processes=[]
    if inodes:
        for path in proc.iterdir():
            if not path.name.isdigit():continue
            try:
                matched=[]
                for fd in (path/'fd').iterdir():
                    try:target=os.readlink(fd)
                    except OSError:continue
                    if target.startswith('socket:[') and target[8:-1] in inodes:
                        matched.append(dict(fd=fd.name,inode=target[8:-1]))
                if matched:
                    processes.append(dict(pid=int(path.name),socket_fds=matched,
                        stat=(path/'stat').read_text(),status=(path/'status').read_text(),
                        pid_namespace=os.readlink(path/'ns/pid'),
                        cmdline=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')))
            except (OSError,ValueError):continue  # Process exit races are not proof of absence.
    ss=dict(available=bool(shutil.which('ss')),observed_in_current_pid_namespace=True)
    if ss['available']:
        command=['ss','-ltnpe','( '+' or '.join('sport = :'+str(port) for port in sorted(ports))+' )']
        result=subprocess.run(command,text=True,capture_output=True,timeout=5)
        ss['command']=command
        ss.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    return dict(at_s=time.time(),ports=sorted(ports),proc_root=str(proc),sockets=observed,
        processes=processes,proc_errors=errors,ss=ss,
        pid_scope='host' if str(proc)=='/host/proc' else 'observer_namespace',
        absence_of_visible_pid_is_not_socket_cleanup_proof=True)


def lease_binding(environment_path,*,member,cohort_id,members):
    environment_path=Path(environment_path);raw=environment_path.read_bytes();env=json.loads(raw)
    manifest=environment_path.parent/env['lease_manifest_file'];data=manifest.read_bytes();lease=json.loads(data)
    need(hashlib.sha256(data).hexdigest()==env['lease_manifest_sha256'], 'real queue lease manifest bytes differ')
    payload=lease['payload']
    need(lease.get('immutable') is True and lease['lease_id']==env['lease_id']
         and lease['gpu_uuids']==env['allocated_gpu_uuids']
         and payload.get('sampling_cohort')==cohort_id and payload.get('cohort_member')==member
         and len(lease['gpu_uuids'])==payload['gpu_count'] and member in members,
         'real lease/member/cohort/physical inventory differs')
    return dict(lease_id=lease['lease_id'],job_id=lease['job_id'],gpu_uuids=lease['gpu_uuids'],
        expected_members=list(members),member=member,cohort_id=cohort_id,
        lease_manifest=dict(path=str(manifest),sha256=hashlib.sha256(data).hexdigest()),
        environment=dict(path=str(environment_path),sha256=hashlib.sha256(raw).hexdigest()))


def validate_registrations(rows,members,cohort_id):
    need({r['lease']['member'] for r in rows}==set(members) and len(rows)==len(members),'exact expected port members required')
    ports=[];uuids=[];leases=[]
    for row in rows:
        need(row.get('status')=='reserved' and row.get('contract')==CONTRACT
             and row['lease']['cohort_id']==cohort_id and row['lease']['expected_members']==list(members),
             'port peer failed or changed cohort/member contract')
        ports.extend(p['port'] for p in row['ports']);uuids.extend(row['lease']['gpu_uuids']);leases.append(row['lease']['lease_id'])
    need(len(set(leases))==len(leases) and len(set(uuids))==len(uuids)==8,
         'port peers do not hold distinct full eight-GPU leases')
    need(len(set(ports))==len(ports),'cohort HTTP/P2P/TP-rank port inventories overlap')
    return dict(status='passed',contract=CONTRACT,ports=sorted(ports),lease_ids=leases,
        gpu_uuids=uuids,expected_members=list(members),formal_eligible=False,
        dynamic_ports='Torch/NCCL/IPC endpoints allocated by native engine; not claimed as zero or statically enumerated')


class PortReservations:
    def __init__(self,root,member,specs,out,*,environment_path,timeout_s=180,proc_root='/host/proc'):
        self.root=Path(root);self.directory=self.root/'startup-ports';self.directory.mkdir(exist_ok=True)
        self.member=member;self.specs=list(specs);self.ports=inventory(specs);self.out=Path(out)
        wave=json.loads((self.root/'wave.json').read_text());self.members=wave['members'];self.cohort=wave['cohort_id']
        need(wave.get('startup_port_contract')==CONTRACT,'cohort did not predeclare complete startup port reservations')
        self.lease=lease_binding(environment_path,member=member,cohort_id=self.cohort,members=self.members)
        need(len(self.lease['gpu_uuids'])==sum(s.tp for s in specs),'actual port pair differs from leased GPU count')
        need(os.environ.get('PDBLEND_GPU_UUIDS','').split(',')==self.lease['gpu_uuids'],
             'container allocated physical UUIDs differ from the real queue lease')
        self.timeout=timeout_s;self.proc_root=proc_root;self.sockets={};self.events=[];self.released=False

    def _record(self,kind,**fields):
        row=dict(at_s=time.time(),kind=kind,member=self.member,lease_id=self.lease['lease_id'],**fields)
        self.events.append(row);publish(self.out/f'startup-ports-{len(self.events):02d}-{kind}.json',row)
        return row

    def reserve(self):
        file=self.directory/(self.member+'.json')
        try:
            with (self.directory/'registration.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                need(not file.exists(),'port reservation cannot reuse a prior member attempt')
                for row in self.ports:
                    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                    try:
                        # No SO_REUSEADDR / SO_REUSEPORT: wildcard reservation
                        # also detects a loopback-only or reuse-enabled owner.
                        sock.bind(('0.0.0.0',row['port']));sock.listen(1)
                    except BaseException:sock.close();raise
                    self.sockets[row['port']]=sock
                registration=dict(contract=CONTRACT,status='reserved',lease=self.lease,ports=self.ports,
                    reserved_at_s=time.time(),observer_pid=os.getpid(),pid_namespace=os.readlink('/proc/self/ns/pid'))
                publish(file,registration);self._record('reserved',registration=registration)
            deadline=time.monotonic()+self.timeout
            while True:
                failed=list(self.directory.glob('*.failed.json'))
                need(not failed,'cohort port reservation peer failed: '+str(failed))
                paths=[self.directory/(member+'.json') for member in self.members]
                if all(p.is_file() for p in paths):
                    receipt=validate_registrations([json.loads(p.read_text()) for p in paths],self.members,self.cohort)
                    self._record('cohort-validated',receipt=receipt,
                        registrations=[dict(path=str(p),sha256=sha(p)) for p in paths]);return receipt
                need(time.monotonic()<deadline,'cohort complete-port registration timeout before model loading')
                time.sleep(.05)
        except BaseException as error:
            try:self.failure(error)
            finally:self.close()
            raise

    def before_engine_start(self,spec):
        need(not self.released and not list(self.directory.glob('*.failed.json')),'port cohort failed before engine handoff')
        ports=[p['port'] for p in self.ports if p['instance_id']==spec.instance_id]
        need(ports and all(p in self.sockets for p in ports),'engine port set was not reserved exactly once')
        self._record('engine-handoff',instance_id=spec.instance_id,ports=ports,
            snapshot=socket_snapshot(ports,proc_root=self.proc_root))
        for port in ports:self.sockets.pop(port).close()

    def after_engine_ready(self,spec,engine_pid):
        ports=[p['port'] for p in self.ports if p['instance_id']==spec.instance_id]
        snapshot=socket_snapshot(ports,proc_root=self.proc_root)
        self._record('engine-listeners',instance_id=spec.instance_id,snapshot=snapshot)
        need(not snapshot['proc_errors'] and {s['port'] for s in snapshot['sockets'] if s['state_hex']=='0A'}==set(ports),
             'ready engine lacks complete actual HTTP/TP-rank listeners')
        ownership=listener_ownership(snapshot,engine_pid)
        self._record('engine-listener-ownership',instance_id=spec.instance_id,receipt=ownership)

    def failure(self,error):
        snapshot=socket_snapshot([r['port'] for r in self.ports],proc_root=self.proc_root)
        row=self._record('failed',error=repr(error),snapshot=snapshot)
        path=self.directory/(self.member+'.failed.json')
        if not path.exists():publish(path,row)

    def close(self):
        if self.released:return
        released=list(self.sockets)
        for sock in self.sockets.values():sock.close()
        self.sockets.clear();self.released=True
        self._record('reservations-released',ports=released,unrelated_processes_signalled=False)


def listener_ownership(snapshot,engine_pid):
    """Join container server identity to host socket owners with namespace/start ticks."""
    path=Path('/proc')/str(engine_pid);raw=(path/'stat').read_text()
    local_fields=raw[raw.rfind(')')+2:].split();namespace=os.readlink(path/'ns/pid')
    sources=[]
    for row in snapshot['processes']:
        fields=row['stat'][row['stat'].rfind(')')+2:].split()
        nspid=next((line.split()[1:] for line in row['status'].splitlines() if line.startswith('NSpid:')),[])
        if (row['pid_namespace']==namespace and nspid and int(nspid[-1])==engine_pid
                and fields[19]==local_fields[19]):sources.append((row,fields))
    need(len(sources)==1,'actual engine parent PID/start/namespace has no unique socket-owner mapping')
    process,fields=sources[0];group=fields[2]
    owned={fd['inode'] for row in snapshot['processes']
        if row['pid_namespace']==namespace and row['stat'][row['stat'].rfind(')')+2:].split()[2]==group
        for fd in row['socket_fds']}
    need(all(row['inode'] in owned for row in snapshot['sockets'] if row['state_hex']=='0A'),
         'actual listener is held outside the owned engine process group')
    return dict(engine_local_pid=engine_pid,engine_observer_pid=process['pid'],pid_namespace=namespace,
        start_ticks=int(fields[19]),observer_process_group=int(group),ports=snapshot['ports'],passed=True,
        host_pid_never_inferred_from_container_pid=True)


def host_main():
    """Host wrapper supplies actual ss/proc evidence even though the image lacks ss."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host-out',type=Path,required=True);p.add_argument('--base-port',type=int,required=True)
    p.add_argument('--tp',type=int,choices=(1,2),required=True);p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args();command=args.command[1:] if args.command[:1]==['--'] else args.command
    need(command and command[0]=='docker','host guard only wraps the reviewed Docker launch')
    args.host_out.mkdir(parents=True,exist_ok=False)
    ports=[args.base_port+offset for offset in (0,16)]+[args.base_port+offset+20000+rank for offset in (0,16) for rank in range(args.tp)]
    publish(args.host_out/'before.json',socket_snapshot(ports))
    try:return subprocess.call(command)
    finally:publish(args.host_out/'after.json',socket_snapshot(ports))


if __name__=='__main__':raise SystemExit(host_main())

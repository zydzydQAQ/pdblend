"""CPU contracts for new, measured declared_batch data; no extrapolated ProfilePoint."""
import hashlib
import json
import math
from pathlib import Path
import time

BATCH = 6
INPUT = 7168
OUTPUT = 512
REQUIRED_KV_TOKENS = BATCH * (INPUT + OUTPUT)  # Complete declared context; block-16 aligned.
INITIAL_PREFILL_TOKENS = BATCH * INPUT  # Total across requests; not a single-step budget claim.
TOKENS = 8192
SEQS = 32
MAIN_PACKAGE_SHA = 'e603d153cc86a9427c9acee94177d940b5d911cba975c4eeb52bc5d71f0b58b7'
DEADLINE_SHA = 'fabdbaa4267f63ec208c59ceac5780592ed238918c008db3eadb01e6dcf64e50'


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_runtime_capacity(state, *, require_accepting=True, now=None):
    now = time.time() if now is None else now
    stamp = state.get('timestamp')
    require(type(stamp) in (int,float) and math.isfinite(stamp) and math.isfinite(now)
            and 0 <= now-stamp <= 1., 'capacity runtime is missing or stale')
    require(not state.get('error') and not state.get('runtime_error')
            and state.get('transport_healthy') is True, 'owner/transport unhealthy')
    gen = state.get('generation')
    require(type(gen) is int and gen >= 0 and gen == state.get('acknowledged_generation')
            and state.get('acknowledged_generations') == [gen]
            and state.get('observed_control_generation') == gen
            and state.get('scheduler_budget_pending') is None, 'real owner generation/ACK missing or pending')
    io = state.get('scheduler_io')
    require(isinstance(io,list) and len(io) == 1, 'exactly one TP2 scheduler owner required')
    cache = io[0].get('controls', {}).get('runtime', {})
    require(cache.get('generation') == gen and cache.get('error') is None,
            'owner control cache generation differs')
    require(state.get('scheduler_budget_effective') == dict(max_num_batched_tokens=TOKENS,max_num_seqs=SEQS),
            'effective scheduler budget must remain 8192/32')
    requested = state.get('scheduler_budget', {})
    require(requested == dict(schema_version=1,max_num_batched_tokens=TOKENS,max_num_seqs=SEQS),
            'reported applied control budget differs')
    limits = state.get('scheduler_budget_limits', {})
    require(limits.get('max_num_batched_tokens') == TOKENS and limits.get('max_num_seqs') == SEQS
            and limits.get('max_model_len') == 8192 and limits.get('max_num_partial_prefills') == 1,
            'startup preallocation/context/partial-prefill limits differ')
    require(INPUT <= TOKENS and BATCH <= SEQS and INPUT+OUTPUT <= limits['max_model_len'],
            'declared batch does not fit scheduler or per-request context limits')
    require(state.get('role') == 'mixed' and state.get('mode') == 'continuous', 'mixed continuous state required')
    if require_accepting:
        require(state.get('accepting') is True and state.get('admit_prefill') is True
                and state.get('admit_decode') is True, 'initial admission state differs')
    zero = ('active','running','waiting','transfer_buffered_tensors','transfer_inflight_receives','transfer_inflight_sends')
    require(all(type(state.get(k)) is int and state[k] == 0 for k in zero), 'runtime not idle or residual observation missing')
    require(state.get('kv_allocations') == {} and state.get('transfer_allocations') == {}, 'KV/transfer allocations remain')
    free, total = state.get('free_kv_tokens'), state.get('total_kv_tokens')
    require(type(free) is int and type(total) is int and REQUIRED_KV_TOKENS <= free <= total,
            'real free KV cannot hold all6 complete 7168+512 allocations')
    return dict(required_kv_tokens=REQUIRED_KV_TOKENS,observed_free_kv_tokens=free,
                observed_total_kv_tokens=total,generation=gen,runtime_timestamp_s=stamp,
                initial_prefill_token_sum=INITIAL_PREFILL_TOKENS,scheduler_budget_tokens=TOKENS,max_num_seqs=SEQS,
                interpretation='capacity prerequisite only; no first-step or KV correctness result before actual work')


def validate_declared_batch_observation(raw, events):
    spec=raw['spec'];rows=raw['requests']
    require(spec['batch_size']==BATCH and spec['input_lengths']==[INPUT]*BATCH
            and spec['output_lengths']==[OUTPUT]*BATCH and spec['budget_tokens']==TOKENS
            and spec['max_num_seqs']==SEQS and spec['tp']==2,
            'point is not declared real6 concurrent 7168->512')
    require(len(rows)==BATCH and len({r['request_id'] for r in rows})==BATCH,'missing or duplicate request')
    reference=None
    for row in rows:
        require(row.get('success') is True and len(row.get('prompt_token_ids',[]))==INPUT
                and len(row.get('output_token_ids',[]))==OUTPUT
                and len(row.get('token_received_s',[]))==OUTPUT
                and row.get('usage',{}).get('prompt_tokens')==INPUT
                and row.get('usage',{}).get('completion_tokens')==OUTPUT,'full actual tokens/usage not present')
        if reference is None:reference=row['output_token_ids']
        require(row['output_token_ids']==reference,'same declared prompt/sampling has divergent batch outputs')
    ids={r['request_id'] for r in rows};gen=raw['generation']
    require(events and all(e.get('generation')==gen and e.get('role')=='mixed'
            and e.get('mode')=='continuous' and type(e.get('tokens')) is int
            and 0<=e['tokens']<=TOKENS for e in events),'owner generation/mode/token budget differs')
    full=[e for e in events if e.get('prefill')==0 and e.get('decode')==BATCH
          and e['tokens']==BATCH and set(e.get('request_ids',[]))==ids]
    require(len(full)>=64,'real full declared_batch decode was not sustained for64 steps')
    require(any(e.get('prefill',0)>0 for e in events),'actual prefill phase absent')
    return dict(batch=BATCH,full_batch_decode_steps=len(full),exact_output_equal_within_batch=True,
                output_reference_sha256=hashlib.sha256(json.dumps(reference,separators=(',',':')).encode()).hexdigest(),
                profile_kind='whole_batch',profile_point_generated=False,
                existing_decode_interference_claim=False,kv_correctness_certified=False)


def validate_execution_release(path, package_root, *, now=None):
    """Explicit future operator release and terminal phases; no auto scheduling."""
    now=time.time() if now is None else now
    package_root=Path(package_root).resolve();path=Path(path).resolve()
    release=json.loads(path.read_text())
    require(release.get('authorized_by')=='root' and release.get('execute_once') is True
            and release.get('candidate_manifest_sha256')==sha(package_root/'manifest.json'),
            'explicit root release must bind this frozen candidate')
    deadline=package_root.parent/'deadline-24h-v1/protocol.json'
    require(sha(deadline)==DEADLINE_SHA,'global deadline changed')
    protocol=json.loads(deadline.read_text())
    global_end=protocol['deadline_s']
    expires=release.get('expires_s')
    require(type(expires) in (int,float) and math.isfinite(expires)
            and now+600+90+30<=min(expires,global_end),'full observation, cleanup and evidence tail do not fit release/deadline')
    main=package_root.parent/'B32B-main-scale-fixed-window-v1'
    require(sha(main/'package-manifest.json')==MAIN_PACKAGE_SHA,'main policy package changed')
    phases=release.get('phase_terminal_evidence',{})
    require(set(phases)=={'main','scale'},'both main and scale terminal evidence required')
    result={}
    for phase,item in phases.items():
        source=Path(item['path']).resolve()
        require(source.parent==main/'invocations' and sha(source)==item['sha256'],'terminal phase source differs')
        status=json.loads(source.read_text())
        require(status.get('selected_phase')==phase and status.get('complete') is True
                and status.get('phase') in ('finished','stopped_by_deadline')
                and type(status.get('finished_s')) in (int,float) and math.isfinite(status['finished_s'])
                and status['finished_s']<=now,'phase is not terminal')
        if not status.get('selected_phase_execution_complete'):
            require(release.get('allow_incomplete_phase_termination') is True
                    and isinstance(release.get('incomplete_phase_reason'),str)
                    and len(release['incomplete_phase_reason'])>=10,
                    'incomplete main/scale must be explicitly acknowledged, never called complete')
        result[phase]=dict(path=str(source),sha256=item['sha256'],
            execution_complete=bool(status.get('selected_phase_execution_complete')))
    return dict(release_path=str(path),release_sha256=sha(path),phase_terminal_evidence=result,
                global_deadline_s=global_end,expires_s=expires,baseline_execution=False,scheduled_automatically=False)

"""Original frozen v3 executor pure barrier predicate, byte-for-byte function."""

def require(ok, message):
    if not ok: raise ValueError(message)

def barrier(before, proof, instance):
    require(proof.get('drained') is True and proof.get('accepting') is False
        and proof.get('generation') == before['generation'] + 1
        and proof.get('drain_proof_type') == 'synchronous_put_owner_barrier', 'drain barrier missing')
    ranks = proof.get('transfers')
    require(isinstance(ranks, list) and len(ranks) == instance['tp'], 'all TP ranks not observed')
    for rank in ranks:
        require(rank.get('buffered_tensors') == 0 and rank.get('inflight_receives') == 0
            and rank.get('listener_alive') is True and not rank.get('allocations')
            and not rank.get('buffered_gpu_bytes'), 'rank transfer residue')
        if instance['native_kind'] == 'v3':
            counts = [rank.get(k) for k in ('send_started', 'send_completed', 'send_failed')]
            require(rank.get('send_counters_observed') is True and rank.get('send_healthy') is True
                and type(rank.get('inflight_sends')) is int and rank['inflight_sends'] == 0
                and all(type(v) is int and v >= 0 for v in counts) and counts[0] == counts[1] and counts[2] == 0,
                'rank sends not settled')

"""Owner-thread scheduler budgets bounded by startup allocation; no GPU imports."""
import copy
import threading


def validate_budget_shape(payload):
    if 'scheduler_budget' not in payload:
        return
    budget = payload['scheduler_budget']
    allowed = {'schema_version', 'max_num_batched_tokens', 'max_num_seqs'}
    if (not isinstance(budget, dict) or set(budget) - allowed
            or type(budget.get('schema_version')) is not int
            or budget['schema_version'] != 1
            or not set(budget) & (allowed - {'schema_version'})):
        raise ValueError('invalid scheduler_budget schema')
    for key in ('max_num_batched_tokens', 'max_num_seqs'):
        if key in budget and (type(budget[key]) is not int or budget[key] <= 0):
            raise ValueError('invalid scheduler_budget ' + key)
    if budget.get('max_num_batched_tokens', 16) % 16:
        raise ValueError('scheduler token budget must be 16-aligned')


def initialize_budget(scheduler):
    if not hasattr(scheduler, '_pdblend_budget_limits'):
        config = scheduler.scheduler_config
        scheduler._pdblend_budget_limits = dict(
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_seqs=config.max_num_seqs,
            max_model_len=config.max_model_len,
            max_num_partial_prefills=config.max_num_partial_prefills)
        scheduler._pdblend_budget_owner = threading.get_ident()
        scheduler._pdblend_budget_pending = None
        scheduler._pdblend_applied_generation = -1
    if scheduler._pdblend_budget_owner != threading.get_ident():
        raise RuntimeError('scheduler budget must run on engine owner thread')
    return scheduler._pdblend_budget_limits


def unfinished_sequences(group):
    return sum(not seq.is_finished() for seq in group.get_seqs())


def validate_budget(scheduler, payload):
    limits = initialize_budget(scheduler)
    validate_budget_shape(payload)
    if ('scheduler_budget' in payload
            and not getattr(scheduler, '_pdblend_budget_dynamic_supported', True)):
        raise ValueError('dynamic scheduler budget requires a single scheduler (PP=1)')
    requested = payload.get('scheduler_budget', {})
    tokens = requested.get('max_num_batched_tokens', limits['max_num_batched_tokens'])
    seqs = requested.get('max_num_seqs', limits['max_num_seqs'])
    if not limits['max_num_seqs'] <= tokens <= limits['max_num_batched_tokens']:
        raise ValueError('token budget outside startup preallocated bounds')
    if not 1 <= seqs <= limits['max_num_seqs']:
        raise ValueError('sequence budget outside startup preallocated bounds')
    if ('scheduler_budget' in payload and tokens < limits['max_model_len']
            and (payload['role'], payload['mode']) != ('mixed', 'continuous')):
        raise ValueError('reduced token budget requires mixed continuous chunked prefill')
    if scheduler.scheduler_config.max_model_len != limits['max_model_len']:
        raise ValueError('max_model_len changed after startup')
    # An already accepted multi-sequence waiting/swapped group must remain
    # schedulable; refusing this control leaves every request and queue intact.
    if any(unfinished_sequences(group) > seqs for queue in (scheduler.waiting, scheduler.swapped)
           for group in queue):
        raise ValueError('sequence budget cannot fit an accepted waiting or swapped group')
    return dict(max_num_batched_tokens=tokens, max_num_seqs=seqs)


def apply_budget(scheduler, payload):
    """Validate and stage/apply at an owner boundary, without waiting or queue edits."""
    target = validate_budget(scheduler, payload)
    previous = getattr(scheduler, '_pdblend_runtime', None)
    generation = payload['generation']
    if type(generation) is not int or generation < 0:
        raise ValueError('invalid generation')
    if previous and generation < previous['generation']:
        raise ValueError('stale generation')
    if previous and generation == previous['generation'] and payload != previous:
        raise ValueError('conflicting generation')
    scheduler._pdblend_runtime = copy.deepcopy(payload)
    running = sum(unfinished_sequences(group) for group in scheduler.running)
    if running > target['max_num_seqs']:
        scheduler._pdblend_budget_pending = dict(generation=generation, **target)
        return False
    config = scheduler.scheduler_config
    if config.max_num_batched_tokens != target['max_num_batched_tokens']:
        tokens = target['max_num_batched_tokens']
        scheduler.partial_prefill_budget_lookup_list = [tokens] + [
            tokens // i for i in range(1, config.max_num_partial_prefills + 1)]
        config.max_num_batched_tokens = tokens
    config.max_num_seqs = target['max_num_seqs']
    scheduler._pdblend_budget_pending = None
    scheduler._pdblend_applied_generation = generation
    scheduler._pdblend_applied_runtime = copy.deepcopy(payload)
    return True


def budget_snapshot(scheduler):
    config = scheduler.scheduler_config
    return dict(
        scheduler_budget_limits=copy.deepcopy(scheduler._pdblend_budget_limits),
        scheduler_budget_effective=dict(max_num_batched_tokens=config.max_num_batched_tokens,
                                        max_num_seqs=config.max_num_seqs),
        scheduler_budget_pending=copy.deepcopy(scheduler._pdblend_budget_pending),
        observed_control_generation=getattr(scheduler, '_pdblend_runtime', {}).get('generation', -1),
        acknowledged_generation=scheduler._pdblend_applied_generation)

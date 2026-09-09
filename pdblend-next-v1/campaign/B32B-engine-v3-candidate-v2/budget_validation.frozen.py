"""Real HTTP/GPU budget checks; writes raw evidence and always attempts cleanup."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import uuid

import aiohttp


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def budget(tokens=8192, seqs=32):
    return dict(schema_version=1, max_num_batched_tokens=tokens, max_num_seqs=seqs)


def stable_state(state):
    return {key: state.get(key) for key in ('generation', 'acknowledged_generation',
        'role', 'mode', 'admit_prefill', 'admit_decode', 'scheduler_budget',
        'scheduler_budget_effective', 'scheduler_budget_pending')}


def verify_ack(state, generation, tokens, seqs):
    require(not state.get('error') and not state.get('runtime_error'), 'engine/runtime error')
    require(state.get('generation') == state.get('acknowledged_generation') == generation,
            'owner generation was not applied')
    require(state.get('scheduler_budget_pending') is None, 'budget still pending')
    require(state.get('scheduler_budget_effective') ==
            dict(max_num_batched_tokens=tokens, max_num_seqs=seqs), 'effective budget mismatch')
    observations = [io.get('controls', {}).get('runtime') for io in state.get('scheduler_io', [])]
    require(observations and all(item and item.get('generation') == generation
                                and item.get('error') is None for item in observations),
            'runtime cache and applied generation disagree')


def verify_completion(reply, input_tokens, output_tokens):
    require(reply['status'] == 200, 'completion failed: ' + str(reply['body'])[:300])
    body = reply['body']
    require(isinstance(body, dict), 'completion response was not JSON')
    require(body.get('usage', {}).get('prompt_tokens') == input_tokens, 'wrong prompt work')
    require(body.get('usage', {}).get('completion_tokens') == output_tokens, 'wrong output work')
    require(len(body.get('token_ids', [])) == output_tokens, 'actual output IDs missing')
    return body['token_ids']


class Validation:
    def __init__(self, args):
        self.args = args
        args.out.mkdir(parents=True, exist_ok=True)
        if (args.out / 'result.json').exists() or (args.out / 'http.jsonl').exists():
            raise FileExistsError('use a new output directory; existing evidence is retained')
        self.log = (args.out / 'http.jsonl').open('x', buffering=1)
        self.active = set()
        self.issued = set()
        self.tasks = []
        self.result = dict(schema_version=1, passed=False, started_s=time.time(),
            purpose='actual GPU scheduler-budget correctness, not energy comparison',
            ports=args.ports, runtime_dir=str(args.runtime_dir), instances={}, cleanup={})
        self.save()

    def save(self):
        path = self.args.out / 'result.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.result, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    def task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    async def http(self, port, method, route, body=None, *, label='', request_id=None, timeout=60):
        record = dict(port=port, method=method, route=route, label=label,
                      request=body, request_id=request_id, started_s=time.time())
        headers = {'X-Request-Id': request_id} if request_id else None
        try:
            async with self.session.request(method, f'http://127.0.0.1:{port}{route}',
                    json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                text = await response.text()
                try:
                    response_body = json.loads(text)
                except ValueError:
                    response_body = text
                record.update(status=response.status, body=response_body, finished_s=time.time())
                return record
        except BaseException as exc:
            record.update(error=repr(exc), finished_s=time.time())
            raise
        finally:
            self.log.write(json.dumps(record, allow_nan=False) + '\n')

    async def runtime(self, port, label='runtime'):
        reply = await self.http(port, 'GET', '/runtime', label=label, timeout=10)
        require(reply['status'] == 200, 'runtime unavailable')
        return reply['body']

    async def wait_runtime(self, port, predicate, timeout=15, label='observe'):
        deadline = time.monotonic() + timeout
        while True:
            state = await self.runtime(port, label)
            require(not state.get('error'), 'engine quarantined: ' + str(state.get('error')))
            if predicate(state):
                return state
            require(time.monotonic() < deadline, 'runtime condition timed out: ' + label)
            await asyncio.sleep(.02)

    async def set_budget(self, port, tokens=8192, seqs=32, label='budget'):
        before = await self.runtime(port)
        payload = dict(generation=before['generation'] + 1, role='mixed', mode='continuous',
            admit_prefill=True, admit_decode=True, scheduler_budget=budget(tokens, seqs))
        reply = await self.http(port, 'POST', '/control', payload, label=label)
        require(reply['status'] == 200, 'control rejected: ' + str(reply['body'])[:300])
        after = await self.runtime(port, label + '-ack')
        verify_ack(after, payload['generation'], tokens, seqs)
        return after

    async def generate(self, port, prompt_length, output_length, label, request_id=None):
        request_id = request_id or ('budget-' + uuid.uuid4().hex)
        prompt = ([9707, 1879, 13] * (prompt_length // 3 + 1))[:prompt_length]
        body = dict(prompt=prompt, max_tokens=output_length, temperature=0, top_p=1,
                    ignore_eos=True, seed=0, stream=False)
        self.active.add((port, request_id))
        self.issued.add((port, request_id))
        try:
            return await self.http(port, 'POST', '/v1/completions', body,
                                   label=label, request_id=request_id, timeout=120)
        finally:
            self.active.discard((port, request_id))

    def event_mark(self, instance_id):
        path = self.args.runtime_dir / (instance_id + '.control.events.jsonl')
        require(path.is_file(), 'owner scheduling events file missing: ' + str(path))
        return path, path.stat().st_size

    def verify_events(self, port, mark, tokens, generation, label):
        path, offset = mark
        with path.open('rb') as handle:
            handle.seek(offset)
            raw = handle.read()
        require(not raw or raw.endswith(b'\n'), 'incomplete owner event line')
        events = [json.loads(line) for line in raw.splitlines() if line]
        require(events and any(event.get('tokens', 0) > 0 for event in events), 'no actual model-step events')
        require(all(type(event.get('tokens')) is int and 0 <= event['tokens'] <= tokens
                    for event in events), 'scheduled batch exceeds applied token budget')
        require(all(event.get('generation') == generation for event in events),
                'control changed inside fixed-budget model-step window')
        destination = self.args.out / f'{port}-{label}.events.jsonl'
        destination.write_bytes(raw)
        return dict(path=str(destination), sha256=hashlib.sha256(raw).hexdigest(),
                    steps=len(events), max_tokens=max(event['tokens'] for event in events),
                    generation=generation, applied_token_budget=tokens)

    async def validate_one(self, port):
        result = self.result['instances'][str(port)] = dict(passed=False, phase='initial', checks={})
        initial = await self.runtime(port)
        require(initial.get('active') == 0 and initial.get('running') == 0
                and initial.get('waiting') == 0, 'validation requires an idle dedicated instance')
        result['initial'] = initial
        result['provenance'] = (await self.http(port, 'GET', '/provenance', label='provenance'))['body']
        instance_id = initial['id']
        references = {}
        for tokens in (8192, 1024, 2048):
            result['phase'] = f'output_identity_{tokens}'; self.save()
            state = await self.set_budget(port, tokens, 32, f'set-{tokens}')
            mark = self.event_mark(instance_id)
            outputs = {}
            for length in (128, 7168):
                reply = await self.generate(port, length, 64, f'budget-{tokens}-prompt-{length}')
                ids = verify_completion(reply, length, 64)
                outputs[str(length)] = ids
                if tokens == 8192:
                    references[length] = ids
                else:
                    require(ids == references[length],
                            f'token IDs differ from 8192 reference for prompt {length}, budget {tokens}')
            events = self.verify_events(port, mark, tokens, state['generation'], f'budget-{tokens}')
            result['checks'][f'budget_{tokens}'] = dict(outputs=outputs, ack=state, events=events)
            self.save()

        result['phase'] = 'sequence_shrink'; self.save()
        before = await self.set_budget(port, 8192, 32, 'before-sequence-shrink')
        long_tasks = [self.task(self.generate(port, 128, 512, f'concurrent-long-{index}'))
                      for index in range(2)]
        await self.wait_runtime(port, lambda state: state.get('running', 0) >= 2,
                                label='two-real-running-requests')
        payload = dict(generation=before['generation'] + 1, role='mixed', mode='continuous',
            admit_prefill=True, admit_decode=True, scheduler_budget=budget(8192, 1))
        control_task = self.task(self.http(port, 'POST', '/control', payload, label='sequence-shrink'))
        pending = await self.wait_runtime(port, lambda state: state.get('scheduler_budget_pending') is not None,
                                          timeout=5, label='actual-pending-budget')
        require(pending['scheduler_budget_pending']['generation'] == payload['generation'], 'wrong pending generation')
        require(pending['acknowledged_generation'] == before['generation'], 'pending target falsely acknowledged')
        require(pending['scheduler_budget_effective'] == dict(max_num_batched_tokens=8192, max_num_seqs=32),
                'sequence budget changed before running work became safe')
        require(not control_task.done(), 'HTTP reported success while target was pending')
        completions = await asyncio.gather(*long_tasks)
        for reply in completions:
            verify_completion(reply, 128, 512)
        committed = await control_task
        require(committed['status'] == 200, 'pending sequence shrink failed')
        after = await self.runtime(port, 'sequence-shrink-applied')
        verify_ack(after, payload['generation'], 8192, 1)
        result['checks']['sequence_shrink'] = dict(pending=pending, applied=after,
                                                  completed_requests=2, output_tokens_each=512)
        self.save()

        result['phase'] = 'rejection_and_restore'; self.save()
        before = await self.set_budget(port, 1024, 32, 'before-illegal-mode')
        invalid = dict(generation=before['generation'] + 1, role='mixed', mode='temporal',
            admit_prefill=True, admit_decode=True, scheduler_budget=budget(1024, 32))
        rejected = await self.http(port, 'POST', '/control', invalid, label='reject-short-budget-temporal')
        require(rejected['status'] in (400, 409), 'unsafe temporal mode was accepted')
        require(stable_state(await self.runtime(port)) == stable_state(before), 'rejection mutated control state')
        restored = await self.set_budget(port, 8192, 32, 'explicit-startup-budget-restore')
        base = dict(generation=restored['generation'], role='mixed', mode='continuous',
                    admit_prefill=True, admit_decode=True, scheduler_budget=budget())
        for label, invalid in [('stale', dict(base, generation=base['generation'] - 1)),
                               ('conflict', dict(base, scheduler_budget=budget(2048, 32)))]:
            rejected = await self.http(port, 'POST', '/control', invalid, label='reject-' + label)
            require(rejected['status'] == 409, label + ' generation was accepted')
            require(stable_state(await self.runtime(port)) == stable_state(restored),
                    label + ' generation mutated state')
        result['checks']['invalid_controls'] = dict(temporal_rejected=True,
            stale_rejected=True, conflict_rejected=True, restored=restored)

        result['phase'] = 'real_request_cancellation'; self.save()
        request_id = 'budget-cancel-' + uuid.uuid4().hex
        request_task = self.task(self.generate(port, 128, 512, 'cancel-probe', request_id))
        await self.wait_runtime(port, lambda state: state.get('running', 0) >= 1, label='cancel-probe-running')
        cancelled = await self.http(port, 'POST', '/cancel', dict(request_id=request_id), label='cancel-probe')
        require(cancelled['status'] == 200 and cancelled['body'].get('cancelled') == request_id,
                'cancellation was not acknowledged')
        ended = await request_task
        require(ended['status'] >= 400, 'cancelled request unexpectedly completed normally')
        idle = await self.wait_runtime(port, lambda state: state.get('active') == 0
            and state.get('running') == 0 and state.get('waiting') == 0, label='cancel-probe-idle')
        result['checks']['cancellation'] = dict(reply=cancelled['body'], final=idle)
        result.update(passed=True, phase='checks_passed')
        self.save()

    async def cleanup_one(self, port):
        cleanup = self.result['cleanup'][str(port)] = dict(passed=False, started_s=time.time())
        try:
            targets = (self.active if self.result['instances'].get(str(port), {}).get('passed')
                       else self.issued)
            for active_port, request_id in list(targets):
                if active_port == port:
                    await self.http(port, 'POST', '/cancel', dict(request_id=request_id), label='failure-cleanup-cancel')
            idle = await self.wait_runtime(port, lambda state: state.get('active') == 0
                and state.get('running') == 0 and state.get('waiting') == 0
                and state.get('scheduler_budget_pending') is None, timeout=40, label='cleanup-idle')
            drained = await self.http(port, 'POST', '/drain',
                dict(expected_generation=idle['generation']), label='final-owner-drain')
            require(drained['status'] == 200 and drained['body'].get('drained') is True,
                    'owner drain barrier failed')
            restored = await self.set_budget(port, 8192, 32, 'final-mixed-startup-budget-restore')
            require(restored.get('active') == 0 and restored.get('running') == 0
                    and restored.get('waiting') == 0 and restored.get('accepting'), 'final engine not idle/ready')
            cleanup.update(passed=True, drain=drained['body'], restored=restored)
        except BaseException as exc:
            cleanup['error'] = repr(exc)
        finally:
            cleanup['finished_s'] = time.time()
            self.save()

    async def run(self):
        async with aiohttp.ClientSession(trust_env=False) as self.session:
            try:
                outcomes = await asyncio.gather(*(self.validate_one(port) for port in self.args.ports),
                                                return_exceptions=True)
                for port, outcome in zip(self.args.ports, outcomes):
                    if isinstance(outcome, BaseException):
                        self.result['instances'].setdefault(str(port), {}).update(passed=False, error=repr(outcome))
            finally:
                await asyncio.gather(*(self.cleanup_one(port) for port in self.args.ports))
                if self.tasks:
                    await asyncio.gather(*self.tasks, return_exceptions=True)
                self.result['passed'] = all(self.result['instances'].get(str(port), {}).get('passed')
                    and self.result['cleanup'].get(str(port), {}).get('passed') for port in self.args.ports)
                self.result['finished_s'] = time.time()
                self.save()
                self.log.close()
        return self.result['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ports', type=int, nargs='+', required=True)
    parser.add_argument('--runtime-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(len(args.ports) == len(set(args.ports)) and all(1 <= port <= 65535 for port in args.ports),
            'ports must be distinct valid integers')
    validation = Validation(args)
    passed = asyncio.run(validation.run())
    print(json.dumps(dict(passed=passed, result=str(args.out / 'result.json'))))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()

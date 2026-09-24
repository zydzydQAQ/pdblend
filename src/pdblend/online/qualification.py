"""Functional qualification against an already running native-backed proxy.

This runner neither starts engines nor acquires GPUs. The campaign owner must
hold the fleet lease and provide actual native controls. A passing result is a
functional receipt, never formal energy or performance eligibility.
"""
from __future__ import annotations

import asyncio
import time
import uuid

import aiohttp

from pdblend.online.router import validate_cancel_receipts
from pdblend.online.sse import StreamScan, TerminalStream


async def qualify_live_proxy(proxy, base_url, prompt, *, max_tokens=128, timeout_s=60.,
                             control_action=None):
    """Exercise downstream abort → native ACK → reuse, optionally during control.

    ``control_action`` is an async callback that performs a real DVFS/parking
    transition on the leased fleet and returns its evidence dictionary. Its
    interval must overlap observed successful token arrivals to qualify.
    """
    if proxy.native_cancel is None:
        raise ValueError('qualification requires a native cancellation adapter')
    if (not isinstance(prompt, list) or not prompt or any(type(t) is not int or t < 0 for t in prompt)
            or type(max_tokens) is not int or max_tokens < 2):
        raise ValueError('qualification requires a flat token prompt and at least two output tokens')
    tag = 'online-qualification-' + uuid.uuid4().hex
    result = dict(request_id=tag, started_s=time.time(), functional_passed=False,
                  hardware_qualified=False, formal_eligible=False, energy_comparable=False,
                  checks={}, control_requested=control_action is not None)
    body = dict(prompt=prompt, max_tokens=max_tokens, ignore_eos=True, temperature=0., seed=701,
                request_id=tag, stream=True)

    async def wait_record(request_id, predicate):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            record = next((r for r in reversed(proxy.router.records) if r.request_id == request_id), None)
            if record is not None and predicate(record):
                return record
            await asyncio.sleep(.02)
        raise TimeoutError(f'qualification timed out waiting for {request_id}')

    async def successful_stream(session):
        parser, scan, token_times = TerminalStream(max_tokens, prompt_tokens=len(prompt)), StreamScan(), []
        async with session.post(base_url.rstrip('/') + '/v1/completions',
                                json=dict(body, request_id=tag + '-reuse')) as response:
            if response.status != 200:
                raise RuntimeError(f'reuse HTTP {response.status}: {(await response.text())[:300]}')
            async for chunk in response.content.iter_any():
                parser.feed(chunk)
                _, n = scan.feed(chunk)
                token_times.extend([time.time()] * n)
            completion_tokens = parser.completion_tokens()
        return dict(completion_tokens=completion_tokens, token_times_s=token_times)

    async def run_control():
        started = time.time()
        evidence = await control_action()
        finished = time.time()
        if not isinstance(evidence, dict) or not evidence:
            raise ValueError('control qualification requires actual transition evidence')
        return dict(started_s=started, finished_s=finished, evidence=evidence)

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s)) as session:
            # Closing this response is an actual downstream network disconnect,
            # rather than calling Router.finish or cancelling a test double.
            async with session.post(base_url.rstrip('/') + '/v1/completions', json=body) as response:
                if response.status != 200:
                    raise RuntimeError(f'abort probe HTTP {response.status}: {(await response.text())[:300]}')
                scan = StreamScan()
                async for chunk in response.content.iter_any():
                    _, n = scan.feed(chunk)
                    if n:
                        if scan.done:
                            raise RuntimeError('probe completed before disconnect; increase max_tokens')
                        response.close()
                        break
            record = await wait_record(tag, lambda r: r.terminal_state == 'cancelled_acknowledged')
            receipts = record.route_estimate.get('native_cancel_receipts')
            engine_id = (proxy.transfer.request_id(record.prefill_instance, record.decode_instance, tag)
                         if record.path == 'PD' else tag)
            validate_cancel_receipts(record, receipts, engine_request_id=engine_id)
            result['cancel_receipts'] = receipts
            result['checks']['native_cancel_and_kv_cleanup'] = True
            result['checks']['reservation_released'] = not proxy.router.has_request(tag)
            result['checks']['admission_reopened'] = all(proxy.router.loads[i].accepting for i in record.engine_instances)
            if control_action is None:
                served = await successful_stream(session)
            else:
                served, control = await asyncio.gather(successful_stream(session), run_control())
                result['control'] = control
                result['checks']['tokens_during_control'] = any(
                    control['started_s'] <= t <= control['finished_s'] for t in served['token_times_s'])
            result['reuse'] = served
            reused = await wait_record(tag + '-reuse', lambda r: r.finished_s is not None)
            result['checks']['successful_reuse'] = reused.terminal_state == 'completed' and reused.error is None
            result['functional_passed'] = all(result['checks'].values())
    except Exception as exc:
        result['error'] = repr(exc)
    result['finished_s'] = time.time()
    return result

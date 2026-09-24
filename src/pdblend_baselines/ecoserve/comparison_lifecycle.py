"""Explicit comparison-only shutdown boundaries; author controller is unchanged.

The helper owns no policy, GPU, clock or request routing decision. Its journal
receipts identify the exact HTTP operations affected by wrapper cancellation.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import math
import time

MODE = 'cohort_cancel_and_serial_close/v1'
CONFIG_KEY = 'eco_comparison_lifecycle'
_request = ContextVar('eco_comparison_request', default=None)
_http = ContextVar('eco_comparison_http', default=None)


class ComparisonLifecycle:
    def __init__(self, mode, emit):
        if mode != MODE:
            raise ValueError('unknown EcoServe comparison lifecycle')
        self.emit = emit
        self.active_http = {}
        self.cancel_causes = {}
        self.tasks = {}
        self.sequence = 0
        self.wait_receipt = None
        self.cancel_receipt = None
        self.close_receipt = None

    @contextmanager
    def http_scope(self, instance_id, method, path, body):
        self.sequence += 1
        identifier = 'http-' + str(self.sequence)
        row = dict(http_id=identifier, instance_id=instance_id, method=method,
                   path=path, body=deepcopy(body), request_id=_request.get(),
                   entered_s=time.time())
        self.active_http[identifier] = row
        token = _http.set(identifier)
        try:
            yield
        finally:
            _http.reset(token)
            self.active_http.pop(identifier)

    def decorate_http(self, fields):
        identifier = _http.get()
        if identifier not in self.active_http:
            raise RuntimeError('comparison HTTP receipt has no active operation')
        row = self.active_http[identifier]
        return dict(fields, lifecycle_http_id=identifier,
                    lifecycle_request_id=row['request_id'],
                    lifecycle_cancel_id=self.cancel_causes.get(identifier))

    def create_request_task(self, coroutine, request_id):
        if request_id in self.tasks.values():
            coroutine.close()
            raise ValueError('duplicate comparison request task')

        async def scoped():
            token = _request.set(request_id)
            try:
                return await coroutine
            finally:
                _request.reset(token)

        task = asyncio.create_task(scoped(), name=request_id)
        self.tasks[task] = request_id
        return task

    async def wait_cohort(self, tasks, timeout_s):
        if (not tasks or set(tasks) != set(self.tasks) or type(timeout_s) not in (int, float)
                or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError('comparison cohort/timeout differs')
        self.wait_receipt = dict(at_s=time.time(), monotonic_s=time.monotonic(),
                                 timeout_s=timeout_s, requests=sorted(self.tasks.values()))
        self.emit('eco_comparison_cohort_wait', **self.wait_receipt)
        # Unlike wait_for(gather(...)), wait never cancels child tasks itself.
        _, pending = await asyncio.wait(tasks, timeout=timeout_s)
        if pending:
            raise TimeoutError()
        await asyncio.gather(*tasks)

    def _begin(self, kind, cancel_id, operations, **fields):
        row = dict(at_s=time.time(), monotonic_s=time.monotonic(), cancel_id=cancel_id,
                   pending_http=deepcopy(operations), **fields)
        for operation in operations:
            self.cancel_causes[operation['http_id']] = cancel_id
        self.emit(kind, **row)
        return row

    async def cancel_cohort(self, tasks, *, error):
        pending = [task for task in tasks if not task.done()]
        if pending:
            if set(tasks) != set(self.tasks):
                raise RuntimeError('comparison cancellation cohort differs')
            requests = sorted(self.tasks[task] for task in pending)
            operations = [row for row in self.active_http.values() if row['request_id'] in requests]
            timeout = bool(error == 'TimeoutError()' and self.wait_receipt is not None
                           and time.monotonic() - self.wait_receipt['monotonic_s']
                           >= self.wait_receipt['timeout_s'])
            self.cancel_receipt = self._begin('eco_comparison_cohort_cancel_begin',
                'cohort-cancel', operations, reason='cohort_timeout' if timeout else 'runner_failure',
                error=error, pending_requests=requests)
            # emit above is synchronous: no child cancellation can precede it.
            for task in pending:
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.cancel_receipt is not None:
            self.emit('eco_comparison_cohort_cancel_end', cancel_id='cohort-cancel',
                      requests=list(self.cancel_receipt['pending_requests']), at_s=time.time())

    async def close(self, runtime):
        if any(not task.done() for task in self.tasks):
            raise RuntimeError('comparison close preceded complete cohort termination')
        self.emit('eco_comparison_close_wait', at_s=time.time())
        # Finish any already-started membership transaction. Tasks that have
        # not acquired this lock can then be cancelled by the original close.
        async with runtime.controller.resize_lock:
            self.close_receipt = self._begin('eco_comparison_close_begin',
                'controller-close', list(self.active_http.values()), resize_lock_held=True,
                cohort_tasks_done=True)
            await runtime.close()
            self.emit('eco_comparison_close_end', cancel_id='controller-close', at_s=time.time(),
                      resize_lock_held=True, controller_closed=runtime.controller.closed,
                      runtime_started=runtime.started)

    def summary(self):
        return dict(mode=MODE, request_tasks=len(self.tasks),
                    cohort_cancelled=self.cancel_receipt is not None,
                    serial_close_started=self.close_receipt is not None,
                    policy_changed=False, hardware_qualification=False)

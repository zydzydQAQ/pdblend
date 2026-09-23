"""Lease-scoped UUID power and clock receipts for development GPU execution."""
from __future__ import annotations
import asyncio
import time
from collections import deque
from .deployment import gpu_devices


class GroupTelemetry:
    def __init__(self, gpus, journal):
        import pynvml as nv
        self.nv, self.journal = nv, journal
        self.gpus = tuple(gpus)
        self.handles, self.uuids, self.changed = {}, {}, set()
        self.readings = deque(maxlen=100000)
        self.last_sensor_timestamp = {}
        self.stop_event, self.task = asyncio.Event(), None
        nv.nvmlInit()
        try:
            for gpu, device in zip(self.gpus, gpu_devices(self.gpus)):
                handle = (nv.nvmlDeviceGetHandleByUUID(device) if device.startswith('GPU-')
                          else nv.nvmlDeviceGetHandleByIndex(int(device)))
                self.handles[gpu] = handle
                uuid = nv.nvmlDeviceGetUUID(handle)
                self.uuids[gpu] = uuid.decode() if isinstance(uuid, bytes) else uuid
            if len(set(self.uuids.values())) != len(self.gpus):
                raise ValueError('physical UUID lease contains duplicates')
        except BaseException:
            nv.nvmlShutdown()
            raise

    def clock(self, gpus, frequency):
        if not set(gpus) <= set(self.gpus) or type(frequency) is not int or frequency <= 0:
            raise ValueError('clock operation outside owned GPU lease')
        for gpu in gpus:
            self.nv.nvmlDeviceSetGpuLockedClocks(self.handles[gpu], frequency, frequency)
            self.changed.add(gpu)
        return dict(gpus=list(gpus), gpu_uuids=[self.uuids[g] for g in gpus],
                    requested_frequency_mhz=frequency, timestamp=time.time())

    async def _sample(self):
        while not self.stop_event.is_set():
            for gpu, handle in self.handles.items():
                row = dict(timestamp=time.time(), gpu=gpu, gpu_uuid=self.uuids[gpu],
                           source='nvml:field:186:scope:0:mW', energy_comparable=False)
                try:
                    field = self.nv.nvmlDeviceGetFieldValues(handle, [(186, 0)])[0]
                    if (field.nvmlReturn != self.nv.NVML_SUCCESS or field.valueType != 1
                            or field.fieldId != 186 or field.scopeId != 0):
                        raise RuntimeError('instantaneous NVML power field unavailable')
                    timestamp = int(field.timestamp)
                    if (not -.05 <= time.time() - timestamp/1e6 <= .25
                            or timestamp < self.last_sensor_timestamp.get(gpu, 0)):
                        raise RuntimeError('stale or regressed instantaneous power timestamp')
                    self.last_sensor_timestamp[gpu] = timestamp
                    row['power_w'] = float(field.value.uiVal) / 1000
                    row['sensor_timestamp_us'] = int(field.timestamp)
                    row['frequency_mhz'] = int(self.nv.nvmlDeviceGetClockInfo(handle, self.nv.NVML_CLOCK_SM))
                except Exception as exc:
                    row['error'] = repr(exc)
                self.readings.append(dict(row))
                self.journal('dynamo_power', **row)
            try:
                await asyncio.wait_for(self.stop_event.wait(), .1)
            except asyncio.TimeoutError:
                pass

    def start(self):
        self.task = asyncio.create_task(self._sample())

    async def close(self):
        self.stop_event.set()
        if self.task:
            await self.task
        failures = []
        for gpu in self.changed:
            try:
                self.nv.nvmlDeviceResetGpuLockedClocks(self.handles[gpu])
                self.journal('dynamo_clock_reset', gpu=gpu, gpu_uuid=self.uuids[gpu])
            except Exception as exc:
                failures.append(repr(exc))
        self.nv.nvmlShutdown()
        if failures:
            raise RuntimeError('Dynamo owned clock cleanup failed: ' + repr(failures))

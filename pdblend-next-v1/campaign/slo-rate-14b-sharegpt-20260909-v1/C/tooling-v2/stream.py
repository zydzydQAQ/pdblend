"""Natural SSE parser extracted from the frozen profiler, with durable evidence hooks only."""
import asyncio
import json
import time
import uuid
import aiohttp

def require(ok,reason):
    if not ok:raise ValueError(reason)

class NaturalStream:
    async def request(self, input_length, output_length, gate=None, release=None, arrival_offset_s=0.):
        row = dict(request_id='pdb-a-batch-' + uuid.uuid4().hex, success=False,
            prompt_token_ids=([9707, 1879, 13] * (input_length // 3 + 1))[:input_length],
            output_token_ids=[], token_received_s=[], stream_events=[], requested_output_tokens=output_length,
            token_timing_semantics='client epoch receipt; token IDs sharing a frame share its timestamp')
        self.issued.add(row['request_id'])
        if gate is not None:
            await gate.wait()
            require(release is not None, 'timed request lacks batch release clock')
            row['planned_arrival_s'] = release['epoch_s'] + arrival_offset_s
            remaining = release['monotonic_s'] + arrival_offset_s - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining)
        row['dispatch_s'] = time.time()
        row.setdefault('planned_arrival_s', row['dispatch_s'])
        row['arrival_offset_s'] = arrival_offset_s
        row['dispatch_delay_s'] = row['dispatch_s'] - row['planned_arrival_s']
        body = dict(prompt=row['prompt_token_ids'], max_tokens=output_length, temperature=0,
                    top_p=1, seed=self.args.seed, ignore_eos=True, stream=True)
        try:
            async with self.session.post(f'http://127.0.0.1:{self.args.port}/v1/completions',
                    json=body, headers={'X-Request-Id': row['request_id']},
                    timeout=aiohttp.ClientTimeout(total=120)) as response:
                row['http_status'] = response.status
                require(response.status == 200, 'completion failed: ' + await response.text()
                        if response.status != 200 else '')
                async for line in response.content:
                    line = line.decode().strip()
                    if not line.startswith('data: '):
                        continue
                    data, received = line[6:], time.time()
                    if data == '[DONE]':
                        row['done_marker'] = True
                        break
                    event = json.loads(data)
                    row['stream_events'].append(dict(received_s=received, event=event))
                    self.stream_journal.write(json.dumps(dict(request_id=row['request_id'],received_s=received,event=event))+'\n')
                    self.stream_journal.flush()
                    require(not event.get('error'), 'stream failed: ' + str(event.get('error')))
                    ids = event.get('token_ids', [])
                    require(all(type(token) is int for token in ids), 'noninteger token identity')
                    row['output_token_ids'].extend(ids)
                    row['token_received_s'].extend([received] * len(ids))
                    require(event.get('token_index') == len(row['output_token_ids']), 'token index gap/duplicate')
                    if event.get('usage'):
                        row['usage'] = event['usage']
                row['stream_end_s'] = time.time()
                row['success'] = bool(row.get('done_marker'))
        except BaseException as exc:
            row.update(error=repr(exc), stream_end_s=time.time())
        self.result_journal.write(json.dumps(row,allow_nan=False)+'\n');self.result_journal.flush()
        return row

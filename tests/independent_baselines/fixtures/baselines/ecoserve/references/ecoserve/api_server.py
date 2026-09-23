import json
import asyncio
import multiprocessing
from typing import AsyncGenerator, Dict

import zmq
import zmq.asyncio
import pickle

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from ecoserve.rpc import RPCRequest
from vllm.sampling_params import SamplingParams

from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()

zmq_context = zmq.asyncio.Context()
macro_instance_socket = zmq_context.socket(zmq.constants.PUSH)
output_socket = zmq_context.socket(zmq.constants.PULL)
output_queues: Dict[str, asyncio.Queue] = {}


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_output_loop())


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    prompt_len = request_dict.pop("prompt_len")
    sampling_params = SamplingParams(**request_dict)

    request_id = random_uuid()
    queue = asyncio.Queue()
    output_queues[request_id] = queue

    req = RPCRequest(request_id=request_id,
                     prompt=prompt,
                     sampling_params=sampling_params,
                     prompt_token_ids=[0 for i in range(prompt_len)],
                     prompt_len=prompt_len)
    request_byte = pickle.dumps(req)
    macro_instance_socket.send_multipart((request_byte, ), copy=False)

    results_generator = _get_output(request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        last_output = ""
        async for request_output in results_generator:
            text_outputs = [request_output]
            ret = {"data": text_outputs[0][len(last_output):]}
            last_output = text_outputs[0]
            yield (json.dumps(ret) + "\n\n").encode("utf-8")

        ret = {"data": "[DONE]"}
        yield (json.dumps(ret) + "\n\n").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results(),
                                 media_type="text/event-stream")

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    assert final_output is not None
    text_outputs = [final_output]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


async def _get_output(request_id):
    try:
        queue = output_queues.get(request_id)
        finished = False
        while not finished:
            request_output = await queue.get()
            finished = request_output.finished
            yield request_output.output_text
    finally:
        output_queues.pop(request_id)


async def run_output_loop():
    while True:
        while await output_socket.poll(timeout=10000) == 0:
            pass
        message = await output_socket.recv(copy=False)
        request_outputs = pickle.loads(message.buffer)
        for request_output in request_outputs:
            queue = output_queues.get(request_output.request_id)
            if queue is not None:
                queue.put_nowait(request_output)


def run_api_server(api_server_ip, port, macro_instance_ip,
                   macro_instance_input_port, api_server_output_port):
    api_server_ip = "0.0.0.0"
    macro_instance_socket.connect(
        f"tcp://{macro_instance_ip}:{macro_instance_input_port}")
    output_socket.bind(f"tcp://{api_server_ip}:{api_server_output_port}")
    uvicorn.run(app,
                host=api_server_ip,
                port=port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)

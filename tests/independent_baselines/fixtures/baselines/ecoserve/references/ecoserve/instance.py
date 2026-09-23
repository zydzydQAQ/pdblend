import os
import pickle
import time
from typing import List

import zmq

from ecoserve.args import InstanceArgs
from vllm.engine.llm_engine import LLMEngine
from ecoserve.rpc import (RPCControl, RPCInstanceState,
                                        RPCOutput, RPCRequest)
from vllm.inputs import TextPrompt
from vllm.logger import init_logger
from vllm.outputs import RequestOutput

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 10000


class Instance:

    def __init__(self, engine_args: InstanceArgs) -> None:
        self.instance_id = engine_args.instance_id
        self.local_path = "0.0.0.0"
        self.instance_port = engine_args.instance_port
        self.api_server_port = engine_args.api_server_port
        self.state_port = engine_args.state_port
        self.macro_instance_ip = engine_args.macro_instance_ip
        self.api_server_ip = engine_args.api_server_ip

        self.ctx = zmq.Context()

        # Send instance states to the macro instance.
        self.state_socket = self.ctx.socket(zmq.PUSH)
        self.state_socket.connect(
            f"tcp://{self.macro_instance_ip}:{self.state_port}")
        self.state_socket.setsockopt(zmq.SNDHWM, 0)

        self.engine = LLMEngine.from_engine_args(engine_args)
        self.send_state()

        # Receive input from the macro instance.
        self.input_socket = self.ctx.socket(zmq.PULL)
        self.input_socket.bind(f"tcp://{self.local_path}:{self.instance_port}")
        self.input_socket.setsockopt(zmq.RCVHWM, 0)

        # Send output stream back to the api server.
        self.output_socket = self.ctx.socket(zmq.PUSH)
        self.output_socket.connect(
            f"tcp://{self.api_server_ip}:{self.api_server_port}")
        self.output_socket.setsockopt(zmq.SNDHWM, 0)

        self.send_output = True
        self.TTFT = 0
        self.prefill_time = time.time() * 1000
        self.output_list = []

    @classmethod
    def from_instance_args(cls, instance_args: InstanceArgs):
        """Creates an Instance from the engine arguments."""
        return cls(instance_args)

    def send_state(self, scheduler_outputs=None):
        if scheduler_outputs is None:
            prefill_mode = False
            schedule_queue = []
        else:
            prefill_mode = scheduler_outputs.num_prefill_groups > 0
            schedule_queue = [
                req.seq_group.request_id
                for req in scheduler_outputs.scheduled_seq_groups
            ]
        waiting_queue = [
            req.request_id for req in self.engine.scheduler[0].waiting
        ]
        all_queue = [
            req.request_id for req in self.engine.scheduler[0].waiting +
            self.engine.scheduler[0].running + self.engine.scheduler[0].swapped
        ]

        free_blocks = self.engine.scheduler[
            0].block_manager.get_num_free_gpu_blocks()

        state_info = RPCInstanceState(
            instance_id=self.instance_id,
            prefill_mode=prefill_mode,
            schedule_time=time.time() * 1000,
            schedule_queue=schedule_queue,
            waiting_queue=waiting_queue,
            all_queue=all_queue,
            used_blocks=self.engine.cache_config.num_gpu_blocks - free_blocks,
            free_blocks=free_blocks)

        state_info_byte = pickle.dumps(state_info)
        self.state_socket.send_multipart((state_info_byte, ), copy=False)

    def start(self):
        try:
            try:
                logger.debug("Starting Engine Loop.")
                self.run_instance_loop()
            except Exception as e:
                logger.exception(repr(e))
        except KeyboardInterrupt:
            logger.debug("Shut down Instance.")
        finally:
            logger.debug("Instance is down.")
            self.cleanup()

    def run_instance_loop(self):
        """Core busy loop of the LLMEngine."""
        while True:
            if not self.engine.has_unfinished_requests():
                # Poll until there is work to do.
                self.send_state()
                while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                    logger.debug("Waiting for new requests in engine loop.")
            # Handle any input from the client.
            self.handle_new_input()
            # Engine step.
            schedule_result = self.engine.schedule(0)
            self.send_state(schedule_result[1])
            request_outputs = self.engine.execute(0, *schedule_result)
            # Send request outputs.
            self._send_outputs(request_outputs)

    def _send_outputs(self, request_outputs: List[RequestOutput]):
        """Send List of RequestOutput to server."""
        if request_outputs:
            outputs = [
                RPCOutput(request_output.request_id,
                          request_output.outputs[0].text,
                          request_output.finished)
                for request_output in request_outputs
            ]
            self.output_list.append(outputs)
            if self.send_output or time.time(
            ) * 1000 - self.prefill_time > self.TTFT or self.engine.scheduler[
                    0].get_num_unfinished_seq_groups() == 0:
                for outputs in self.output_list:
                    output_bytes = pickle.dumps(outputs)
                    self.output_socket.send_multipart((output_bytes, ),
                                                      copy=False)
                self.output_list.clear()

    def handle_new_input(self):
        """Handle new input from the macro instance."""
        while self.input_socket.poll(timeout=0) != 0:
            frames = self.input_socket.recv_multipart(copy=False)
            request = pickle.loads(frames[0].buffer)
            if isinstance(request, RPCRequest):
                self._handle_process_request(request)
            elif isinstance(request, RPCControl):
                self.send_output = request.send_output
                self.TTFT = request.TTFT
                self.prefill_time = time.time() * 1000
            else:
                raise ValueError("Unknown RPCRequest Type: "
                                 f"{type(request)}")

    def _handle_process_request(self, request: RPCRequest):
        """Handle RPCProcessRequest by adding it to the LLMEngine."""
        request_id = request.request_id
        try:
            self.engine.add_request(
                request_id,
                TextPrompt(prompt=request.prompt),
                request.sampling_params,
                arrival_time=request.arrival_time,
            )
        except Exception:
            logger.exception("Failed to add request %s, aborting.", request_id)
            # Remove request from the engine.
            self.engine.abort_request(request_id)

    def cleanup(self):
        self.ctx.destroy(linger=0)


def set_visible_gpu(local_rank: int, tensor_parallel_size: int):
    size = tensor_parallel_size
    # Each local instance gets `size` consecutive GPUs. If CUDA_VISIBLE_DEVICES
    # is already set, treat it as the pool of available devices and slice into
    # it; otherwise fall back to absolute device indices.
    pool = os.environ.get("CUDA_VISIBLE_DEVICES")
    if pool:
        devices = [d for d in pool.split(",") if d != ""]
        selected = devices[local_rank * size:local_rank * size + size]
    else:
        selected = [str(local_rank * size + i) for i in range(size)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected)


def run_instance(args: InstanceArgs, local_rank: int):
    set_visible_gpu(local_rank, args.tensor_parallel_size)
    instance = Instance(args)
    instance.start()

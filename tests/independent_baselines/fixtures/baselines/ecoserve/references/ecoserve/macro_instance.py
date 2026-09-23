import csv
import pickle
import time
from dataclasses import dataclass
from typing import Deque, List, Optional

import zmq

from ecoserve.rpc import RPCControl, RPCInstanceState, RPCRequest
from vllm.logger import init_logger

logger = init_logger(__name__)

BLOCK_SIZE = 16


@dataclass
class RequestState:
    request_id: str
    arrival_time: int
    num_iterations: int
    ttft: int
    predict_time: int
    predict_length: int
    prefill_blocks: int


@dataclass
class InstanceState:
    instance_id: int
    requests: Deque[RequestState]
    waiting_queue: List[str]
    free_blocks: int
    prefill_mode: bool
    schedule_time: int
    max_predict_time: int


class MacroInstance:

    def __init__(self,
                 macro_instance_ip: str,
                 macro_instance_input_port: str,
                 state_port: str,
                 instance_ips: List[str],
                 instance_input_ports: List[int],
                 prefill_data_path: str,
                 TTFT: int,
                 TPOT: int,
                 record_file: Optional[str] = None):
        self.context = zmq.Context()
        self.macro_instance_ip = macro_instance_ip

        self.instance_states = []
        self.TTFT = TTFT
        self.TPOT = TPOT

        self.prefill_data = {}
        self.prefill_data_path = prefill_data_path
        self._init_prefill_data()

        self.input_socket = self.context.socket(zmq.constants.PULL)
        self.input_socket.bind(
            f"tcp://{macro_instance_ip}:{macro_instance_input_port}")
        self.input_socket.setsockopt(zmq.RCVHWM, 0)

        self.state_socket = self.context.socket(zmq.constants.PULL)
        self.state_socket.bind(f"tcp://{macro_instance_ip}:{state_port}")
        self.state_socket.setsockopt(zmq.RCVHWM, 0)

        self.instance_count = len(instance_ips)

        self.instances_sockets = []
        for i in range(len(instance_ips)):
            # Initialize per-instance state.
            self.instance_states.append(
                InstanceState(i, Deque(), [], 0, False,
                              time.time() * 1000, 0))

            instance_socket = self.context.socket(zmq.constants.PUSH)
            instance_socket.connect(
                f"tcp://{instance_ips[i]}:{str(instance_input_ports[i])}")
            instance_socket.setsockopt(zmq.SNDHWM, 0)
            self.instances_sockets.append(instance_socket)

        self.prefill_instance = 0

        # Optional scheduling trace. Recording is disabled unless a path is
        # provided so the macro instance can run on any machine out of the box.
        self.record_file = None
        self.writer = None
        if record_file is not None:
            self.record_file = open(record_file,
                                    'w',
                                    newline='',
                                    encoding='utf-8')
            self.writer = csv.writer(self.record_file)
            self.writer.writerow(["instance", "prompt length", "time"])

    def run(self):
        self.run_init_state_loop()
        self.run_macro_instance_loop()

    def run_macro_instance_loop(self):
        while True:
            while self.input_socket.poll(timeout=10000) == 0:
                pass
            frames = self.input_socket.recv_multipart(copy=False)
            request = pickle.loads(frames[0].buffer)
            if isinstance(request, RPCRequest):
                self._handle_request(request)

    def run_init_state_loop(self):
        for i in range(self.instance_count):
            while self.state_socket.poll(timeout=10000) == 0:
                pass
            frames = self.state_socket.recv_multipart(copy=False)
            state = pickle.loads(frames[0].buffer)
            if isinstance(state, RPCInstanceState):
                self._update_state(state)

    def _update_state(self, state: RPCInstanceState):
        self.instance_states[state.instance_id].free_blocks = state.free_blocks
        self.instance_states[
            state.instance_id].prefill_mode = state.prefill_mode
        finished_queue = []
        for request_state in self.instance_states[state.instance_id].requests:
            if request_state.request_id not in state.all_queue and request_state.num_iterations != 0:
                finished_queue.append(request_state)
                continue
            if not state.prefill_mode and request_state.request_id in state.schedule_queue:
                if request_state.num_iterations == 0:
                    request_state.ttft = state.schedule_time - request_state.arrival_time
                request_state.num_iterations += 1

        for request_state in finished_queue:
            self.instance_states[state.instance_id].requests.remove(
                request_state)

    def _handle_request(self, request: RPCRequest):
        while self.state_socket.poll(timeout=0) != 0:
            frames = self.state_socket.recv_multipart(copy=False)
            state = pickle.loads(frames[0].buffer)
            if isinstance(state, RPCInstanceState):
                self._update_state(state)
        instance_id = self.schedule(request)
        request_byte = pickle.dumps(request)
        self.instances_sockets[instance_id].send_multipart((request_byte, ),
                                                           copy=False)

    def _send_control_info(self, send_output: bool, instance_id: int):
        request = RPCControl(send_output, self.TTFT)
        request_byte = pickle.dumps(request)
        self.instances_sockets[instance_id].send_multipart((request_byte, ),
                                                           copy=False)

    def schedule(self, request: RPCRequest):
        now = time.time() * 1000
        num_tokens = request.prompt_len
        predict_time = self._get_predict_time(num_tokens)
        num_blocks = (num_tokens + BLOCK_SIZE) // BLOCK_SIZE

        request_state = RequestState(request.request_id, now, 0, self.TTFT,
                                     predict_time, -1, num_blocks)

        if self._check_constraints(num_blocks, predict_time):
            schedule_instance = self.prefill_instance
        else:
            schedule_instance = self._switch_instance()

        if self.writer is not None:
            self.writer.writerow([schedule_instance, num_tokens, now])

        self.prefill_instance = schedule_instance
        self.instance_states[schedule_instance].requests.append(request_state)
        self.instance_states[schedule_instance].waiting_queue.append(
            request_state.request_id)
        return schedule_instance

    def _check_constraints(self, num_blocks, predict_time) -> bool:
        """Check whether the current prefill instance satisfies the SLO
        constraints, deciding if the request can stay on it."""
        # Time budget left by the requests already on the prefill instance.
        logger.debug("prefill instance %d", self.prefill_instance)
        TPOT_left_time = []
        TTFT_left_time = []
        need_blocks = num_blocks
        need_time = predict_time

        instance_state = self.instance_states[self.prefill_instance]

        for request_state in instance_state.requests:
            if request_state.request_id in instance_state.waiting_queue:
                TTFT_left_time.append(self.TTFT)
                need_blocks += request_state.prefill_blocks
                need_time += request_state.predict_time
            else:
                TPOT_left_time.append(request_state.ttft +
                                      request_state.num_iterations *
                                      self.TPOT -
                                      instance_state.schedule_time +
                                      request_state.arrival_time)

        if need_blocks > instance_state.free_blocks:
            return False
        if len(TPOT_left_time) != 0:
            TPOT_left_time = [max(TPOT_left_time)]
        left_time = TTFT_left_time + TPOT_left_time
        # No requests on this instance yet: it can take the request.
        if len(left_time) == 0:
            return True

        avail_time = min(left_time)
        logger.debug("avail_time=%s need_time=%s", avail_time, need_time)
        if avail_time > need_time:
            return True
        else:
            if need_time > self.TTFT:
                return False
            TPOT_left_time = []
            next_instance = (self.prefill_instance + 1) % self.instance_count
            instance_state = self.instance_states[next_instance]

            for request_state in instance_state.requests:
                TPOT_left_time.append(request_state.ttft +
                                      request_state.num_iterations *
                                      self.TPOT - time.time() * 1000 +
                                      request_state.arrival_time)

            if len(TPOT_left_time) == 0:
                return False
            TPOT_left_time = max(TPOT_left_time)

            return TPOT_left_time < self.TTFT * (self.instance_count -
                                                 1) / self.instance_count

    def _switch_instance(self) -> int:
        now = time.time() * 1000
        self.instance_states[self.prefill_instance].waiting_queue = []
        self._send_control_info(True, self.prefill_instance)
        scheduler_instance = (self.prefill_instance + 1) % self.instance_count
        self._send_control_info(False, scheduler_instance)
        self.instance_states[scheduler_instance].schedule_time = now
        return scheduler_instance

    def _get_predict_time(self, num_tokens) -> int:
        if num_tokens < 16:
            return int(self.prefill_data.get(16, None) * num_tokens / 16)
        time = self.prefill_data.get(4096, None)
        return int(self.prefill_data.get(num_tokens, time * num_tokens / 4096))

    def _init_prefill_data(self):
        with open(self.prefill_data_path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Convert the CSV string fields into numeric types.
                length = int(row['Length'])
                prefill_time = float(row['Prefill Time'])
                # Store in a dict for fast lookup.
                self.prefill_data[length] = prefill_time


def run_macro_instance(macro_instance_ip: str,
                       macro_instance_input_port: str,
                       state_port: str,
                       instance_ips: List[str],
                       instance_input_ports: List[int],
                       prefill_data_path: str,
                       TTFT: int,
                       TPOT: int,
                       record_file: Optional[str] = None):
    macro_instance = MacroInstance(macro_instance_ip,
                                   macro_instance_input_port,
                                   state_port,
                                   instance_ips,
                                   instance_input_ports,
                                   prefill_data_path,
                                   TTFT,
                                   TPOT,
                                   record_file=record_file)
    macro_instance.run()

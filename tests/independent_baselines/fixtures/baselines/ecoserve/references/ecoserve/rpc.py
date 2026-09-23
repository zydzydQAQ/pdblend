from dataclasses import dataclass
from typing import Dict, List, Optional

from vllm.sampling_params import SamplingParams


@dataclass
class RPCRequest:
    request_id: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    prompt: Optional[str] = None
    arrival_time: Optional[float] = None
    predict_output_len: int = 0
    prompt_len: int = 0


@dataclass
class RPCInstanceState:
    instance_id: int
    prefill_mode: bool
    schedule_time: int
    schedule_queue: List[int]
    waiting_queue: List[int]
    all_queue: List[int]
    used_blocks: int
    free_blocks: int


@dataclass
class RPCOutput:
    request_id: int
    output_text: str
    finished: bool


@dataclass
class RPCControl:
    send_output: bool
    TTFT: int

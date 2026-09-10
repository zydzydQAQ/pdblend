# 来源: DistServe (https://github.com/LLMServe/DistServe)
#   evaluation/2-benchmark-serving/structs.py
# 保留 Dataset/TestRequest 的 marshal 序列化 (0-prepare-dataset.py / make_trace.py 所需);
# 删除后半部分依赖 distserve.lifetime 的 RequestResult 定义
# (仅 DistServe 自带客户端使用, 原生 vLLM 下不需要)。
import dataclasses
from typing import List
import marshal

@dataclasses.dataclass
class TestRequest:
    """
    TestRequest: A request for testing the server's performance
    """
    
    prompt: str
    prompt_len: int
    output_len: int
    
@dataclasses.dataclass
class Dataset:
    """
    Dataset: A dataset for testing the server's performance
    """
 
    dataset_name: str	# "sharegpt" / "alpaca" / ...
    reqs: List[TestRequest]
    
    def dump(self, output_path: str):
        marshal.dump({
            "dataset_name": self.dataset_name,
            "reqs": [(req.prompt, req.prompt_len, req.output_len) for req in self.reqs]
        }, open(output_path, "wb"))
    
    @staticmethod
    def load(input_path: str):
        loaded_data = marshal.load(open(input_path, "rb"))
        return Dataset(
            loaded_data["dataset_name"],
            [TestRequest(req[0], req[1], req[2]) for req in loaded_data["reqs"]]
        )

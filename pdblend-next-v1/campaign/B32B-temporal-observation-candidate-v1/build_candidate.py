"""Offline exact-parent patch materializer. Never writes an installed vLLM tree."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'B32B-temporal-engine-review-v1/actual-sources/worker/model_runner.py'
EXPECTED='87ba22eba9b39316e990877b6855f79e78602ad09452d112ad3f78a29960327f'

def patch(source):
    assert hashlib.sha256(source.encode()).hexdigest()==EXPECTED
    def once(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source=source.replace(old,new)
    once('LORA_WARMUP_RANK = 8',
         'from vllm import pdblend_diagnostics as _pdb_diag\n\nLORA_WARMUP_RANK = 8')
    once('    previous_hidden_states: Optional[torch.Tensor] = None\n',
         '    previous_hidden_states: Optional[torch.Tensor] = None\n'
         '    diagnostic_bindings: Optional[Dict[str, Any]] = None\n')
    old='        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)'
    assert source.count(old)==2
    source=source.replace(old,
         '        if self.diagnostic_bindings is not None:\n'
         '            tensor_dict["diagnostic_bindings"] = self.diagnostic_bindings\n'+old)
    once('                                   virtual_engine=virtual_engine)\n',
         '                                   virtual_engine=virtual_engine,\n'
         '                                   diagnostic_bindings=_pdb_diag.bind(\n'
         '                                       seq_group_metadata_list, model_input,\n'
         '                                       sampling_metadata, self.block_size))\n')
    once('        multi_modal_kwargs = model_input.multi_modal_kwargs or {}\n',
         '        diagnostic_context = None\n'
         '        if model_input.diagnostic_bindings is not None:\n'
         '            diagnostic_context = _pdb_diag.begin(\n'
         '                model_input, get_tensor_model_parallel_rank(),\n'
         '                self.parallel_config.tensor_parallel_size,\n'
         '                self.parallel_config.pipeline_parallel_size,\n'
         '                self.model_config.enforce_eager)\n\n'
         '        multi_modal_kwargs = model_input.multi_modal_kwargs or {}\n')
    once('        logits = self.model.compute_logits(hidden_or_intermediate_states,\n'
         '                                           model_input.sampling_metadata)\n',
         '        logits = self.model.compute_logits(hidden_or_intermediate_states,\n'
         '                                           model_input.sampling_metadata)\n'
         '        if diagnostic_context and self.is_driver_worker:\n'
         '            _pdb_diag.capture_logits(diagnostic_context, logits)\n')
    once('        if not self.is_driver_worker:\n            return []\n',
         '        if diagnostic_context:\n'
         '            _pdb_diag.finish(diagnostic_context,\n'
         '                             output if self.is_driver_worker else None)\n\n'
         '        if not self.is_driver_worker:\n            return []\n')
    ast.parse(source)
    return source

def main():
    src=PARENT.read_text();new=patch(src)
    for name,data in [('model_runner.py',new),('candidate.patch',''.join(difflib.unified_diff(
        src.splitlines(True),new.splitlines(True),fromfile='actual/vllm/worker/model_runner.py',
        tofile='candidate/vllm/worker/model_runner.py')))]:
        p=ROOT/name
        if p.exists(): assert p.read_text()==data,'refuse different candidate overwrite'
        else: p.write_text(data)
    print(json.dumps(dict(parent_sha256=EXPECTED,candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),
                         installed_tree_modified=False)))
if __name__=='__main__':main()

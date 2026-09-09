"""Independent service-DVFS-off binding; no GPU action on import/check."""
import hashlib
import json
from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parent
_BINDING = json.loads((_ROOT / 'binding.json').read_text())
for _path, _digest in _BINDING['adapter_files'].items():
    if hashlib.sha256(Path(_path).read_bytes()).hexdigest() != _digest:
        raise RuntimeError('ablation adapter source changed: ' + _path)
sys.path.insert(0, str(Path(_BINDING['adapter'])))
import adapter as _adapter
_BOUND = _adapter.bind(_ROOT)
globals().update({k: v for k, v in vars(_BOUND).items() if not k.startswith('__')})
if __name__ == '__main__':
    _adapter.main(_BOUND)

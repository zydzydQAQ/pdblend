"""Local evidence helpers; no host qualification is inherited."""
from pathlib import Path
import sys
ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
sys.path.insert(0, str(ROOT / 'common/uniform-rate-20260909-v2'))
from support import checked, load, save

"""Original independent verifier with this successor's fresh output paths."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import source_contract, verifier


def verify(reference):
    source_contract()
    return verifier.verify(reference)

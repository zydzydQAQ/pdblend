"""Use the saved native verifier with this continuation's actual evidence paths."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_node as prep


def verify(reference):
    # The source verifier retains its scoped DNS null/[] equivalence and its
    # checks of the completed original eight-cell predecessor.
    verifier = prep.load(prep.RESUME / 'native_saved_verifier.py', 'uniform_B_native_saved')
    verifier.b = prep.control()
    return verifier.verify(reference)

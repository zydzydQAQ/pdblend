"""Exact B release adapter, preserving qualified Eco's distinct correctness protocol."""
import contracts as c

def verify_release(path,expected_sha256):
    c.require(c.Path(path).resolve()==c.RELEASE and expected_sha256==c.RELEASE_SHA,'fixed actual B release required')
    return c.contract().released.verify_release(path,expected_sha256,expected_model='32b',deep=True)

def process_scan():
    return c.contract().released.v1.process_scan()

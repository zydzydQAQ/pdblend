"""Process-local reuse of verified bytes while an immutable campaign grows."""
from collections import OrderedDict
from pathlib import Path
import time


class UnchangedFileHashes:
    """Hash once, then reuse only while the full filesystem identity is stable.

    This cache is never populated from claimed receipt hashes or persisted
    across processes. Callers still compare the returned digest with every
    bound reference. Clear it for a final complete verification.
    """
    def __init__(self, capacity=4096, *, clock=time.time):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('positive hash-cache capacity required')
        self.capacity = capacity
        self.clock = clock
        self.entries = OrderedDict()

    @staticmethod
    def identity(path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def read(self, path, hash_bytes):
        path = Path(path).resolve()
        before = self.identity(path)
        # Some filesystems retain the same mtime/ctime for multiple writes in
        # one clock tick. Never cache freshly written artifacts, even if their
        # nominal nanosecond timestamps and size have not changed.
        stable = self.clock() - max(before[-2:]) / 1_000_000_000 >= 2.
        cached = self.entries.get(path)
        if stable and cached is not None and cached[0] == before:
            self.entries.move_to_end(path)
            return cached[1]
        self.entries.pop(path, None)
        digest = hash_bytes(path)
        if self.identity(path) != before:
            raise ValueError('bound file changed while hashing: ' + str(path))
        if stable:
            self.entries[path] = (before, digest)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return digest

    def clear(self):
        self.entries.clear()

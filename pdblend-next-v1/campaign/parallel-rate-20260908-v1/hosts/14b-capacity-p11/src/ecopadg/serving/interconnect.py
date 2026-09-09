"""Physical placement classes from a recorded NVIDIA peer-topology matrix."""
from dataclasses import dataclass
import hashlib
import re


@dataclass(frozen=True)
class InterconnectTopology:
    matrix: tuple[tuple[str,...],...]
    source_sha256: str

    @classmethod
    def parse(cls,text):
        clean=re.sub(r'\x1b\[[0-9;]*m','',text)
        rows={}
        for line in clean.splitlines():
            fields=line.split()
            if fields and re.fullmatch(r'GPU\d+',fields[0]):
                rows[int(fields[0][3:])]=fields[1:]
        n=len(rows)
        if not n or set(rows)!=set(range(n)):
            raise ValueError('incomplete GPU topology')
        matrix=tuple(tuple(rows[i][:n]) for i in range(n))
        valid={'X','PIX','PXB','PHB','NODE','SYS'}
        if any(len(row)!=n or any(v not in valid and not re.fullmatch(r'NV\d+',v)
                   for v in row) for row in matrix):
            raise ValueError('invalid peer topology matrix')
        return cls(matrix,hashlib.sha256(text.encode()).hexdigest())

    def link_class(self,source,target):
        if set(source)&set(target): raise ValueError('PD groups overlap')
        return '+'.join(sorted({self.matrix[s][d] for s in source for d in target}))

    def intra_class(self,group):
        return '+'.join(sorted({self.matrix[s][d] for s in group for d in group if s!=d})) or 'single'

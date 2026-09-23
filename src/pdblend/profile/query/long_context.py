"""Exact-batch long-context query without collection dependencies."""
import math
from .index import AxisIndex, CurveIndex, FrequencyIndex, IndexQualificationError, MAX_INDEX_BYTES
KIND='exact_batch_piecewise_context_training_means_v1'

def predict(candidate,metric,frequency,batch,context):
    if metric not in ('step_seconds','power_w') or candidate.get('kind')!=KIND or batch not in candidate['exact_batches']:
        raise ValueError('unsupported metric or unqualified batch regime')
    if not math.isfinite(batch) or int(batch) != batch or not math.isfinite(frequency) or int(frequency) != frequency:
        raise ValueError('unqualified exact frequency/batch')
    nodes=candidate['nodes'].get(f'{int(frequency)}/{int(batch)}')
    if not nodes or not nodes[0]['context']<=context<=nodes[-1]['context']:
        raise ValueError('outside frozen long training context domain')
    for left,right in zip(nodes,nodes[1:]):
        if left['context']<=context<=right['context']:
            w=(context-left['context'])/(right['context']-left['context'])
            return left[metric]*(1-w)+right[metric]*w
    raise ValueError('uncovered long training context')


class CompiledLongTable:
    def __init__(self, candidate):
        if candidate.get('kind') != KIND:
            raise ValueError('unknown long context family')
        self.batch_axis = AxisIndex(sorted(candidate['exact_batches']))
        frequencies = sorted({int(key.split('/')[0]) for key in candidate['nodes']})
        self.frequency_axis = FrequencyIndex(frequencies)
        self.tables = [None] * (max(frequencies) + 1)
        self.bytes = self.batch_axis.bytes + self.frequency_axis.bytes + len(self.tables) * 8
        for frequency in frequencies:
            tables = []
            for batch in self.batch_axis.knots:
                nodes = candidate['nodes'].get(f'{frequency}/{batch}')
                if not nodes or len(nodes) < 2:
                    raise ValueError('long context requires two nodes per exact frequency/batch')
                coordinates = [node['context'] for node in nodes]
                pair = tuple(CurveIndex(coordinates, [node[metric] for node in nodes], expression='weighted')
                             for metric in ('step_seconds', 'power_w'))
                self.bytes += sum(curve.bytes for curve in pair)
                tables.append(pair)
            self.tables[frequency] = tuple(tables)
        if self.bytes > MAX_INDEX_BYTES:
            raise IndexQualificationError('O(1) long table memory limit exceeded')

    def curve(self, metric, frequency, batch):
        if metric not in ('step_seconds', 'power_w') or not self.frequency_axis.contains(frequency):
            raise ValueError('missing_profile: unqualified long metric/frequency')
        index = self.batch_axis.lower(batch)
        if self.batch_axis.knots[index] != batch:
            raise ValueError('missing_profile: unqualified long batch')
        return self.tables[int(frequency)][index][0 if metric == 'step_seconds' else 1]

    def predict(self, metric, frequency, batch, context):
        return self.curve(metric, frequency, batch).predict(context)

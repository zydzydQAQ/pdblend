"""Explicit production-arrival-pattern adaptation to independent prompt corpora.

No evaluation order or future output length enters the controller. Historical
time-of-day and weekday/weekend slots retain real seconds; only rates and the
calibration class proportions are mapped to the declared workload.
"""
from collections import Counter
import hashlib
import json
import math

from .policy import SHAPES,classify


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def calibration_mapping(rows,average_rps,provenance):
    if (not rows or type(average_rps) not in (int,float) or not math.isfinite(average_rps) or average_rps<=0
            or provenance.get('split')!='calibration' or provenance.get('evaluation_trace_used') is not False
            or not provenance.get('source_sha256')):raise ValueError('independent calibration and declared average rate required')
    lengths=[];counts=Counter()
    for row in rows:
        n,o=row['input_tokens'],row['output_tokens']
        if (row.get('split','calibration')!='calibration' or type(n) is not int or not 1<=n<=7168
                or type(o) is not int or not 1<=o<=512):raise ValueError('only legal frozen calibration lengths may set mapping')
        lengths.append(dict(input_tokens=n,output_tokens=o));counts[classify(n,o)]+=1
    return dict(schema=1,mode='author_total_arrival_pattern_with_calibration_class_fractions',
        declared_average_rps=average_rps,class_fractions={s:counts[s]/len(rows) for s in SHAPES},
        calibration_lengths=lengths,calibration_lengths_sha256=digest(lengths),provenance=dict(provenance),
        future_trace_used=False,seconds_per_real_second=1,
        formula='max_over_horizon(sum_classes(author_weekday_or_weekend_slot_median_rps)) / (actual_author_week_requests / week_seconds) * declared_average_rps * calibration_class_fraction',
        adaptation='author production arrival pattern; prompt shape proportions from independent corpus calibration; not production forecast-accuracy reproduction')


class MappedWeeklyLoad:
    def __init__(self,template,source,mapping):
        rebuilt=calibration_mapping(mapping['calibration_lengths'],mapping['declared_average_rps'],mapping['provenance'])
        if mapping!=rebuilt:raise ValueError('history mapping differs from frozen calibration identity or class fractions')
        duration=source['end_s']-source['start_s'];count=source['aggregated_requests']
        if (duration<604800 or type(count) is not int or count<=0):raise ValueError('actual full-week arrival exposure required')
        self.template=template;self.mapping=mapping;self.original_average=count/duration
        self.scale=mapping['declared_average_rps']/self.original_average
        self.trained_until=template.trained_until if template is not None else source['end_s']

    def forecast(self,at_s,horizon_s):
        if (type(at_s) not in (int,float) or type(horizon_s) not in (int,float)
                or not math.isfinite(at_s) or not math.isfinite(horizon_s)
                or at_s<self.trained_until or horizon_s<=0):raise ValueError('forecast must follow source week in real time')
        cursor=at_s;peak=0.;slot_s=self.template.slot_s
        while cursor<at_s+horizon_s:
            weekend=(int(cursor//86400)+3)%7>=5;slot=int(cursor%86400//slot_s)
            peak=max(peak,sum(self.template.table[(weekend,slot,s)] for s in SHAPES))
            cursor=(math.floor(cursor/slot_s)+1)*slot_s
        return {shape:peak*self.scale*weight for shape,weight in self.mapping['class_fractions'].items()}

    def receipt(self):
        return dict(mapping=self.mapping,source_average_rps=self.original_average,rate_multiplier=self.scale,
                    real_seconds_preserved=True,production_prediction_accuracy_evaluation=False)


class ReferenceClock:
    def __init__(self,template,*,reference_start_s,wall_start_s):
        if (not math.isfinite(reference_start_s+wall_start_s) or reference_start_s<template.trained_until):
            raise ValueError('reference forecast calendar must follow actual source week')
        self.template=template;self.reference_start_s=reference_start_s;self.wall_start_s=wall_start_s
        self.trained_until=wall_start_s

    def forecast(self,at_s,horizon_s):
        if at_s<self.wall_start_s:raise ValueError('real forecast predates measurement clock origin')
        return self.template.forecast(self.reference_start_s+(at_s-self.wall_start_s),horizon_s)

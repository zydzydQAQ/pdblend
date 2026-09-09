"""Verify original offered work, user SLO and client-to-drain energy window."""
import csv


def validate_replay(b,row,summary,epoch,limits,out):
    b.require(summary.get('measurement_schema')==3 and summary.get('measurement_valid') is True,
              'invalid original-work measurement')
    b.require(summary.get('measurement_window_protocol') is None and not summary.get('fixed_window'),
              'fixed300 must remain disabled for original arrivals')
    b.require(epoch.get('accepted') is True and epoch.get('before_any_request_worker') is True
              and limits['issued_s']<=epoch['actual_epoch_s']<=limits['latest_arrival_epoch_s'],
              'original-work epoch is missing or late')
    b.require(abs(summary['measurement_start_s']-epoch['actual_epoch_s'])<1e-5
              and summary['measurement_end_s']<=limits['cell_execution_deadline_s'],
              'original-work measurement exceeds its real execution budget')
    b.require(summary['n_expected']==row['n_requests']==64
              and summary['trace_sha256']==row['trace_sha256'],'original work hash differs')
    trace=b.read(row['trace']);records=list(csv.DictReader((out/'bench.csv').open()))
    b.require(len(records)==64 and {int(r['idx']) for r in records}==set(range(64)),
              'missing original failed or successful request rows')
    for record in records:
        request=trace['requests'][int(record['idx'])]
        b.require(int(record['prompt_len'])==request['prompt_len']
                  and int(record['output_len'])==request['output_len'], 'original offered tokens changed')
        b.require(abs(float(record['planned_arrival_s'])-epoch['actual_epoch_s']-request['arrival_s'])<1e-5,
                  'original planned arrival changed')
        if record['success']=='1':
            b.require(int(record['input_tokens'])==request['prompt_len']
                      and int(record['generated_tokens'])==request['output_len']
                      and record['token_ids_verified']=='1','completed original token work differs')
    return True

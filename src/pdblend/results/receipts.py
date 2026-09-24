"""JSON-safe receipts for common runtime ownership observations."""
from dataclasses import asdict


def request_record_receipt(record):
    """Keep routing values exact and engine ownership as a sorted JSON array."""
    if record is None:
        return None
    value=asdict(record)
    value['engine_instances']=sorted(record.engine_instances)
    return value

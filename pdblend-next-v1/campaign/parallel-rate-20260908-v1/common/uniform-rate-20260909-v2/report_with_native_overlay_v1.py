"""Keep the audited report and append independent native-refusal classification."""
from report import *
import report as _base
import report_native_overlay_v1 as _overlay


def prepare_snapshot(snapshot):
    return _overlay.apply(snapshot)


def collect(declaration, out):
    return prepare_snapshot(_base.collect(declaration, out))

#!/usr/bin/env python3
"""Create a checksum-verified host-portable Dynamo profile."""
import argparse
from pathlib import Path

from pdblend_baselines.dynamollm.portable_profile import relocate_profile


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--recorded-root", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    relocate_profile(artifact_root=a.artifact_root, recorded_root=a.recorded_root,
                     profile=a.profile, out=a.out)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

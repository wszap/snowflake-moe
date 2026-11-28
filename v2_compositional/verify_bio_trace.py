# -*- coding: utf-8 -*-
"""Hard gate for a completed V1.010 phi trace."""
import argparse
import json
import sys

from bio_update import validate_trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--require-active", action="store_true")
    parser.add_argument("--minimum-scale-std", type=float, default=0.005)
    parser.add_argument("--require-resync", action="store_true")
    args = parser.parse_args()
    verdict = validate_trace(
        args.trace,
        require_active=args.require_active,
        minimum_scale_std=args.minimum_scale_std,
        require_resync=args.require_resync,
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    if not verdict["ok"]:
        print("G-PHI: MECHANISM_NOT_ACTIVE_OR_INVALID", file=sys.stderr)
        return 7
    print("G-PHI: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

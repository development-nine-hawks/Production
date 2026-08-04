"""Run the self-test suite, fastest layers first so failures surface early.

    python selftest/run_all.py              # everything (~2 min)
    python selftest/run_all.py --fast       # logic layers only, no DM libs needed
    python selftest/run_all.py --only t3_orientation
    python selftest/run_all.py --keep-going # do not stop at the first bad layer

Exit code 0 = all checks passed, 1 = at least one failed.
"""
import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _harness  # noqa: E402
from _harness import run_module, failure_count  # noqa: E402

# (module, needs_dm_libs, one-line description)
LAYERS = [
    ("t1_matrix",      False, "GF(2^4) + 2x2 matrix arithmetic"),
    ("t2_codec",       False, "payload codec, exhaustive over 2^16"),
    ("t3_orientation", False, "Top-vs-Right classifier invariant"),
    ("t4_label_e2e",   True,  "synthetic label e2e + v0 regression"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="logic layers only (pure logic, no external deps)")
    ap.add_argument("--only", metavar="MODULE",
                    help="run a single layer, e.g. t1_matrix")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue past a failing layer")
    args = ap.parse_args()

    layers = LAYERS
    if args.only:
        layers = [l for l in LAYERS if l[0] == args.only]
        if not layers:
            print(f"no such layer: {args.only}\n"
                  f"available: {', '.join(m for m, _, _ in LAYERS)}")
            return 2
    elif args.fast:
        layers = [l for l in LAYERS if not l[1]]

    _harness.ensure_dev_secret()

    t0 = time.time()
    ran, skipped = [], []
    for name, needs_dm, desc in layers:
        before = failure_count()
        try:
            mod = importlib.import_module(name)
        except _harness.CheckFailed as e:
            if needs_dm:
                skipped.append((name, str(e).splitlines()[0]))
                continue
            raise
        run_module(vars(mod), f"{name} â€” {desc}")
        ran.append((name, failure_count() - before))
        if failure_count() > before and not args.keep_going:
            print(f"\n  stopping at first failing layer ({name}). "
                  f"Use --keep-going to run the rest.")
            break

    print(f"\n{'=' * 74}\n  SUMMARY  ({time.time() - t0:.0f}s)\n{'=' * 74}")
    for name, fails in ran:
        print(f"  {'PASS' if fails == 0 else 'FAIL':<5} {name}"
              + (f"  ({fails} failed)" if fails else ""))
    for name, why in skipped:
        print(f"  SKIP  {name}  ({why})")
    if skipped:
        print("\n  Skipped layers need the DataMatrix libraries:")
        print("    pip install -r requirements.txt")

    _harness.finish()


if __name__ == "__main__":
    sys.exit(main())

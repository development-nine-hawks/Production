"""Minimal test harness. No external deps. pytest-compatible naming.

Why not pytest: this repo has no test infrastructure, no requirements-dev.txt
and no CI to run one in. What was actually missing is assertions and a non-zero
exit code, which is ~40 lines. Every test here is named `test_*`, takes no
arguments and raises on failure, so `pip install pytest && pytest selftest/`
works verbatim the day someone wants it.
"""
import os
import sys
import time
import types
import traceback

SELFTEST_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SELFTEST_DIR)
OUT_DIR = os.path.join(SELFTEST_DIR, "_out")

for _p in (BACKEND_DIR, SELFTEST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.makedirs(OUT_DIR, exist_ok=True)

_FAILURES = []
_CHECKS = 0
_ENV_READY = False


class CheckFailed(AssertionError):
    pass


def info(msg):
    print(f"      {msg}")


def check(cond, what, detail=""):
    """Assert `cond`, recording the outcome. Returns the boolean."""
    global _CHECKS
    _CHECKS += 1
    if cond:
        print(f"  [ok]   {what}")
        return True
    print(f"  [FAIL] {what}")
    if detail:
        for line in str(detail).splitlines():
            print(f"         {line}")
    _FAILURES.append((what, detail))
    return False


def need(mod, name):
    """Fetch an attribute, failing with a clear message if the API moved."""
    if not hasattr(mod, name):
        raise CheckFailed(
            f"{mod.__name__} has no attribute '{name}' — the API this test "
            f"targets has moved or been renamed."
        )
    return getattr(mod, name)


def ensure_dev_secret():
    """Force the DB offline for the duration of the suite.

    Must run BEFORE config is imported.

    MONGODB_URI: config.py hardcodes LIVE MongoDB Atlas credentials as its
    default (config.py:11-14), and verify_pattern performs a real lookup. Left
    alone, running this suite reads from — and pattern generation would write
    to — the production database. Pointing at an unroutable host makes
    database.init_db() fail its ping and fall back to the local JSON store in
    data/localdb/, which is gitignored. Tests must never touch production.
    """
    global _ENV_READY
    if _ENV_READY:
        return          # idempotent: one layer may import another (t8 -> t7)
    if "config" in sys.modules:
        raise RuntimeError(
            "ensure_dev_secret() must be called before `config` is imported"
        )
    os.environ["MONGODB_URI"] = "mongodb://127.0.0.1:1/selftest-offline"
    os.environ["MONGODB_DB"] = "selftest_offline"
    _ENV_READY = True


def import_engine(require_dm_libs=False):
    """Import cdp_engine, stubbing zxingcpp when it is absent.

    cdp_engine.py does a bare top-level `import zxingcpp`, so layers 1-5 — which
    test pure arithmetic and have no business needing a barcode decoder — would
    otherwise be unrunnable without the DM libraries installed.

    Returns (module, have_real_dm_libs).
    """
    ensure_dev_secret()
    have = True
    try:
        import zxingcpp  # noqa: F401
        from pylibdmtx.pylibdmtx import encode  # noqa: F401
    except Exception as e:
        have = False
        if require_dm_libs:
            raise CheckFailed(
                f"This layer needs the DataMatrix libraries: {e}\n"
                f"Install with: pip install -r requirements.txt"
            )
        stub = types.ModuleType("zxingcpp")
        stub.BarcodeFormat = types.SimpleNamespace(DataMatrix=128)
        stub.Binarizer = types.SimpleNamespace(
            LocalAverage=0, GlobalHistogram=1, FixedThreshold=2, BoolCast=3)
        stub.read_barcodes = lambda *a, **k: []
        sys.modules.setdefault("zxingcpp", stub)

    import cdp_engine
    return cdp_engine, have


def run_module(mod_globals, title):
    """Run every test_* callable in a module's namespace. Returns failure count."""
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")
    tests = sorted(
        (n, f) for n, f in mod_globals.items()
        if n.startswith("test_") and callable(f)
    )
    before = len(_FAILURES)
    for name, fn in tests:
        t0 = time.time()
        print(f"\n  > {name}")
        try:
            fn()
        except CheckFailed as e:
            check(False, f"{name} aborted", str(e))
        except Exception:
            check(False, f"{name} raised", traceback.format_exc())
        info(f"({time.time() - t0:.1f}s)")
    return len(_FAILURES) - before


def finish():
    print(f"\n{'=' * 74}")
    if _FAILURES:
        print(f"  FAILED — {len(_FAILURES)} of {_CHECKS} checks")
        for what, _ in _FAILURES:
            print(f"    ✗ {what}")
        print("=" * 74)
        sys.exit(1)
    print(f"  PASSED — all {_CHECKS} checks")
    print("=" * 74)
    sys.exit(0)


def failure_count():
    return len(_FAILURES)

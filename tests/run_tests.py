"""Run GhostAgent's tests with the stick's Python, from the repo root:

    GhostAgent\\python\\python.exe -I tests\\run_tests.py            all tests
    GhostAgent\\python\\python.exe -I tests\\run_tests.py -v         one line per test
    GhostAgent\\python\\python.exe -I tests\\run_tests.py -k fake    only tests whose name contains "fake"

No GPU, no model and no network beyond 127.0.0.1. Browser tests use the bundled Chromium, headless.
Exit code 0 when everything passed, 1 otherwise.
"""

import argparse
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True  # no __pycache__ folders in the repo or on the stick

import helpers  # noqa: E402,F401  (isolates LOCALAPPDATA and friends before any test imports the agent)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the GhostAgent test suite.")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-k", dest="patterns", action="append", help="only run tests matching this substring")
    ap.add_argument("--failfast", action="store_true")
    a = ap.parse_args(argv)

    loader = unittest.TestLoader()
    if a.patterns:
        loader.testNamePatterns = [f"*{p}*" for p in a.patterns]
    suite = loader.discover(str(HERE), pattern="test_*.py", top_level_dir=str(HERE))
    t0 = time.time()
    result = unittest.TextTestRunner(verbosity=2 if a.verbose else 1, failfast=a.failfast).run(suite)
    took = time.time() - t0
    print()
    if not result.testsRun:
        # Most likely a mistyped -k pattern: say so rather than report a green run of nothing.
        print("No tests matched, so nothing was tested.")
        return 1
    print(f"ran {result.testsRun} tests in {took:.1f}s: {len(result.failures)} failed, {len(result.errors)} errors, "
          f"{len(result.skipped)} skipped")
    print("OK" if result.wasSuccessful() else "FAILED")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

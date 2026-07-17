"""Minimal test runner for environments without pytest. CI uses real pytest."""
import importlib.util, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

passed = failed = 0
for tf in sorted(Path(__file__).parent.glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(tf.stem, tf)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for name in dir(mod):
        if name.startswith("test_") and callable(getattr(mod, name)):
            try:
                getattr(mod, name)(); passed += 1
                print(f"  PASS {tf.stem}::{name}")
            except Exception:
                failed += 1
                print(f"  FAIL {tf.stem}::{name}")
                traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)

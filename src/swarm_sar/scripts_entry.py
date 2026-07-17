"""Console-script entry points (see [project.scripts] in pyproject.toml)."""
import runpy, sys
from pathlib import Path
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run(name):
    target = _SCRIPTS / name
    if not target.exists():
        # Wheel installs place the package in site-packages, where the repo's
        # scripts/ directory does not exist. Fail with guidance instead of a
        # cryptic FileNotFoundError.
        raise SystemExit(
            f"{name} is only available from a source checkout "
            "(git clone + pip install -e .). Run the script directly from the "
            "repository's scripts/ directory."
        )
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


def demo():      _run("run_demo.py")
def train():     _run("train.py")
def evaluate():  _run("evaluate.py")

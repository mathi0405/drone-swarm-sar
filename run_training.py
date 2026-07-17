"""Compatibility launcher for the end-to-end training/reporting command."""
import multiprocessing
import runpy
import sys
from pathlib import Path


def main():
    script = Path(__file__).resolve().parent / "scripts" / "train_and_report.py"
    sys.argv = [str(script), "--config", "configs/training/mappo_easy.yaml", *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

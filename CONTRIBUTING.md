# Contributing

Thanks for your interest in Swarm-SAR!

## Dev setup

```bash
pip install -e ".[all]" -r requirements-dev.txt
pre-commit install
```

## Workflow

1. Create a branch: `git checkout -b feature/my-thing`.
2. Format & lint: `make format && make lint`.
3. Add tests and run them: `make test` (keep coverage from regressing).
4. Keep the smoke demo green: `python scripts/run_demo.py --episodes 1 --figures`.
5. Open a PR describing the change and, for research changes, the metric impact
   (mean ± std across ≥5 seeds).

## Conventions

- Black (line length 100) + Ruff. Type hints on public functions.
- New modules get a docstring explaining *why* they exist.
- New heavy dependencies must be optional (guarded import) unless essential to the
  core NumPy path.
- Reproducibility first: thread seeds through; never rely on global RNG state.

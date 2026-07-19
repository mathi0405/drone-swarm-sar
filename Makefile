.PHONY: help install install-dev demo figures train evaluate experiments dashboard test lint format docker paper clean
PY ?= python

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:        ## Install core (numpy/matplotlib) runtime
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .
install-dev:    ## Install everything incl. RL, viz and dev tools
	$(PY) -m pip install -e ".[all]" -r requirements-dev.txt
demo:           ## Run a self-contained heuristic SAR episode
	$(PY) scripts/run_demo.py --episodes 1 --figures
figures:        ## Regenerate all publication figures
	$(PY) scripts/generate_figures.py --out results/figures
train:          ## Train MAPPO (see configs/training/*.yaml)
	$(PY) scripts/train.py --config configs/training/mappo_transformer_gnn.yaml
train-real:     ## Turnkey GPU run: train->eval->REAL figures (needs .[rl])
	$(PY) scripts/train_and_report.py --archs mlp gru gnn transformer transformer_gnn --seeds 0 1 2 --timesteps 1000000
smoke-train:    ## Tiny end-to-end training validation (~1-2 min)
	$(PY) scripts/train_and_report.py --smoke
CKPT ?= results/trained/best.pt
evaluate:       ## Evaluate a trained checkpoint: make evaluate CKPT=path/to/best.pt
	$(PY) scripts/evaluate.py --config configs/training/mappo_transformer_gnn.yaml --ckpt $(CKPT)
experiments:    ## Run ablation grid (multi-seed)
	$(PY) scripts/run_experiments.py --config configs/experiments/ablation_comm.yaml
dashboard:      ## Launch the Streamlit dashboard
	streamlit run src/swarm_sar/dashboard/app.py
test:           ## Run unit tests
	$(PY) -m pytest -q
lint:           ## Ruff + black --check
	ruff check src tests && black --check src tests
format:         ## Auto-format
	black src tests scripts && ruff check --fix src tests
docker:         ## Build the Docker image
	docker build -t swarm-sar:latest .
paper:          ## Compile the IEEE paper (needs pdflatex)
	cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + ; rm -rf build dist *.egg-info

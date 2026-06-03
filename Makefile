# ============================================================
#  Attrition — LLM Agents, War of Attrition, and the US Debt Ceiling
#  GNU Makefile
# ============================================================

PYTHON     := PYTHONPATH=. python
STREAMLIT  := PYTHONPATH=. streamlit

.PHONY: setup data theory cost-estimate run run-full reproduce-paper dashboard test clean help

# ---- Primary targets ----------------------------------------

setup: ## Install all runtime dependencies
	python -m pip install -r requirements.txt

data: ## Fetch FRED data, preprocess, and build episode dataset
	$(PYTHON) -m src.data.fred_fetcher
	$(PYTHON) -m src.data.preprocessor
	$(PYTHON) -m src.data.episode_builder

theory: ## Run Alesina-Drazen (1991) theoretical baseline
	$(PYTHON) -m src.theory.alesina_drazen

cost-estimate: ## Print estimated API cost before running simulations
	$(PYTHON) -m src.utils.cost_estimator

run: ## Smoke run — 2 sims/condition, full pipeline (data → theory → simulations → analysis → viz)
	SIMS_PER_CONDITION=2 LLM_CACHE_MODE=read-write $(MAKE) data theory
	SIMS_PER_CONDITION=2 LLM_CACHE_MODE=read-write $(PYTHON) -m src.simulations.batch_runner
	SIMS_PER_CONDITION=2 LLM_CACHE_MODE=read-write $(PYTHON) -m src.analysis.main
	SIMS_PER_CONDITION=2 LLM_CACHE_MODE=read-write $(PYTHON) -m src.viz.figures

run-full: ## Full run — 25 sims/condition, full pipeline
	LLM_CACHE_MODE=read-write $(MAKE) data theory
	LLM_CACHE_MODE=read-write $(PYTHON) -m src.simulations.batch_runner
	LLM_CACHE_MODE=read-write $(PYTHON) -m src.analysis.main
	LLM_CACHE_MODE=read-write $(PYTHON) -m src.viz.figures

reproduce-paper: ## Reproduce paper results from cached LLM responses (no new API calls)
	LLM_CACHE_MODE=read-only $(PYTHON) -m src.analysis.main
	LLM_CACHE_MODE=read-only $(PYTHON) -m src.viz.figures

dashboard: ## Launch the Streamlit results dashboard
	$(STREAMLIT) run dashboard/app.py

test: ## Run the test suite with coverage
	python -m pytest tests/ -v

clean: ## Remove generated outputs (preserves cache/llm/)
	rm -rf outputs/transcripts outputs/results outputs/figures

# ---- Help ---------------------------------------------------

help: ## Show this help message
	@echo ""
	@echo "  Attrition — LLM Agents, War of Attrition, and the US Debt Ceiling"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

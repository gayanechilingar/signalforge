# Convenience targets. Everything here is a thin wrapper over `sf` or pytest so
# there is exactly one implementation of each operation.

.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint fmt typecheck test test-live gate demo ingest index extract score \
        bakeoff ab serve docker clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install with dev extras
	$(UV) sync --all-extras

lint: ## Ruff check
	$(UV) run ruff check src tests

fmt: ## Ruff format
	$(UV) run ruff format src tests

typecheck: ## mypy
	$(UV) run mypy src/signalforge

test: ## Hermetic tests (no network, no Ollama)
	$(UV) run pytest -q -m "not live"

test-live: ## All tests including EDGAR and Ollama
	$(UV) run pytest -q

gate: ## Regression gate, as CI runs it
	$(UV) run sf eval gate --chain ci

demo: ## Full pipeline on two companies, end to end
	$(UV) run sf ingest AAPL MSFT --limit 4
	$(UV) run sf index
	$(UV) run sf extract guidance_tone
	$(UV) run sf extract event_class
	$(UV) run sf extract risk_delta
	$(UV) run sf score

ingest: ## make ingest T="AAPL MSFT"
	$(UV) run sf ingest $(T)

index: ## Embed new chunks
	$(UV) run sf index

extract: ## make extract TASK=guidance_tone
	$(UV) run sf extract $(TASK)

score: ## Recompute signals and alerts
	$(UV) run sf score

bakeoff: ## make bakeoff TASK=guidance_tone
	$(UV) run sf eval bakeoff $(TASK) --models llama32-3b,llama31-8b,llama3-8b

ab: ## make ab TASK=guidance_tone
	$(UV) run sf eval ab $(TASK) --chain llama31-8b

serve: ## API + dashboard on :8000
	$(UV) run sf serve --reload

docker: ## Build the image
	docker build -t signalforge:local .

clean: ## Remove the warehouse and caches
	rm -rf data .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

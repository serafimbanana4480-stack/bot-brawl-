# ============================================================================
# Makefile — Brawl Stars Bot
# ============================================================================
# Requires: Python 3.10+, pip, docker (optional)
# ============================================================================

.PHONY: help install install-dev test lint typecheck format security build docker train run clean

PYTHON := python
PIP := pip
DOCKER_IMAGE := brawl-bot

help:
	@echo "Brawl Stars Bot — Available targets:"
	@echo "  make install      Install runtime dependencies"
	@echo "  make install-dev  Install dev dependencies (includes runtime)"
	@echo "  make test         Run pytest with coverage"
	@echo "  make lint         Run ruff linter"
	@echo "  make format       Run ruff + black formatters"
	@echo "  make typecheck    Run mypy type checker"
	@echo "  make security     Run bandit + pip-audit"
	@echo "  make build        Build Docker image"
	@echo "  make train        Start model training pipeline"
	@echo "  make run          Start the bot API server"
	@echo "  make clean        Remove build artifacts"

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
test:
	pytest --cov=. --cov-report=term-missing --cov-report=html -v

lint:
	ruff check .

format:
	ruff check . --fix
	ruff format .
	black .

typecheck:
	mypy .

security:
	bandit -r . -f screen
	pip-audit --desc

# ---------------------------------------------------------------------------
# Build / Run
# ---------------------------------------------------------------------------
build:
	docker build -t $(DOCKER_IMAGE):latest .

docker-run:
	docker run -p 8000:8000 --rm $(DOCKER_IMAGE):latest

train:
	$(PYTHON) -m training.train_yolo

run:
	$(PYTHON) -m uvicorn api_server:app --host 127.0.0.1 --port 8003 --reload

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ coverage.xml bandit-report.json pip-audit-report.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

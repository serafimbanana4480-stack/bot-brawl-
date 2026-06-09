# Contributing to Soberana Omega Brawl Stars Bot

Thank you for your interest in contributing! This document outlines the workflow, style guide, and development setup.

## Development Environment Setup

### Prerequisites
- **Python 3.10, 3.11, or 3.12** (3.11 recommended)
- **Git**
- **Docker** (optional, for containerized builds)
- **ADB** (Android Debug Bridge) — required for emulator integration
- **BlueStacks** or another Android emulator (Windows target)

### 1. Clone & Install
```bash
git clone <repo-url>
cd "bot brawl"
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
make install-dev
```

### 2. Pre-commit Hooks
We use pre-commit to enforce code quality before every commit:
```bash
pre-commit install
pre-commit run --all-files
```

### 3. Environment Configuration
Copy the example environment file and adjust values:
```bash
cp .env.example .env
# Edit .env with your local settings (emulator port, model paths, etc.)
```

**Never commit `.env` or `config.json` with real secrets.**

## Running Tests

```bash
# Full suite with coverage
make test

# Individual test file
pytest tests/test_vision.py -v

# Integration tests (requires emulator)
pytest tests/integration/ -v --timeout=120
```

## Code Style Guide

- **Formatter**: `ruff format` + `black` (safety net)
- **Linter**: `ruff` (line length 120, target Python 3.9+)
- **Type Checker**: `mypy` with `ignore_missing_imports = true`
- **Docstrings**: Google-style or NumPy-style for public APIs
- **Naming**:
  - `snake_case` for functions/variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

### Quick Quality Check
```bash
make lint      # ruff
make format    # ruff + black
make typecheck # mypy
make security  # bandit + pip-audit
```

## Submitting a Pull Request

1. **Branch**: Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Commit**: Write clear, imperative commit messages:
   ```
   feat(vision): add brawler detection caching
   fix(api): handle missing CORS origins gracefully
   docs(readme): update installation steps
   ```

3. **Push & PR**: Push your branch and open a PR against `main`.
   - Fill out the PR template (if available).
   - Ensure **CI passes** (lint, typecheck, tests, security scan).
   - Request review from at least one maintainer.

4. **Merge**: Squash-merge is preferred for clean history.

## CI/CD Pipeline

Our GitHub Actions workflow runs on every push/PR:
- **Lint** (`ruff`, `black`) — Windows runner
- **Type Check** (`mypy`) — Windows runner
- **Tests** (`pytest`) — Matrix: Python 3.10, 3.11, 3.12 — Windows runner
- **Security** (`bandit`, `pip-audit`) — Windows runner
- **Docker Build** — Ubuntu runner (after tests pass)

## Reporting Issues

- Use GitHub Issues with the appropriate label (`bug`, `feature`, `docs`).
- Include Python version, OS, and steps to reproduce.
- Attach logs (redact sensitive info) if applicable.

## Questions?

Open a Discussion or reach out to the Soberana Omega Team.

---

**Happy coding!**

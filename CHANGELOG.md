# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-06-08

### Added
- Initial release of the Soberana Omega Brawl Stars Bot.
- Real-time computer vision pipeline with YOLOv8 object detection.
- Adaptive combat system with anti-ban humanization.
- FastAPI backend with WebSocket real-time updates.
- Dashboard for match history, replay analysis, and telemetry.
- ADB integration for BlueStacks emulator control.
- Brawler queue management with trophy targeting.
- Safety system: session limits, APM bounds, auto-stop on detection.
- Training pipeline for YOLO model fine-tuning.

### Infrastructure & Tooling
- `pyproject.toml` with setuptools backend and optional GPU dependencies.
- `requirements.txt` (runtime) and `requirements-dev.txt` (development) with pinned major versions.
- Multi-stage `Dockerfile` with ADB, non-root user, and `HEALTHCHECK`.
- GitHub Actions CI/CD with matrix builds (Python 3.10/3.11/3.12) on Windows runners.
- Jobs: lint (`ruff`), format (`black`), type check (`mypy`), tests (`pytest`), security (`bandit`, `pip-audit`), Docker build.
- `.pre-commit-config.yaml` with `ruff`, `black`, `mypy`, `bandit`, and generic file hygiene hooks.
- Structured JSON logging (`core/logging_config.py`) with correlation IDs, ISO8601 timestamps, and log rotation.
- Health check system (`core/health_checks.py`) with `/health` and `/health/deep` endpoints.
- Secrets management via `.env` + `python-dotenv`; `.env.example` provided.
- `Makefile` with targets: `install`, `install-dev`, `test`, `lint`, `format`, `typecheck`, `security`, `build`, `train`, `run`, `clean`.
- `CONTRIBUTING.md` with setup instructions, style guide, and PR workflow.
- `CHANGELOG.md` (this file) following Keep a Changelog format.

### Changed
- Migrated sensitive configuration from `config.json` to environment variables (`.env`).
- Dockerfile now uses `pip install -e .` instead of broken `requirements.txt` reference.
- CI runner switched from `ubuntu-latest` to `windows-latest` to match primary target platform (Windows + BlueStacks).

### Fixed
- Dockerfile build failure due to missing `requirements.txt`.
- CI workflow referencing non-existent `requirements.txt`.

### Security
- Added `bandit` static analysis for Python security issues.
- Added `pip-audit` to detect known vulnerabilities in dependencies.
- Container runs as non-root user (`brawlbot`).
- `.env` and `config.json` excluded from version control.

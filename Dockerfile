# ============================================================================
# Multi-stage Dockerfile for Brawl Stars Bot
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only pyproject.toml first for layer caching
COPY pyproject.toml ./

# Install project in editable mode (dependencies resolved from pyproject.toml)
RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Soberana Omega Team"
LABEL description="AI-powered Brawl Stars Bot"
LABEL version="1.0.0"

WORKDIR /app

# Create non-root user
RUN groupadd -r brawlbot && useradd -r -g brawlbot -d /app -s /sbin/nologin brawlbot

# Install runtime system dependencies: ADB + CV libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    android-tools-adb \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Ensure required asset directories exist before copying application code
RUN mkdir -p /app/images /app/models /app/logs && chown -R brawlbot:brawlbot /app

# Copy application code (after dependencies for layer caching)
COPY --chown=brawlbot:brawlbot . /app

# Switch to non-root user
USER brawlbot

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    ADB_SERVER_SOCKET=tcp:host.docker.internal:5037 \
    BRAWL_BOT_API_PORT=8003

# Expose API port (must match config.json and BRAWL_BOT_API_PORT)
EXPOSE 8003

# Health check — uses the same port the application listens on
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Default entrypoint
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8003"]

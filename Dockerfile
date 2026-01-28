# AI Futu Trader Docker Image
# Multi-stage build for optimal size

# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2: Runtime
# ==========================================
FROM python:3.11-slim as runtime

# Labels
LABEL maintainer="AI Futu Trader"
LABEL version="1.0.0"
LABEL description="Ultra-low latency trading system with Futu OpenD"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r trader && useradd -r -g trader trader

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R trader:trader /app

# Copy application code
COPY --chown=trader:trader src/ /app/src/
COPY --chown=trader:trader tests/ /app/tests/
COPY --chown=trader:trader requirements.txt /app/

# Switch to non-root user
USER trader

# Expose Prometheus metrics port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/metrics')" || exit 1

# Default command
CMD ["python", "-m", "src.run", "--simulate"]

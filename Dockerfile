# Multi-stage build for SRE Autopilot (Standalone HF Space)
# Aligned with the official echo_env standalone pattern.

# Build stage
FROM ghcr.io/meta-pytorch/openenv-base:latest AS builder

WORKDIR /app

# Ensure uv is available
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    install -m 0755 /root/.local/bin/uv /usr/local/bin/uv && \
    install -m 0755 /root/.local/bin/uvx /usr/local/bin/uvx

# Copy environment code
COPY . /app/env
WORKDIR /app/env

# Install dependencies using uv sync (no-editable for production)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# Final runtime stage
FROM ghcr.io/meta-pytorch/openenv-base:latest

# HF Spaces requirement: Use non-root user (ID 1000)
# (Already handled in base image, but we ensure correct home dir)
WORKDIR /home/user/app

# Copy the virtual environment and code from builder
COPY --from=builder /app/env/.venv /home/user/app/.venv
COPY --from=builder /app/env /home/user/app/env

# Set environment variables
ENV PATH="/home/user/app/.venv/bin:$PATH"
ENV PYTHONPATH="/home/user/app/env:$PYTHONPATH"
ENV PORT=7860
ENV ENABLE_WEB_INTERFACE=true

# Health check matching echo_env pattern
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run the FastAPI server from the environment directory
CMD ["sh", "-c", "cd /home/user/app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]

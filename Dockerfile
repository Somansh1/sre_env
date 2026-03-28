# Production-ready Dockerfile for SRE Autopilot (Standalone HF Space)
# Standardized on port 8000 to match the echo_env pattern and README metadata.

FROM ghcr.io/meta-pytorch/openenv-base:latest

# HF Spaces requirement: Use non-root user (ID 1000)
WORKDIR /home/user/app

# Copy all files to the root
COPY --chown=user . /home/user/app/

# Install dependencies directly into the system/user python
RUN pip install --no-cache-dir \
    "openenv-core[core]==0.2.1" \
    "fastapi>=0.115.0" \
    "pydantic>=2.0.0" \
    "uvicorn>=0.24.0" \
    "requests>=2.31.0"

# Install the current directory as a package
RUN pip install --no-cache-dir -e .

# Set environment variables
ENV PYTHONPATH="/home/user/app:$PYTHONPATH"
ENV PORT=8000
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONUNBUFFERED=1

# Health check matching echo_env
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server
# Using uvicorn directly to ensure it picks up the root modules correctly
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

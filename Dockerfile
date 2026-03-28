# Robust Dockerfile for SRE Autopilot (Standalone HF Space)
# Uses pip for maximum compatibility on Hugging Face Spaces.

FROM ghcr.io/meta-pytorch/openenv-base:latest

# HF Spaces requirement: Use non-root user (ID 1000)
# The base image already has a 'user' but we ensure we are in the right spot.
WORKDIR /home/user/app

# Copy all files to the app directory
COPY --chown=user . /home/user/app/

# Install dependencies directly into the system/user python to avoid venv issues on HF
# We install openenv-core[core] to get the standardized environment wrapper
RUN pip install --no-cache-dir \
    "openenv-core[core]==0.2.1" \
    "fastapi>=0.115.0" \
    "pydantic>=2.0.0" \
    "uvicorn>=0.24.0" \
    "requests>=2.31.0"

# Install the current package in editable mode to ensure local imports work
RUN pip install --no-cache-dir -e .

# Set environment variables
ENV PYTHONPATH="/home/user/app:$PYTHONPATH"
ENV PORT=7860
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run the FastAPI server
# We use the 'server' script defined in pyproject.toml or call uvicorn directly
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

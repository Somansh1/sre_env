# Use a standard Python 3.10 image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Create a non-root user for HF Spaces security (User ID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# 1. Install openenv core from the CORRECT official source
RUN git clone https://github.com/meta-pytorch/OpenEnv.git /tmp/openenv \
    && cd /tmp/openenv && pip install --user .

# 2. Pre-install known dependencies
RUN pip install --user fastapi pydantic uvicorn requests

# 3. Copy your environment code
COPY --chown=user . /home/user/app/envs/sre_env

# 4. Install the environment WITHOUT trying to re-resolve core
RUN cd /home/user/app/envs/sre_env && pip install --user --no-deps .

# Expose port (HF default)
EXPOSE 7860
ENV PORT=7860
ENV PYTHONPATH="/home/user/app"

# Launch the FastAPI server
CMD ["python", "-m", "envs.sre_env.server.app"]

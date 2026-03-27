# Use a standard Python 3.10 image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Create a non-root user for HF Spaces security (User ID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Install openenv core from source
RUN git clone https://github.com/meta-platforms/openenv.git /tmp/openenv \
    && cd /tmp/openenv && pip install --user -e .

# Copy your environment code (ensure everything is owned by 'user')
COPY --chown=user . /home/user/app/envs/sre_env

# Install your environment dependencies
RUN cd /home/user/app/envs/sre_env && pip install --user -e .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860
ENV PORT=7860
ENV PYTHONPATH="/home/user/app"

# Launch the FastAPI server as a module
CMD ["python", "-m", "envs.sre_env.server.app"]

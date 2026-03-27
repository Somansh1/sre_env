FROM openenv-base:latest

WORKDIR /app

# Copy the entire sre_env directory into the container
COPY . /app/envs/sre_env

# Install the environment package
RUN cd /app/envs/sre_env && pip install -e .

EXPOSE 8000

ENV PYTHONPATH="/app"
ENV PORT=8000

CMD ["python", "-m", "envs.sre_env.server.app"]

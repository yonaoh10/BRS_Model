# Call-QA pipeline image (CPU). Runs the whole pipeline in mock mode out of the
# box, and the real ASR/judge stages against a GPU box via config.cloud.yaml.
# Not the bank deployment - that follows the offline runbook in README.md.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY requirements.txt pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir -r requirements.txt

# The package is installed for real (not editable): that is what the bank
# will do, and it is how the missing-templates defect was found.
COPY src/ src/
RUN pip install --no-cache-dir .

COPY config/ config/
COPY scripts/ scripts/
COPY dashboard/ dashboard/
COPY cloud/ cloud/

# Nothing in the pipeline downloads models at runtime.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

# data/ (recordings, artifacts, state) and config/ are mounted by compose.
RUN mkdir -p data/input/calls data/output

ENTRYPOINT ["scripts/docker_entrypoint.sh"]
CMD ["run", "--mock"]

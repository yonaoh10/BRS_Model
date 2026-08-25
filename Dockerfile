# Optional container image (CPU, mock/core only). Not required for the PoC.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/
RUN pip install --no-cache-dir -e .

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["python", "-m", "callqa"]
CMD ["--help"]

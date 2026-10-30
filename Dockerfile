FROM ghcr.io/astral-sh/uv:0.10.8 AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    CHECKPOINT_PATH=/models/checkpoint.pt \
    VOCAB_PATH=/models/vocab.pkl \
    MERGES_PATH=/models/merges.pkl \
    DEVICE=cpu

COPY requirements-service.txt ./
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    -r requirements-service.txt

COPY cs336_basics ./cs336_basics

EXPOSE 8000

CMD ["python", "-m", "cs336_basics.service"]

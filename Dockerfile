# Arena. Serves the FastAPI app (UI + API). No model weights are bundled: steering
# scoring runs on NDIF, and only the tokenizer/config are fetched at runtime.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf_home

WORKDIR /app

COPY requirements.txt requirements-model.txt ./
RUN pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-model.txt

COPY app/ ./app/
COPY db/ ./db/
COPY web/ ./web/
COPY data/directions/ ./data/directions/
COPY data/probes/ ./data/probes/

RUN mkdir -p /tmp/hf_home /app/data/cache && chmod -R 777 /tmp/hf_home /app/data

EXPOSE 7860
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements-api.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-api.txt

RUN mkdir -p artifacts/models artifacts/reports
COPY artifacts/models/model.joblib ./artifacts/models/model.joblib
COPY artifacts/models/metadata.json ./artifacts/models/metadata.json
COPY artifacts/reports/feature_importance.json ./artifacts/reports/feature_importance.json
COPY artifacts/reports/demo_samples.json ./artifacts/reports/demo_samples.json
COPY src/__init__.py ./src/__init__.py
COPY src/api ./src/api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

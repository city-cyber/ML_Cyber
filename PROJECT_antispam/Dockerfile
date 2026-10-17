FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MODEL_DIR=/app/microservice/app/model

WORKDIR /app

COPY microservice/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home app

COPY src/__init__.py src/preprocessing.py src/inference.py src/artifacts.py /app/src/
COPY microservice/__init__.py /app/microservice/__init__.py
COPY microservice/app/ /app/microservice/app/

USER 10001:10001
RUN python -c "from microservice.app.main import app; from src.artifacts import load_artifact; load_artifact()"
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["python", "-m", "uvicorn", "microservice.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

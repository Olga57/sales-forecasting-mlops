FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.3.3 \
    PIP_DEFAULT_TIMEOUT=2000 \
    POETRY_REQUESTS_TIMEOUT=2000


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip && \
    pip install "poetry==$POETRY_VERSION"

WORKDIR /app


COPY pyproject.toml poetry.lock ./

# IMPORTANT: faster + less lock resolving issues
RUN poetry config virtualenvs.create false \
    && poetry config installer.max-workers 10 \
    && poetry config installer.parallel true \
    && poetry install --only main --no-interaction --no-ansi --no-root



EXPOSE 8000 5000
WORKDIR /app
COPY . /app

CMD ["bash", "-c", "\
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlruns \
--host 0.0.0.0 \
--port 5000 & \
uvicorn fmcg_sales_forecasting.api.app:app \
--host 0.0.0.0 \
--port 8000"]
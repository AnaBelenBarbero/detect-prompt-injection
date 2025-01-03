FROM python:3.12-slim

ARG POETRY_VERSION=1.8.2

# Set working directory
WORKDIR /code

# Copy requirements files
COPY pyproject.toml poetry.lock README.md ./

# Install system dependencies and poetry
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl

# Install project dependencies
RUN pip3 install --no-cache-dir poetry==${POETRY_VERSION} \ 
    && poetry env use 3.12
    
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

COPY src/ ./src

EXPOSE 80

CMD ["poetry", "run", "uvicorn", "run", "src.predict:app", "--host", "0.0.0.0", "--port", "${PORT:-80}"]


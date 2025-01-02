# Use Python 3.12 slim as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /code

# Copy requirements files
COPY pyproject.toml poetry.lock ./

# Install system dependencies and poetry
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false

# Install project dependencies
RUN poetry install --no-dev --no-interaction

# TODO: Review which files to copy
COPY . .

CMD ["fastapi", "run", "app/main.py", "--port", "80"]

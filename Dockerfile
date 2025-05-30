FROM python:3.12-slim

ARG POETRY_VERSION=1.8.2

# Set working directory
WORKDIR /code

COPY pyproject.toml poetry.lock README.md ./

COPY src/ ./src

RUN pip3 install --no-cache-dir poetry==${POETRY_VERSION} \ 
    && poetry env use 3.12

# only main
RUN poetry install --only main --no-interaction --no-ansi

EXPOSE 80

CMD ["poetry", "run", "uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "80"]

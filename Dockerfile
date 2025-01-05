FROM python:3.12-slim

ARG POETRY_VERSION=1.8.2

# Set working directory
WORKDIR /code

# Copy requirements files
COPY pyproject.toml poetry.lock README.md ./

COPY src/ ./src

RUN pip3 install --no-cache-dir poetry==${POETRY_VERSION} \ 
    && poetry env use 3.12

RUN poetry install --no-dev --no-interaction --no-ansi

EXPOSE 80

CMD ["poetry", "run", "uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "80"]

# Install system dependencies and poetry
#RUN apt-get update \
#    && apt-get install -y --no-install-recommends curl
#
## Install project dependencies
#RUN pip3 install --no-cache-dir poetry==${POETRY_VERSION} \ 
#    && poetry env use 3.12
#
#COPY src/ ./src
#
#RUN poetry config virtualenvs.create false \
#    && poetry install --no-dev --no-interaction --no-ansi
#
#EXPOSE $PORT
#
#CMD ["poetry", "run", "uvicorn", "src.dummy_predict:app", "--host", "0.0.0.0", "--port", "$PORT"]


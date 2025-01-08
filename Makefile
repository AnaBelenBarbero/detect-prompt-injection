.PHONY: run-train run-api-dev docker-build-local docker-build-local-w-frontend docker-run-local

run-train:
	poetry run python train.py

run-api-dev:
	poetry install
	poetry run fastapi dev src/predict.py 

run-frontend-dev:
	poetry install --with frontend
	poetry run streamlit run app/main.py

docker-build-local:
	docker build -t detect-prompt-injection -f docker_dev/Dockerfile_dev .

docker-build-local-w-frontend:
	docker build -t detect-prompt-injection -f docker_dev/Dockerfile_dev_w_frontend .

docker-run-local:
	docker run -p 80:80 -p 8501:8501 detect-prompt-injection

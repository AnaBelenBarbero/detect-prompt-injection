run-train:
	poetry run python train.py

run-api-dev:
	poetry run fastapi dev src/predict.py 

docker-build:
	docker build -t detect-prompt-injection .

docker-run:
	docker run -p 8000:8000 detect-prompt-injection

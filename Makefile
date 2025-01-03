run-train:
	poetry run python train.py

run-api-dev:
	poetry run fastapi dev src/predict.py 

docker-build:
	docker build -t detect-prompt-injection .

docker-run:
	docker run -p 80:80 detect-prompt-injection

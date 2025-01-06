run-train:
	poetry run python train.py

run-api-dev:
	poetry run fastapi dev src/predict.py 

docker-build-local:
	docker build -t detect-prompt-injection -f Dockerfile_dev .

docker-run-local:
	docker run -p 80:80 detect-prompt-injection

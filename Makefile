run-train:
	poetry run python train.py

run-api-dev:
	fastapi dev predict.py 

docker-build:
	docker build -t bias-hr-impact .

docker-run:
	docker run -p 8000:8000 bias-hr-impact

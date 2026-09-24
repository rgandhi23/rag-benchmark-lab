.PHONY: install test lint run evaluate docker-build docker-up docker-down release-check

install:
	python -m pip install '.[dev]'

test:
	pytest --cov=ragbench --cov-report=term-missing

lint:
	ruff check .

run:
	uvicorn ragbench.api:app --reload

evaluate:
	ragbench evaluate

docker-build:
	docker build --tag rag-benchmark-lab:local .

docker-up:
	docker compose up --build --detach

docker-down:
	docker compose down

release-check: lint test docker-build

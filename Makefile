.PHONY: install test lint run evaluate

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

.PHONY: train test lint format

train:
	python -m src.models.train

test:
	pytest tests/ -v

lint:
	ruff check src/

format:
	ruff format src/

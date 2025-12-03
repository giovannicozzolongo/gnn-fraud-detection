.PHONY: train train-temporal train-ssl baselines ablation eval test lint format clean download

download:
	python -m src.data.download

train:
	python -m src.models.train

train-temporal:
	python -m src.models.train --temporal

train-ssl:
	python -m src.models.train --temporal --ssl

baselines:
	python -m src.models.train --baselines

ablation:
	python -m src.evaluation.ablation

eval:
	python -m src.evaluation.plots

test:
	pytest tests/ -v

lint:
	ruff check src/

format:
	ruff format src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +

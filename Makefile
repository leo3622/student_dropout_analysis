PYTHON ?= python3
VENV ?= .venv

.PHONY: install train test

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

train:
	$(PYTHON) scripts/model_pipeline.py

test:
	PYTHONPATH=. pytest -q

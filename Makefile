.PHONY: install test train train-fast evaluate demo serve all

install:
	pip install -r requirements.txt

test:
	python -m pytest -q tests

train:
	python train.py

train-fast:
	python train.py --skip-bert

evaluate:
	python evaluate.py

demo:
	python app.py

serve:
	uvicorn api:app --host 0.0.0.0 --port 8000

all: install test train evaluate

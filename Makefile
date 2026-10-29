.PHONY: install dev run test lint clean docker-build docker-run

install:
	pip install --upgrade pip
	pip install -r requirements.txt

dev: install
	pip install -e ".[dev]"

run:
	streamlit run app.py

test:
	python -m unittest discover -s tests -v

pytest:
	pytest -v

lint:
	ruff check .

docker-build:
	docker build -t pretrade:latest .

docker-run:
	docker run --rm -p 8501:8501 --env-file .env pretrade:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info

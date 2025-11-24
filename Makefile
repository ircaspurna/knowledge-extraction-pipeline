# Makefile for Knowledge Extraction Pipeline

.PHONY: help install install-dev test lint format clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install package dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make test         - Run test suite"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code"
	@echo "  make clean        - Remove build artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src/knowledge_extraction --cov-report=html

lint:
	ruff check src/ tests/ scripts/
	mypy src/ tests/ scripts/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

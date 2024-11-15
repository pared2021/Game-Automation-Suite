# Game Automation Suite Makefile

.PHONY: help install test lint format clean coverage security check all

help:
	@echo "Available commands:"
	@echo "  make install    - Install all dependencies"
	@echo "  make test      - Run tests"
	@echo "  make lint      - Run code quality checks"
	@echo "  make format    - Format code"
	@echo "  make clean     - Clean up build and test artifacts"
	@echo "  make coverage  - Generate test coverage report"
	@echo "  make security  - Run security checks"
	@echo "  make check     - Run all checks (lint, test, security)"
	@echo "  make all       - Run all tasks"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=game_automation --cov-report=term-missing

lint:
	flake8 game_automation tests
	black --check game_automation tests
	mypy game_automation
	pylint game_automation tests

format:
	black game_automation tests
	isort game_automation tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete

coverage:
	pytest tests/ --cov=game_automation --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

security:
	bandit -r game_automation -c setup.cfg
	safety check
	pip-audit

check: lint test security

all: clean install format check coverage

# Development shortcuts
.PHONY: dev watch

dev:
	python main/full_feature_launcher.py

watch:
	watchmedo auto-restart --directory=./ --pattern=*.py --recursive -- python main/full_feature_launcher.py

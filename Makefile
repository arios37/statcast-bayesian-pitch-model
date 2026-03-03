.PHONY: test lint clean

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

clean:
	rm -rf data/*.parquet data/*.nc data/*.csv
	rm -rf figures/*.png
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .ruff_cache
	rm -rf *.egg-info src/*.egg-info

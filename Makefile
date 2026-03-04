.PHONY: test lint clean export frontend-install frontend-dev frontend-build deploy

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
	rm -rf frontend/dist frontend/node_modules

# Export posteriors from PyMC model to JSON for the frontend
export:
	python -m src.export_posteriors --output frontend/public/data/posteriors.json

# Frontend commands
frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

# Full pipeline: export model data then build frontend
deploy: export frontend-build

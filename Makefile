.PHONY: test lint sample mock-e2e clean

test:
	python -m pytest -q

lint:
	ruff check src tests scripts

sample:
	python scripts/generate_sample_data.py

mock-e2e: sample
	python -m callqa run --mock
	python -m callqa report --mock
	python -m callqa calibrate --mock
	@echo "Open data/output/reports/index.html"

clean:
	rm -rf data/output data/callqa_state.db*

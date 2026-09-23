.PHONY: test lint sample mock-e2e eval golden dashboard clean

# Everything here runs with the core dependencies alone (requirements.txt).
# No GPU, no models and no network are needed for any target in this file.

test:
	python -m pytest -q

lint:
	ruff check src tests scripts dashboard

sample:
	python scripts/generate_sample_data.py

mock-e2e: sample
	python -m callqa run --mock
	python -m callqa report --mock
	python -m callqa calibrate --mock
	@echo "Open data/output/reports/index.html"

eval:
	python -m callqa eval --mock --baseline eval/baseline.json

golden:
	python scripts/build_golden_set.py

dashboard:
	python dashboard/server.py

clean:
	rm -rf data/output data/callqa_state.db*

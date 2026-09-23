.PHONY: test lint sample mock-e2e eval golden dashboard clean

# Everything here runs with the core dependencies alone (requirements.txt).
# No GPU, no models and no network are needed for any target in this file.
#
# PYTHON is resolved rather than hardcoded: a stock Debian or RHEL host has
# python3 and no `python` at all, so every target here failed on exactly the
# kind of clean server this is meant to be deployed on.
PYTHON ?= $(shell command -v python3 2>/dev/null || command -v python)

test:
	$(PYTHON) -m pytest -q

lint:
	# $(wildcard) expands to nothing when dashboard/ has been deleted, which
	# the hand-off documentation says is supported. Naming it unconditionally
	# made `make lint` fail on a tree that was in exactly the state the
	# documentation describes.
	ruff check src tests scripts $(wildcard dashboard)

sample:
	$(PYTHON) scripts/generate_sample_data.py

mock-e2e: sample
	$(PYTHON) -m callqa run --mock
	$(PYTHON) -m callqa report --mock
	$(PYTHON) -m callqa calibrate --mock
	@echo "Open data/output/reports/index.html"

eval:
	$(PYTHON) -m callqa eval --mock --baseline eval/baseline.json

golden:
	$(PYTHON) scripts/build_golden_set.py

dashboard:
	$(PYTHON) dashboard/server.py

clean:
	rm -rf data/output data/callqa_state.db*

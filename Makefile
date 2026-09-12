.PHONY: test lint sample mock-e2e clean cloud-up cloud-status cloud-down cloud-remove

test:
	python -m pytest -q

lint:
	ruff check src tests scripts dashboard cloud

sample:
	python scripts/generate_sample_data.py

mock-e2e: sample
	python -m callqa run --mock
	python -m callqa report --mock
	python -m callqa calibrate --mock
	@echo "Open data/output/reports/index.html"

clean:
	rm -rf data/output data/callqa_state.db*

# --- dev-phase cloud option (see cloud/README.md) -------------------------
cloud-up:
	python cloud/runpod_cli.py up

cloud-status:
	python cloud/runpod_cli.py status

cloud-down:
	python cloud/runpod_cli.py down

# Removes the cloud option entirely (files + code references), then verifies.
# Run this when moving to the bank servers. Dry-run first:
#   python scripts/remove_cloud_option.py
cloud-remove:
	python scripts/remove_cloud_option.py --apply

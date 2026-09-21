UV ?= uv

.PHONY: install install-local TensorRT run bench verify

install:
	$(UV) sync --no-dev --upgrade
	$(UV) run --no-sync rtmw-download

install-local:
	$(UV) run --no-project --python 3.12 -m src.rtmw_preview.install_local --uv "$(UV)"
	$(UV) run --no-sync rtmw-download

tensorrt:
	$(UV) sync --no-dev
	$(UV) run --no-sync python -m rtmw_preview.pose

run:
	$(UV) run --no-dev rtmw-preview

bench:
	$(UV) run --no-dev python -m rtmw_preview.bench

verify:
	$(UV) run --no-dev python -m rtmw_preview.verify

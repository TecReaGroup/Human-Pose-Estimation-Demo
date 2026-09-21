UV ?= uv

.PHONY: install install-local TensorRT run

install:
	$(UV) sync --no-dev --upgrade
	$(UV) run --no-sync rtmw-download

install-local:
	$(UV) sync --no-dev --upgrade --find-links package || $(UV) sync --no-dev --upgrade --refresh
	$(UV) run --no-sync rtmw-download

TensorRT:
	$(UV) sync --no-dev
	$(UV) run --no-sync python -m rtmw_preview.pose

run:
	$(UV) run --no-dev rtmw-preview

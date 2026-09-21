UV ?= uv

.PHONY: install install-local TensorRT run

install:
	$(UV) sync --no-dev --upgrade
	$(UV) run --no-sync rtmw-download

install-local:
	$(UV) sync --no-dev --upgrade --find-links package || $(UV) sync --no-dev --upgrade --refresh
	$(UV) run --no-sync rtmw-download

TensorRT:
	$(UV) sync --no-dev --upgrade
	$(UV) run --no-sync python -c "from rtmw_preview.runtime import configure_logging; configure_logging(); from rtmw_preview.pose import load_gpu_runtime; load_gpu_runtime()"

run:
	$(UV) run --no-dev rtmw-preview

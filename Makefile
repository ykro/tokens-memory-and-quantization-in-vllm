PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)
REGION ?= us-east4
QUANT_MODEL ?= google/gemma-4-E4B-it-qat-w4a16-ct
QWEN_MODEL ?= Qwen/Qwen3.5-4B
QWEN_BASE_URL ?=

.PHONY: all verify verify-live report test lint facts deploy deploy-quant deploy-qwen \
	stage stage-quant tunnel tunnel-quant tunnel-qwen teardown mock clean

# No GPU needed: lint, tests and the judge against the built-in mock servers
all: lint test verify

verify:
	$(PYTHON) verify_memory.py --mock

# Real numbers: needs `make tunnel` and `make tunnel-quant` open in other terminals.
# Add QWEN_BASE_URL=http://localhost:8002/v1 (with `make tunnel-qwen`) to compare tokenizers.
verify-live:
	$(PYTHON) verify_memory.py $(if $(QWEN_BASE_URL),--qwen-base-url $(QWEN_BASE_URL))

report:
	$(PYTHON) verify_memory.py --verbose $(if $(QWEN_BASE_URL),--qwen-base-url $(QWEN_BASE_URL))

test:
	$(PYTEST) -q

lint:
	$(RUFF) check .

# Refreshes benchmarks/models.json and benchmarks/configs/ from the Hugging Face API
facts:
	$(PYTHON) -m vllm_lab.hf_facts

stage:
	bash gcp/stage_weights.sh google/gemma-4-E4B-it

stage-quant:
	bash gcp/stage_weights.sh $(QUANT_MODEL)

deploy:
	bash gcp/deploy_vllm.sh google/gemma-4-E4B-it vllm-gemma

deploy-quant:
	bash gcp/deploy_vllm.sh $(QUANT_MODEL) vllm-gemma-qat

deploy-qwen:
	VLLM_SERVED_NAME=qwen bash gcp/deploy_vllm.sh $(QWEN_MODEL) vllm-qwen

tunnel:
	gcloud run services proxy vllm-gemma --region $(REGION) --port 8000

tunnel-quant:
	gcloud run services proxy vllm-gemma-qat --region $(REGION) --port 8001

tunnel-qwen:
	gcloud run services proxy vllm-qwen --region $(REGION) --port 8002

teardown:
	bash gcp/teardown.sh vllm-gemma vllm-gemma-qat vllm-qwen

mock:
	$(PYTHON) -m vllm_lab.mock_server

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__

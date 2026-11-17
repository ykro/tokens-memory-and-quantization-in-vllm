PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)
REGION ?= us-east4
QUANT_MODEL ?= google/gemma-4-E4B-it-qat-w4a16-ct
QWEN_MODEL ?= Qwen/Qwen3.5-4B
QWEN_BASE_URL ?=
EXTRA_ARGS ?=

.PHONY: all verify verify-live report-saved report test lint facts deploy deploy-quant deploy-qwen deploy-toolong deploy-12b deploy-e2b \
	stage stage-quant warm cool tunnel tunnel-quant tunnel-qwen teardown mock clean

# No GPU needed: lint, tests and the judge against the built-in mock servers
all: lint test verify

verify:
	$(PYTHON) verify_memory.py --mock

# Real numbers: needs `make tunnel` and `make tunnel-quant` open in other terminals.
# Add QWEN_BASE_URL=http://localhost:8002/v1 (with `make tunnel-qwen`) to compare tokenizers.
# Re-prints the live report from benchmarks/latest_results.json and memory_ladder.json (no GPU)
report-saved:
	$(PYTHON) verify_memory.py --from-saved

verify-live:
	$(PYTHON) verify_memory.py $(if $(QWEN_BASE_URL),--qwen-base-url $(QWEN_BASE_URL)) $(EXTRA_ARGS)

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

# A context window that does not fit the KV-cache left after the weights (131,072 tokens at
# 0.82 utilization): vLLM refuses to start and says how much KV-cache it would need
deploy-toolong:
	VLLM_MAX_MODEL_LEN=131072 VLLM_GPU_MEMORY_UTILIZATION=0.82 bash gcp/deploy_vllm.sh google/gemma-4-E4B-it vllm-gemma-toolong

deploy-12b:
	bash gcp/deploy_vllm.sh google/gemma-4-12B-it-qat-w4a16-ct vllm-gemma-12b

deploy-e2b:
	bash gcp/deploy_vllm.sh google/gemma-4-E2B-it vllm-gemma-e2b

deploy-qwen:
	VLLM_SERVED_NAME=qwen bash gcp/deploy_vllm.sh $(QWEN_MODEL) vllm-qwen

# Keep the bf16 GPU instance warm while you work (no cold start), and let it scale to zero after
warm:
	gcloud run services update vllm-gemma --region $(REGION) --project $(GCP_PROJECT_ID) --min-instances 1

cool:
	gcloud run services update vllm-gemma --region $(REGION) --project $(GCP_PROJECT_ID) --min-instances 0

tunnel:
	gcloud run services proxy vllm-gemma --region $(REGION) --port 8000

tunnel-quant:
	gcloud run services proxy vllm-gemma-qat --region $(REGION) --port 8001

tunnel-qwen:
	gcloud run services proxy vllm-qwen --region $(REGION) --port 8002

teardown:
	bash gcp/teardown.sh vllm-gemma vllm-gemma-qat vllm-qwen vllm-gemma-toolong vllm-gemma-12b vllm-gemma-e2b

mock:
	$(PYTHON) -m vllm_lab.mock_server

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__

#!/usr/bin/env bash
# Deploys the official vLLM image on Cloud Run with one NVIDIA L4 GPU, for any model.
# On-demand: min-instances 0, so the GPU is billed only while a request keeps it awake.
# C2 runs two services side by side, the bf16 model and its int4 version:
#
#   GCP_PROJECT_ID=my-project bash gcp/deploy_vllm.sh google/gemma-4-E4B-it vllm-gemma
#   GCP_PROJECT_ID=my-project bash gcp/deploy_vllm.sh \
#       google/gemma-4-E4B-it-qat-w4a16-ct vllm-gemma-qat
#
# Arguments (or env): MODEL (VLLM_MODEL), SERVICE (VLLM_SERVICE). Each service takes one L4
# from your regional quota while it is awake; tear down what you no longer use.
# Then open a local tunnel per service (no public URL, no API key in your code):
#   gcloud run services proxy vllm-gemma --region us-east4 --port 8000
#   gcloud run services proxy vllm-gemma-qat --region us-east4 --port 8001
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID to your Google Cloud project id}"
REGION="${GCP_REGION:-us-east4}"            # a region with Cloud Run L4 quota
MODEL="${1:-${VLLM_MODEL:-google/gemma-4-E4B-it}}"
SERVICE="${2:-${VLLM_SERVICE:-vllm-gemma}}"
SERVED_NAME="${VLLM_SERVED_NAME:-gemma}"
IMAGE="${VLLM_IMAGE:-docker.io/vllm/vllm-openai:v0.33.0}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
BUCKET="${WEIGHTS_BUCKET:-${PROJECT_ID}-vllm-weights}"

# Weights: from the bucket filled by gcp/stage_weights.sh when it exists (fast, predictable
# cold starts), otherwise straight from Hugging Face (simplest, but anonymous downloads are
# throttled and a cold start can take several minutes). Quantized checkpoints need no extra
# flag: vLLM reads the quantization method from the checkpoint's config.json.
SOURCE_FLAGS=()
MODEL_ARG="${MODEL}"
if gcloud storage ls "gs://${BUCKET}/${MODEL}/config.json" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  SOURCE_FLAGS=(--add-volume "name=weights,type=cloud-storage,bucket=${BUCKET},readonly=true"
                --add-volume-mount "volume=weights,mount-path=/models")
  MODEL_ARG="/models/${MODEL}"
  echo "Weights from gs://${BUCKET}/${MODEL}"
else
  echo "Weights from Hugging Face (${MODEL}); run gcp/stage_weights.sh for faster cold starts"
fi

gcloud run deploy "${SERVICE}" \
  --project "${PROJECT_ID}" --region "${REGION}" \
  --image "${IMAGE}" \
  --args="--model=${MODEL_ARG},--served-model-name=${SERVED_NAME},--max-model-len=${MAX_LEN},--gpu-memory-utilization=${GPU_UTIL},--port=8080,--host=0.0.0.0" \
  --port 8080 \
  ${SOURCE_FLAGS[@]+"${SOURCE_FLAGS[@]}"} \
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy \
  --cpu 8 --memory 32Gi --no-cpu-throttling \
  --min-instances 0 --max-instances 1 --concurrency 64 --timeout 900 \
  --startup-probe "httpGet.path=/health,httpGet.port=8080,periodSeconds=10,failureThreshold=150,timeoutSeconds=5" \
  --no-allow-unauthenticated

gcloud run services describe "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" \
  --format 'value(status.url)'

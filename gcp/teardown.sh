#!/usr/bin/env bash
# Deletes the Cloud Run services of this unit. With min-instances 0 an idle service costs
# nothing, but deleting it is the only way to be sure. Services that do not exist are skipped.
#
#   GCP_PROJECT_ID=my-project bash gcp/teardown.sh vllm-gemma vllm-gemma-qat vllm-qwen
set -euo pipefail
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID to your Google Cloud project id}"
REGION="${GCP_REGION:-us-east4}"
SERVICES=("$@")
[ ${#SERVICES[@]} -eq 0 ] && SERVICES=("${VLLM_SERVICE:-vllm-gemma}")
for SERVICE in "${SERVICES[@]}"; do
  if gcloud run services describe "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" \
      >/dev/null 2>&1; then
    gcloud run services delete "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" --quiet
  else
    echo "${SERVICE}: not deployed, skipping"
  fi
done

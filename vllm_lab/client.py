"""Talking to your own vLLM server: the OpenAI SDK for chat, plain HTTP for the rest.

The only thing that changes from a hosted API is BASE_URL. With the Cloud Run tunnel
(`gcloud run services proxy vllm-gemma --region us-east4 --port 8000`) the server is
at http://localhost:8000/v1 and no API key is needed: the tunnel signs each request with
your gcloud identity.

Besides the OpenAI-compatible /v1 routes, vLLM serves two routes at the root that this unit
uses: POST /tokenize (how many tokens a text costs, with the model's own tokenizer) and
GET /metrics (Prometheus text, including how much KV-cache the server reserved).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request

from openai import OpenAI

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
QUANT_BASE_URL = os.environ.get("VLLM_QUANT_BASE_URL", "http://localhost:8001/v1")
QWEN_BASE_URL = os.environ.get("VLLM_QWEN_BASE_URL")  # optional, only for experiment 04


def make_client(base_url: str | None = None) -> OpenAI:
    # vLLM ignores the key unless it was started with --api-key; the SDK just needs a string.
    return OpenAI(base_url=base_url or BASE_URL, api_key=os.environ.get("VLLM_API_KEY", "local"))


def root_of(base_url: str) -> str:
    return base_url.rsplit("/v1", 1)[0]


def get_text(url: str, timeout: float = 600) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode()


def post_json(url: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def served_model(base_url: str) -> dict:
    """The first entry of GET /v1/models: served name, weights path (root), max_model_len."""
    return json.loads(get_text(base_url + "/models"))["data"][0]


def chat_once(client: OpenAI, model: str, prompt: str, max_tokens: int = 16) -> dict:
    """One deterministic, non-streaming request."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        seed=7,
    )
    return {
        "text": (resp.choices[0].message.content or "").strip(),
        "completion_tokens": resp.usage.completion_tokens,
        "wall_s": time.perf_counter() - t0,
    }


def tokenize(base_url: str, model: str, text: str) -> int:
    """Token count of `text` with the server's tokenizer, without special tokens."""
    body = {"model": model, "prompt": text, "add_special_tokens": False}
    return int(post_json(root_of(base_url) + "/tokenize", body)["count"])


def kv_cache_info(base_url: str) -> dict | None:
    """Reads vllm:cache_config_info from /metrics: block_size, num_gpu_blocks and friends.

    capacity_tokens = num_gpu_blocks x block_size is how many tokens of KV-cache vLLM
    reserved after loading the weights. Returns None if the metric is missing.
    """
    text = get_text(root_of(base_url) + "/metrics", timeout=60)
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            labels = dict(re.findall(r'(\w+)="([^"]*)"', line))
            blocks = int(labels.get("num_gpu_blocks") or 0)
            size = int(labels.get("block_size") or 0)
            return {
                "num_gpu_blocks": blocks,
                "block_size": size,
                "capacity_tokens": blocks * size,
                "gpu_memory_utilization": labels.get("gpu_memory_utilization"),
                "cache_dtype": labels.get("cache_dtype"),
            }
    return None

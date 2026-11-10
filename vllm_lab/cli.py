"""Shared command line for the experiments: live server by default, --mock without a GPU."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from vllm_lab.client import BASE_URL, QUANT_BASE_URL, QWEN_BASE_URL, root_of

TUNNELS = {
    "gemma": "gcloud run services proxy vllm-gemma --region us-east4 --port 8000",
    "gemma-qat": "gcloud run services proxy vllm-gemma-qat --region us-east4 --port 8001",
    "qwen": "gcloud run services proxy vllm-qwen --region us-east4 --port 8002",
}


def parser(description: str, quant: bool = False, qwen: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--mock", action="store_true", help="use the built-in fake servers (no GPU)")
    p.add_argument("--base-url", default=BASE_URL, help="bf16 Gemma server, /v1 URL")
    if quant:
        p.add_argument("--quant-base-url", default=QUANT_BASE_URL,
                       help="quantized Gemma server, /v1 URL")
        p.add_argument("--extra-base-url", action="append", default=[],
                       help="optional more servers to line up in experiment 05 (E2B, 12B int4), "
                            "repeat the flag, /v1 URL")
    if qwen:
        p.add_argument("--qwen-base-url", default=QWEN_BASE_URL,
                       help="optional Qwen server for the tokenizer comparison, /v1 URL")
    return p


def parse(description: str, quant: bool = False, qwen: bool = False) -> argparse.Namespace:
    return parser(description, quant, qwen).parse_args()


def _alive(url: str) -> bool:
    try:
        urllib.request.urlopen(root_of(url) + "/health", timeout=600).read()
        return True
    except (urllib.error.URLError, OSError):
        return False


def server(args: argparse.Namespace, profile: str = "gemma", url: str | None = None,
           required: bool = True) -> str | None:
    """Returns a ready /v1 URL: a fresh mock with --mock, or the live one after a health check."""
    if args.mock:
        from vllm_lab import mock_server

        _, mock_url = mock_server.start(profile=profile)
        return mock_url
    url = url or args.base_url
    if url and _alive(url):
        return url
    if not required:
        print(f"(no {profile} server at {url}; skipping that comparison)")
        return None
    sys.exit(
        f"No vLLM server at {url}.\nOpen the tunnel first:  {TUNNELS[profile]}\n"
        "or run with --mock."
    )


def resolve(args: argparse.Namespace) -> str:
    return server(args, "gemma")


def show(title: str, payload: dict) -> None:
    print(f"\n== {title}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

"""A tiny fake of vLLM's OpenAI-compatible server, for running everything without a GPU.

It speaks the same HTTP as `vllm serve`: /health, /v1/models, /v1/chat/completions (with
and without streaming), plus the two root routes this unit needs, POST /tokenize and
GET /metrics in Prometheus text. Three profiles stand in for the three servers of the
unit: `gemma` (E4B in bf16), `gemma-qat` (E4B with int4 weights) and `qwen`.

Its numbers are made up on purpose, but they follow the arithmetic of `vllm_lab.memory`
and the measured E4B budget, so the shape of every result matches the real thing. The
tokenizer is a deterministic approximation (about one token per few characters of each
word). Only the live judge (`make verify-live`) measures a real GPU and a real tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vllm_lab.memory import (
    ACTIVATION_RESERVE_GIB,
    GIB,
    GPU_MEMORY_UTILIZATION,
    L4_BUDGET_GIB,
    NON_TORCH_GIB,
)
from vllm_lab.quiz import QUIZ

PREFILL_FIXED_S = 0.002
DECODE_PER_TOKEN_S = 0.001
NATURAL_LENGTH = 40
BLOCK_SIZE = 16

PROFILES = {
    "gemma": {"root": "google/gemma-4-E4B-it", "served": "gemma", "weights_bytes": 15992595884,
              "kv_bytes_per_token": 16384, "chars_per_token": 4.2, "max_model_len": 8192,
              "wrong": {}},
    "gemma-qat": {"root": "google/gemma-4-E4B-it-qat-w4a16-ct", "served": "gemma",
                  "weights_bytes": 11513861828, "kv_bytes_per_token": 16384,
                  "chars_per_token": 4.2, "max_model_len": 8192,
                  "wrong": {6: "Ag"}},  # one miss out of ten, on purpose
    "qwen": {"root": "Qwen/Qwen3.5-4B", "served": "qwen", "weights_bytes": 9319828096,
             "kv_bytes_per_token": 32768, "chars_per_token": 3.6, "max_model_len": 8192,
             "wrong": {}},
}

_SENTENCE = (
    "La memoria de la GPU se reparte entre los pesos del modelo, las activaciones y el "
    "KV-cache, que guarda lo que el modelo ya leyó de cada petición."
)
WORDS = _SENTENCE.split(" ")


def count_tokens(text: str, chars_per_token: float = 4.2) -> int:
    """Deterministic stand-in for a tokenizer: words split into pieces, punctuation alone."""
    pieces = re.findall(r"\w+|[^\w\s]", text)
    return max(1, sum(math.ceil(len(p) / chars_per_token) if p[0].isalnum() or p[0] == "_"
                      else 1 for p in pieces))


def fake_tokens(prompt: str, n: int) -> list[str]:
    start = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % len(WORDS)
    return [WORDS[(start + i) % len(WORDS)] + " " for i in range(n)]


def kv_capacity_tokens(profile: dict) -> int:
    weights_gib = profile["weights_bytes"] / GIB + NON_TORCH_GIB
    kv_gib = L4_BUDGET_GIB - weights_gib - ACTIVATION_RESERVE_GIB
    return int(kv_gib * GIB // profile["kv_bytes_per_token"])


def metrics_text(profile: dict) -> str:
    blocks = kv_capacity_tokens(profile) // BLOCK_SIZE
    return (
        "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n"
        "# TYPE vllm:cache_config_info gauge\n"
        f'vllm:cache_config_info{{block_size="{BLOCK_SIZE}",cache_dtype="auto",'
        f'enable_prefix_caching="True",gpu_memory_utilization="{GPU_MEMORY_UTILIZATION}",'
        f'num_cpu_blocks="0",num_gpu_blocks="{blocks}"}} 1.0\n'
        "# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.\n"
        "# TYPE vllm:kv_cache_usage_perc gauge\n"
        f'vllm:kv_cache_usage_perc{{engine="0",model_name="{profile["served"]}"}} 0.0\n'
    )


def quiz_answer(prompt: str, profile: dict) -> str | None:
    for i, (question, accepted) in enumerate(QUIZ):
        if question in prompt:
            return profile["wrong"].get(i, accepted[0].capitalize())
    return None


def make_handler(profile: dict) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep the terminal clean
            pass

        def _send(self, code: int, data: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code: int, body: dict) -> None:
            self._send(code, json.dumps(body).encode(), "application/json")

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {})
            elif self.path == "/version":
                self._json(200, {"version": "mock"})
            elif self.path == "/metrics":
                self._send(200, metrics_text(profile).encode(), "text/plain; version=0.0.4")
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": profile["served"], "object": "model", "owned_by": "vllm",
                     "max_model_len": profile["max_model_len"], "root": profile["root"]}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                n = count_tokens(body["prompt"], profile["chars_per_token"])
                seed = int(hashlib.sha256(body["prompt"].encode()).hexdigest(), 16)
                self._json(200, {"count": n, "max_model_len": profile["max_model_len"],
                                 "tokens": [(seed >> i) % 262144 for i in range(n)]})
                return
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not found"})
                return
            prompt = "".join(m["content"] for m in body["messages"])
            max_tokens = int(body.get("max_tokens") or 256)
            answer = quiz_answer(prompt, profile)
            if answer is not None:
                tokens = [answer]
            else:
                n = max_tokens if body.get("ignore_eos") else min(max_tokens, NATURAL_LENGTH)
                tokens = fake_tokens(prompt, n)
            n = len(tokens)
            finish = "length" if n == max_tokens else "stop"
            ptoks = count_tokens(prompt, profile["chars_per_token"])
            usage = {"prompt_tokens": ptoks, "completion_tokens": n, "total_tokens": ptoks + n}
            time.sleep(PREFILL_FIXED_S)
            if not body.get("stream"):
                time.sleep(DECODE_PER_TOKEN_S * (n - 1))
                self._json(200, {
                    "id": "chatcmpl-mock", "object": "chat.completion",
                    "model": profile["served"],
                    "choices": [{"index": 0, "finish_reason": finish,
                                 "message": {"role": "assistant", "content": "".join(tokens)}}],
                    "usage": usage})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i, tok in enumerate(tokens):
                if i:
                    time.sleep(DECODE_PER_TOKEN_S)
                self._chunk({"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                             "model": profile["served"],
                             "choices": [{"index": 0, "delta": {"content": tok},
                                          "finish_reason": None}]})
            self._chunk({"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                         "model": profile["served"],
                         "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            if (body.get("stream_options") or {}).get("include_usage"):
                self._chunk({"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                             "model": profile["served"], "choices": [], "usage": usage})
            self._raw(b"data: [DONE]\n\n")
            self._raw(b"")

        def _chunk(self, obj: dict) -> None:
            self._raw(f"data: {json.dumps(obj)}\n\n".encode())

        def _raw(self, data: bytes) -> None:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

    return Handler


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):  # clients closing keep-alive sockets
        pass


def start(port: int = 0, profile: str = "gemma") -> tuple[ThreadingHTTPServer, str]:
    """Starts a mock in a background thread; returns the server and its /v1 base URL."""
    server = _QuietServer(("127.0.0.1", port), make_handler(PROFILES[profile]))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


if __name__ == "__main__":
    servers = [start(8000 + i, name) for i, name in enumerate(("gemma", "gemma-qat", "qwen"))]
    for (_, url), name in zip(servers, ("gemma", "gemma-qat", "qwen"), strict=True):
        print(f"Mock {name:9} at {url}")
    print("Ctrl+C to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        for srv, _ in servers:
            srv.shutdown()

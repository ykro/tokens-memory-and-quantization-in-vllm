"""The memory arithmetic of serving a model, in plain Python.

A GPU running vLLM holds three things:

1. The weights: parameters x bytes per parameter. Fixed, paid once at startup.
2. Activations: scratch space for one forward pass. vLLM measures its peak at startup.
3. The KV-cache: for every token of every active request, the keys and values of each
   attention layer, so decode does not recompute the whole prompt at each step.

vLLM takes `gpu_memory_utilization` of the GPU as its budget, subtracts (1) and (2), and
gives everything left to (3). Worked example, measured on the real server for this unit
(vLLM 0.33.0, google/gemma-4-E4B-it in bf16, Cloud Run with one NVIDIA L4):

    usable on the GPU             22.03 GiB   (the card says 24 GB; 24e9 bytes = 22.35 GiB)
    x 0.92 utilization  = budget  20.27 GiB
    - weights + non-torch memory  15.38 GiB
    - peak activations             1.46 GiB
    = KV-cache                     3.43 GiB   <- this decides how many tokens fit at once

KV bytes per token, the textbook formula:

    2 (a key and a value) x layers x kv_heads x head_dim x bytes per number

Gemma 4 and Qwen 3.5 pay less than that formula says, for three reasons this module
handles from each config.json:

* Sliding-window layers only look at the last `sliding_window` tokens (512 or 1024), so
  their cache stops growing after that many tokens.
* Gemma 4 E2B/E4B reuse the cache of earlier layers in their last `num_kv_shared_layers`
  layers: those layers keep no cache of their own.
* Qwen 3.5 uses "linear attention" in 3 of every 4 layers: a fixed-size summary per
  request instead of one entry per token.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3
BYTES_PER_DTYPE = {"bf16": 2.0, "fp8": 1.0, "int4": 0.5}

# One NVIDIA L4 as vLLM sees it on Cloud Run (measured, see the module docstring).
L4_NOMINAL_GB = 24
L4_USABLE_GIB = 22.03
GPU_MEMORY_UTILIZATION = 0.92
L4_BUDGET_GIB = 20.27
MEASURED_E4B = {
    "model": "google/gemma-4-E4B-it",
    "dtype": "bf16",
    "vllm": "0.33.0",
    "usable_gib": L4_USABLE_GIB,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    "budget_gib": L4_BUDGET_GIB,
    "weights_and_non_torch_gib": 15.38,
    "peak_activation_gib": 1.46,
}
MEASURED_E4B["kv_cache_gib"] = round(
    MEASURED_E4B["budget_gib"] - MEASURED_E4B["weights_and_non_torch_gib"]
    - MEASURED_E4B["peak_activation_gib"], 2)
# Assumption for the other models: the same activation peak as E4B. It depends on vLLM's
# max_num_batched_tokens more than on the model, so it is a reasonable first guess.
ACTIVATION_RESERVE_GIB = MEASURED_E4B["peak_activation_gib"]
# Non-torch memory (CUDA context, NCCL, allocator slack): 15.38 GiB measured minus the
# 14.89 GiB of E4B bf16 weights (7,996,156,490 params x 2 bytes). Assumed equal for all.
NON_TORCH_GIB = 0.49


def weights_gib(params: int, dtype: str) -> float:
    return params * BYTES_PER_DTYPE[dtype] / GIB


def kv_room_gib(weights: float, budget_gib: float = L4_BUDGET_GIB) -> float:
    """What is left for the KV-cache. Zero or negative means vLLM refuses to start."""
    return budget_gib - weights - NON_TORCH_GIB - ACTIVATION_RESERVE_GIB


@dataclass
class KVShape:
    """How one model's KV-cache grows, read from its config.json."""

    layers: int
    kv_heads: int
    head_dim: int
    full_layers: int        # own a cache that grows with every token
    sliding_layers: int     # own a cache capped at sliding_window tokens
    linear_layers: int      # fixed-size state, no per-token cache
    shared_layers: int      # reuse another layer's cache
    sliding_window: int
    full_bytes_per_token: int
    sliding_bytes_per_token: int
    naive_bytes_per_token: int
    max_position: int

    def request_bytes(self, tokens: int, naive: bool = False) -> int:
        """KV bytes one request of `tokens` tokens (prompt + answer) keeps on the GPU."""
        if naive:
            return self.naive_bytes_per_token * tokens
        return (self.full_bytes_per_token * tokens
                + self.sliding_bytes_per_token * min(tokens, self.sliding_window))

    def tokens_that_fit(self, kv_bytes: float, naive: bool = False) -> int:
        """Tokens of a single long request that fit in `kv_bytes`."""
        if kv_bytes <= 0:
            return 0
        if naive:
            return int(kv_bytes // self.naive_bytes_per_token)
        fixed = self.sliding_bytes_per_token * self.sliding_window
        if self.full_bytes_per_token == 0:
            return self.max_position if kv_bytes >= fixed else 0
        return max(int((kv_bytes - fixed) // self.full_bytes_per_token), 0)

    def requests_that_fit(self, kv_bytes: float, tokens: int, naive: bool = False) -> int:
        return int(max(kv_bytes, 0) // self.request_bytes(tokens, naive))


def text_config(config: dict) -> dict:
    """Multimodal configs (Gemma 4, Qwen 3.5) nest the language model under text_config."""
    return config.get("text_config", config)


def kv_shape(config: dict, kv_bytes: float = 2.0) -> KVShape:
    """KV-cache shape from a config.json. kv_bytes: 2 for bf16 (vLLM default), 1 for fp8."""
    t = text_config(config)
    layers = t["num_hidden_layers"]
    kv_heads = t.get("num_key_value_heads") or t["num_attention_heads"]
    head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    types = t.get("layer_types")
    if not types:
        every = t.get("full_attention_interval")
        types = (["full_attention" if (i + 1) % every == 0 else "linear_attention"
                  for i in range(layers)] if every else ["full_attention"] * layers)
    shared = t.get("num_kv_shared_layers") or 0
    owned = types[: layers - shared]
    # Full (global) layers in Gemma 4 may use their own head count and size.
    g_heads = t.get("num_global_key_value_heads") or kv_heads
    g_dim = t.get("global_head_dim") or head_dim
    # attention_k_eq_v (Gemma 4 12B and 31B): keys double as values in the global layers,
    # so those layers store one tensor per token instead of two. Assumption from the config
    # field; the live /metrics reading in experiment 02 is the ground truth.
    g_tensors = 1 if t.get("attention_k_eq_v") else 2
    window = t.get("sliding_window") or 0
    full = owned.count("full_attention")
    sliding = owned.count("sliding_attention")
    linear = owned.count("linear_attention")
    return KVShape(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        full_layers=full,
        sliding_layers=sliding,
        linear_layers=linear,
        shared_layers=shared,
        sliding_window=window,
        full_bytes_per_token=int(full * g_tensors * g_heads * g_dim * kv_bytes),
        sliding_bytes_per_token=int(sliding * 2 * kv_heads * head_dim * kv_bytes),
        naive_bytes_per_token=int(2 * layers * kv_heads * head_dim * kv_bytes),
        max_position=t.get("max_position_embeddings") or 0,
    )

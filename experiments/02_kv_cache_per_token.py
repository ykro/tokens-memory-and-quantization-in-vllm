"""02. The memory nobody counts: KV-cache bytes per token.

While it answers, the model keeps a key and a value for every token it has read, in every
attention layer, so decode does not reread the whole prompt at each step. The textbook
formula, from each model's config.json (benchmarks/configs/):

    2 x layers x kv_heads x head_dim x 2 bytes (bf16)

Gemma 4 and Qwen 3.5 pay less: sliding-window layers keep only the last 512 or 1024
tokens, Gemma 4 E-models reuse the cache of earlier layers in their last layers, and
Qwen 3.5 replaces 3 of every 4 attention layers with a fixed-size state. What remains
after the weights (experiment 01) divided by the bytes per token says how many tokens of
context fit. In live mode, vLLM's own answer comes from /metrics (vllm:cache_config_info).

    python experiments/02_kv_cache_per_token.py [--mock]
"""

from __future__ import annotations

from vllm_lab import cli, hf_facts, kv_cache_info, memory, served_model


def run(base_url: str) -> dict:
    rows = []
    for m in hf_facts.load_models()["models"]:
        if m["quantized"]:
            continue
        shape = memory.kv_shape(hf_facts.load_config(m["id"]))
        room = memory.kv_room_gib(memory.weights_gib(m["params_total"], "bf16"))
        rows.append({
            "id": m["id"],
            "layers": shape.layers,
            "kv_heads": shape.kv_heads,
            "head_dim": shape.head_dim,
            "layers_with_growing_cache": shape.full_layers,
            "layers_with_window": shape.sliding_layers,
            "layers_shared_or_linear": shape.shared_layers + shape.linear_layers,
            "sliding_window": shape.sliding_window,
            "naive_kib_per_token": round(shape.naive_bytes_per_token / 1024, 1),
            "real_kib_per_token": round(shape.full_bytes_per_token / 1024, 1),
            "window_mib_per_request": round(shape.sliding_bytes_per_token *
                                            shape.sliding_window / 2**20, 1),
            "kv_room_gib_bf16": round(room, 2),
            "tokens_fit_naive": shape.tokens_that_fit(room * memory.GIB, naive=True),
            "tokens_fit_real": shape.tokens_that_fit(room * memory.GIB),
        })
    model = served_model(base_url)
    served = hf_facts.find_model(model.get("root"))
    live = kv_cache_info(base_url)
    measured = dict(memory.MEASURED_E4B)
    if served and live:
        shape = memory.kv_shape(hf_facts.load_config(served["id"]))
        live["estimate_tokens"] = shape.tokens_that_fit(measured["kv_cache_gib"] * memory.GIB)
        live["estimate_tokens_naive"] = shape.tokens_that_fit(
            measured["kv_cache_gib"] * memory.GIB, naive=True)
        live["served_weights"] = served["id"]
        live["max_model_len"] = model.get("max_model_len")
    return {"measured_e4b_budget": measured, "rows": rows, "vllm_kv_cache": live}


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    result = run(cli.resolve(args))
    cli.show("Measured budget, Gemma 4 E4B bf16 on one L4", result["measured_e4b_budget"])
    print(f"\n{'model':16} {'textbook':>9} {'past the':>9} {'window':>8} {'room':>10} "
          f"{'tokens that fit':>20}")
    print(f"{'':16} {'KiB/tok':>9} {'window':>9} {'MiB/req':>8} {'(bf16)':>10} "
          f"{'textbook / real':>20}")
    for r in result["rows"]:
        fit = (f"{r['tokens_fit_naive']:,} / {r['tokens_fit_real']:,}"
               if r["kv_room_gib_bf16"] > 0 else "does not start")
        print(f"{r['id'].split('/')[1]:16} {r['naive_kib_per_token']:>9} "
              f"{r['real_kib_per_token']:>9} {r['window_mib_per_request']:>8} "
              f"{r['kv_room_gib_bf16']:>6} GiB {fit:>20}")
    cli.show("What vLLM reserved (/metrics)", result["vllm_kv_cache"] or {"metric": "missing"})


if __name__ == "__main__":
    main()

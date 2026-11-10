"""03. Long context or many users: the trade-off behind --max-model-len.

The KV-cache is one fixed pool. Every active request takes from it in proportion to its
length (prompt plus answer). With the same pool, requests of 2k tokens fit many at a time,
and requests of 32k only a few. `--max-model-len` is the promise to accept requests up to
that length: vLLM refuses to start if the pool cannot hold even one of them, and when the
pool fills up, new requests wait in line.

This experiment divides the measured E4B pool (3.43 GiB) by the size of one request, with
the textbook formula and with Gemma 4's real layout, and turns it into what each
simultaneous slot costs per hour on the instance (benchmarks/pricing.json). In live mode
it also uses vLLM's own capacity from /metrics.

    python experiments/03_max_model_len_tradeoff.py [--mock]
"""

from __future__ import annotations

import json
from pathlib import Path

from vllm_lab import cli, hf_facts, kv_cache_info, memory, served_model

PRICING = Path(__file__).resolve().parent.parent / "benchmarks" / "pricing.json"
LENGTHS = (2048, 8192, 32768)


def run(base_url: str) -> dict:
    price = json.loads(PRICING.read_text())
    root = served_model(base_url).get("root")
    served = hf_facts.find_model(root) or hf_facts.find_model("google/gemma-4-E4B-it")
    shape = memory.kv_shape(hf_facts.load_config(served["id"]))
    pool = memory.MEASURED_E4B["kv_cache_gib"] * memory.GIB
    live = kv_cache_info(base_url)
    capacity = live["capacity_tokens"] if live else None
    rows = []
    for n in LENGTHS:
        naive = shape.requests_that_fit(pool, n, naive=True)
        real = shape.requests_that_fit(pool, n)
        row = {
            "tokens_per_request": n,
            "mib_per_request_naive": round(shape.request_bytes(n, naive=True) / 2**20, 1),
            "mib_per_request_real": round(shape.request_bytes(n) / 2**20, 1),
            "concurrent_naive": naive,
            "concurrent_real": real,
            "usd_per_slot_hour_real": round(price["usd_per_hour"] / real, 4) if real else None,
        }
        if capacity is not None:
            row["concurrent_vllm"] = capacity // n
        rows.append(row)
    return {
        "model": served["id"],
        "kv_pool_gib": memory.MEASURED_E4B["kv_cache_gib"],
        "vllm_capacity_tokens": capacity,
        "usd_per_hour": price["usd_per_hour"],
        "rows": rows,
    }


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    result = run(cli.resolve(args))
    print(f"\nKV pool {result['kv_pool_gib']} GiB, {result['model']}, "
          f"${result['usd_per_hour']}/h for the instance")
    print(f"{'tokens/request':>15} {'MiB/request':>12} {'at once':>18} {'$/slot-hour':>12}")
    print(f"{'':>15} {'real':>12} {'textbook/real/vLLM':>18} {'(real)':>12}")
    for r in result["rows"]:
        live = r.get("concurrent_vllm", "-")
        print(f"{r['tokens_per_request']:>15,} {r['mib_per_request_real']:>12} "
              f"{str(r['concurrent_naive']) + '/' + str(r['concurrent_real']) + '/' + str(live):>18} "
              f"{r['usd_per_slot_hour_real'] or '-':>12}")


if __name__ == "__main__":
    main()

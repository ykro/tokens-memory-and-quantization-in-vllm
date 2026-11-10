"""The Outside Judge for tokens, memory and quantization: runs the five experiments once and checks what must hold.

    python verify_memory.py          # live vLLM through the tunnels (real numbers)
    python verify_memory.py --mock   # fake servers, checks the logic without a GPU

Live mode needs two tunnels open: the bf16 server on port 8000 and the quantized one on
port 8001 (`make tunnel` and `make tunnel-quant`). The Qwen server on port 8002 is
optional (`--qwen-base-url`); without it experiment 04 measures only Gemma.

The checks are about behavior (bigger models need more memory, int4 needs less than bf16,
every token costs KV-cache, longer requests leave room for fewer at once, the quantized
model frees memory without losing more than the stated tolerance), not exact numbers.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import platform
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

EXPERIMENTS = [
    "01_weights_math",
    "02_kv_cache_per_token",
    "03_max_model_len_tradeoff",
    "04_tokenizer_comparison",
    "05_quantization_quality",
]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def server_version(base_url: str) -> str:
    try:
        root = base_url.rsplit("/v1", 1)[0]
        with urllib.request.urlopen(root + "/version", timeout=30) as r:
            return json.loads(r.read()).get("version", "?")
    except (OSError, ValueError):
        return "?"


def main() -> int:
    from vllm_lab import cli

    p = cli.parser(__doc__.splitlines()[0], quant=True, qwen=True)
    p.add_argument("--verbose", action="store_true", help="print each experiment's raw result")
    args = p.parse_args()

    base_url = cli.resolve(args)
    quant_url = cli.server(args, "gemma-qat", args.quant_base_url)
    qwen_url = (cli.server(args, "qwen", args.qwen_base_url, required=False)
                if args.mock or args.qwen_base_url else None)
    mods = {name: load(name) for name in EXPERIMENTS}
    res = {
        "01_weights_math": mods["01_weights_math"].run(base_url),
        "02_kv_cache_per_token": mods["02_kv_cache_per_token"].run(base_url),
        "03_max_model_len_tradeoff": mods["03_max_model_len_tradeoff"].run(base_url),
        "04_tokenizer_comparison": mods["04_tokenizer_comparison"].run(base_url, qwen_url),
        "05_quantization_quality": mods["05_quantization_quality"].run(base_url, quant_url),
    }
    r1, r2, r3, r4, r5 = (res[n] for n in EXPERIMENTS)
    if args.verbose:
        for name, value in res.items():
            cli.show(name, value)

    w = r1["rows"]  # sorted by parameter count
    e4b = next(r for r in w if r["id"].endswith("gemma-4-E4B-it"))
    grows = all(a["bf16_gib"] < b["bf16_gib"] for a, b in itertools.pairwise(w))
    int4_smaller = all(r["int4_gib"] < r["fp8_gib"] < r["bf16_gib"] for r in w)
    kv = r2["vllm_kv_cache"] or {}
    kv_ok = all(r["naive_kib_per_token"] > 0 and r["real_kib_per_token"] > 0
                for r in r2["rows"]) and kv.get("capacity_tokens", 1) > 0
    conc = [row["concurrent_real"] for row in r3["rows"]]
    live_conc = [row.get("concurrent_vllm", 0) for row in r3["rows"]]
    fewer = (all(a >= b for a, b in itertools.pairwise(conc)) and conc[0] > conc[-1]
             and all(a >= b for a, b in itertools.pairwise(live_conc)))
    g = r4["gemma"]
    tok_ok = all(g[lang]["tokens"] > 0 and g[lang]["repeatable"] for lang in ("es", "en", "code"))
    q = r4["qwen"]
    if q:
        tok_ok = tok_ok and all(q[lang]["tokens"] > 0 for lang in ("es", "en", "code"))
    bf, qt = r5["bf16"], r5["quantized"]
    mem_lower = bool(r5["weights_saved_gib"]) and r5["weights_saved_gib"] > 0
    if r5["kv_capacity_gain_x"] is not None:
        mem_lower = mem_lower and r5["kv_capacity_gain_x"] > 1
    acc_ok = r5["accuracy_drop"] <= r5["tolerance"] + 1e-9

    rows = [
        ("01. Weights grow with params", f"E4B bf16 {e4b['bf16_gib']:.2f} GiB",
         f"int4 {e4b['int4_gib']:.2f} GiB", grows),
        ("01. int4 < fp8 < bf16", f"{len(w)} models",
         "smaller every time" if int4_smaller else "not monotonic", int4_smaller),
        ("02. KV-cache per token > 0",
         (f"E4B {next(r for r in r2['rows'] if r['id'] == e4b['id'])['real_kib_per_token']}"
          " KiB/token"),
         (f"vLLM {kv['capacity_tokens']:,} tok" if kv else "no /metrics"), kv_ok),
        ("03. More context, fewer at once",
         "/".join(str(r["tokens_per_request"] // 1024) + "k" for r in r3["rows"]),
         "/".join(str(c) for c in conc) + " requests", fewer),
        ("04. Tokenizer is repeatable",
         f"es {g['es']['tokens_per_word']} en {g['en']['tokens_per_word']} tok/word",
         (f"Qwen es {q['es']['tokens_per_word']} tok/word" if q else "Gemma only"), tok_ok),
        ("05. Quantized needs less memory",
         f"{bf['weights_file_gib']} -> {qt['weights_file_gib']} GiB",
         (f"KV x{r5['kv_capacity_gain_x']}" if r5["kv_capacity_gain_x"] else "files only"),
         mem_lower),
        ("05. Quantized accuracy in tolerance",
         f"bf16 {bf['accuracy']:.0%} int4 {qt['accuracy']:.0%}",
         f"drop {r5['accuracy_drop']:.0%} <= {r5['tolerance']:.0%}", acc_ok),
    ]

    mode = "mock (fake servers, numbers are not real)" if args.mock else "live"
    version = "mock" if args.mock else server_version(base_url)
    lines = [
        "=" * 100,
        "THE OUTSIDE JUDGE — Tokens, Memory and Quantization in vLLM",
        "=" * 100,
        f"Mode    : {mode}",
        f"Server  : vLLM {version} | bf16 {bf['weights']} | quantized {qt['weights']}",
        (f"GPU     : NVIDIA L4, {r1['gpu']['usable_gib']} GiB usable, "
         f"budget {r1['gpu']['budget_gib']} GiB at {r1['gpu']['gpu_memory_utilization']}"),
        f"Client  : Python {platform.python_version()} | {platform.system()} {platform.machine()}",
        "-" * 100,
        f"{'Check':36} | {'Measured':30} | {'Result':22} | Check",
        "-" * 100,
    ]
    for name, measured, result, ok in rows:
        lines.append(f"{name:36} | {measured:30} | {result:22} | {'PASS' if ok else 'FAIL'}")
    lines += [
        "-" * 100,
        f"Model sizes from the Hugging Face API, checked {r1['checked']}.",
        "Concurrency counts KV-cache only; vLLM also caps running requests (max_num_seqs).",
    ]
    all_ok = all(ok for *_, ok in rows)
    lines.append("ALL CHECKS PASSED." if all_ok else "SOME CHECKS FAILED.")
    report = "\n".join(lines)
    print(report)

    if not args.mock:
        out = ROOT / "benchmarks"
        out.mkdir(exist_ok=True)
        payload = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "vllm_version": version,
            "experiments": res,
            "checks": {name: ok for name, *_, ok in rows},
        }
        (out / "latest_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        (out / "terminal_output.txt").write_text(report + "\n")
        print("Saved benchmarks/latest_results.json and benchmarks/terminal_output.txt")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

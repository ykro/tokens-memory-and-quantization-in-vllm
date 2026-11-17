"""The Outside Judge for tokens, memory and quantization: runs the five experiments once and checks what must hold.

    python verify_memory.py          # live vLLM through the tunnels (real numbers)
    python verify_memory.py --mock   # fake servers, checks the logic without a GPU
    python verify_memory.py --from-saved   # re-prints the report from benchmarks/latest_results.json

Live mode needs two tunnels open: the bf16 server on port 8000 and the quantized one on
port 8001 (`make tunnel` and `make tunnel-quant`). The Qwen server on port 8002 is
optional (`--qwen-base-url`); without it experiment 04 measures only Gemma.

The model ladder row (E2B, E4B, E4B int4, 12B int4 on the same quiz) reads
benchmarks/memory_ladder.json, which every live run of experiment 05 adds to, because the
L4 quota rarely allows every server up at once.

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


LADDER_ORDER = ("E2B", "E4B", "E4B int4", "12B int4")


def ladder_label(weights: str) -> str:
    """'/models/google/gemma-4-E4B-it-qat-w4a16-ct' -> 'E4B int4'."""
    name = weights.rsplit("gemma-4-", 1)[-1]
    size = name.split("-")[0]
    return f"{size} int4" if "w4a16" in name else size


def build_ladder(r5: dict) -> list[dict]:
    """Models measured on the same quiz: the saved ladder file first, this run fills the gaps.

    Rows from this run's extra servers replace the saved ones; the bf16 and quantized servers of
    this run only fill models the file does not have (so a mock run never replaces real numbers).
    """
    path = ROOT / "benchmarks" / "memory_ladder.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    rows = {ladder_label(w): dict(v) for w, v in saved.items()}
    for e in (r5["bf16"], r5["quantized"]):
        rows.setdefault(ladder_label(e["weights"]), {k: e[k] for k in (
            "weights", "accuracy", "weights_file_gib", "kv_capacity_tokens")})
    for e in r5["extras"]:
        rows[ladder_label(e["weights"])] = {k: e[k] for k in (
            "weights", "accuracy", "weights_file_gib", "kv_capacity_tokens")}
    out = [{"model": label, **rows[label]} for label in LADDER_ORDER if label in rows]
    out += [{"model": label, **v} for label, v in rows.items() if label not in LADDER_ORDER]
    return out


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
    p.add_argument("--from-saved", action="store_true",
                   help="rebuild the report from benchmarks/latest_results.json (no servers)")
    args = p.parse_args()

    saved_at = None
    if args.from_saved:
        saved = json.loads((ROOT / "benchmarks" / "latest_results.json").read_text())
        res = saved["experiments"]
        version = saved["vllm_version"]
        saved_at = saved["generated_at"]
    else:
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
            "05_quantization_quality": mods["05_quantization_quality"].run(
                base_url, quant_url, args.extra_base_url),
        }
        version = "mock" if args.mock else server_version(base_url)
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
    reported = "/".join(str(c) for c in live_conc)  # what vLLM's own capacity gives
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
    lad = build_ladder(r5)
    lad_ok = (len(lad) >= 3 and all((m["kv_capacity_tokens"] or 0) > 0 for m in lad)
              and all(m["accuracy"] >= 1 - r5["tolerance"] - 1e-9 for m in lad))

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
         f"vLLM {reported} at "
         + "/".join(str(r["tokens_per_request"] // 1024) + "k" for r in r3["rows"]),
         "estimate " + "/".join(str(c) for c in conc), fewer),
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
        ("05. Ladder: same quiz, 4 models",
         "KV " + "/".join(f"{m['kv_capacity_tokens']:,}" for m in lad),
         f"quiz >= {min(m['accuracy'] for m in lad):.0%} in all", lad_ok),
    ]

    if args.mock:
        mode = "mock (fake servers, numbers are not real)"
    elif args.from_saved:
        mode = f"live numbers saved {saved_at} (re-printed, no servers)"
    else:
        mode = "live"
    lines = [
        "=" * 109,
        "THE OUTSIDE JUDGE — Tokens, Memory and Quantization in vLLM",
        "=" * 109,
        f"Mode    : {mode}",
        f"Server  : vLLM {version} | bf16 {bf['weights']} | quantized {qt['weights']}",
        (f"GPU     : NVIDIA L4, {r1['gpu']['usable_gib']} GiB usable, "
         f"budget {r1['gpu']['budget_gib']} GiB at {r1['gpu']['gpu_memory_utilization']}"),
        f"Client  : Python {platform.python_version()} | {platform.system()} {platform.machine()}",
        "-" * 109,
        f"{'Check':36} | {'Measured':38} | {'Result':22} | Check",
        "-" * 109,
    ]
    for name, measured, result, ok in rows:
        lines.append(f"{name:36} | {measured:38} | {result:22} | {'PASS' if ok else 'FAIL'}")
    lines += [
        "-" * 109,
        f"Model sizes from the Hugging Face API, checked {r1['checked']}.",
        "Row 03 divides the capacity vLLM reports by the request length; 'estimate' is the same",
        "division with the config.json math, which was 2.4x too optimistic (see experiment 02).",
        "Concurrency counts KV-cache only; vLLM also caps running requests (max_num_seqs).",
        "Ladder (KV tokens, in order): " + ", ".join(m["model"] for m in lad)
        + ". File sizes: " + "/".join(str(m["weights_file_gib"]) for m in lad) + " GiB.",
        "A smaller file does not mean more KV-cache: the 12B int4 file is smaller than E4B bf16 and still leaves the least KV.",
    ]
    all_ok = all(ok for *_, ok in rows)
    lines.append("ALL CHECKS PASSED." if all_ok else "SOME CHECKS FAILED.")
    report = "\n".join(lines)
    print(report)

    if not args.mock:
        out = ROOT / "benchmarks"
        out.mkdir(exist_ok=True)
        payload = {
            "generated_at": saved_at or datetime.now(UTC).isoformat(timespec="seconds"),
            "vllm_version": version,
            "experiments": res,
            "ladder": lad,
            "checks": {name: ok for name, *_, ok in rows},
        }
        (out / "latest_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        (out / "terminal_output.txt").write_text(report + "\n")
        print("Saved benchmarks/latest_results.json and benchmarks/terminal_output.txt")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

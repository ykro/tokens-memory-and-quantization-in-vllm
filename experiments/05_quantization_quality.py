"""05. Quantization as the way out: less memory, how much quality?

Asks the same ten closed questions (vllm_lab/quiz.py) to two servers: Gemma 4 E4B in bf16
(--base-url) and the int4 checkpoint Google trained for it
(google/gemma-4-E4B-it-qat-w4a16-ct, --quant-base-url). "w4a16" means 4-bit weights and
16-bit activations: the matrices shrink, the math still runs in bf16. "qat" means the
model was trained knowing it would be quantized, which keeps quality closer to bf16.

Memory comes from the checked-in file sizes (benchmarks/models.json) and, in live mode,
from each server's KV-cache capacity in /metrics: the gigabytes the weights free up go
straight to the KV-cache. Ten questions only catch large drops; for your app, evaluate on
your own data.

    python experiments/05_quantization_quality.py [--mock] [--quant-base-url URL]
"""

from __future__ import annotations

from vllm_lab import (
    QUANT_BASE_URL,
    chat_once,
    cli,
    hf_facts,
    kv_cache_info,
    make_client,
    memory,
    served_model,
)
from vllm_lab.quiz import QUIZ, is_correct

TOLERANCE = 0.2  # the quantized model may miss at most 2 more questions out of 10


def ask_all(base_url: str) -> dict:
    model = served_model(base_url)
    client = make_client(base_url)
    answers = []
    for question, accepted in QUIZ:
        text = chat_once(client, model["id"], question, max_tokens=16)["text"]
        answers.append({"question": question, "answer": text,
                        "ok": is_correct(text, accepted)})
    facts = hf_facts.find_model(model.get("root"))
    kv = kv_cache_info(base_url)
    return {
        "weights": model.get("root"),
        "accuracy": sum(a["ok"] for a in answers) / len(answers),
        "weights_file_gib": round(facts["safetensors_bytes"] / memory.GIB, 2) if facts else None,
        "kv_capacity_tokens": kv["capacity_tokens"] if kv else None,
        "answers": answers,
    }


def run(base_url: str, quant_base_url: str = QUANT_BASE_URL) -> dict:
    bf16 = ask_all(base_url)
    quant = ask_all(quant_base_url)
    saved = None
    if bf16["weights_file_gib"] and quant["weights_file_gib"]:
        saved = round(bf16["weights_file_gib"] - quant["weights_file_gib"], 2)
    kv_x = None
    if bf16["kv_capacity_tokens"] and quant["kv_capacity_tokens"]:
        kv_x = round(quant["kv_capacity_tokens"] / bf16["kv_capacity_tokens"], 2)
    return {
        "bf16": bf16,
        "quantized": quant,
        "weights_saved_gib": saved,
        "kv_capacity_gain_x": kv_x,
        "accuracy_drop": round(bf16["accuracy"] - quant["accuracy"], 2),
        "tolerance": TOLERANCE,
    }


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0], quant=True)
    base = cli.resolve(args)
    quant = cli.server(args, "gemma-qat", args.quant_base_url)
    r = run(base, quant)
    print(f"\n{'#':>2} {'bf16':20} {'int4 (qat)':20} question")
    for i, (a, b) in enumerate(zip(r["bf16"]["answers"], r["quantized"]["answers"],
                                   strict=True), 1):
        print(f"{i:>2} {('ok  ' if a['ok'] else 'MISS') + ' ' + a['answer'][:14]:20} "
              f"{('ok  ' if b['ok'] else 'MISS') + ' ' + b['answer'][:14]:20} "
              f"{a['question'][:60]}")
    cli.show("Memory vs quality", {k: r[k] for k in ("weights_saved_gib", "kv_capacity_gain_x",
                                                     "accuracy_drop", "tolerance")} | {
        "bf16": {k: r["bf16"][k] for k in ("weights", "accuracy", "weights_file_gib",
                                            "kv_capacity_tokens")},
        "quantized": {k: r["quantized"][k] for k in ("weights", "accuracy", "weights_file_gib",
                                                      "kv_capacity_tokens")}})


if __name__ == "__main__":
    main()

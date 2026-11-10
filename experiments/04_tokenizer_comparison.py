"""04. A token is not a word: the same text in Spanish, English and code.

Sends the same content, written three ways, to the server's POST /tokenize, which uses the
model's own tokenizer. Prices, context limits and the KV-cache are all counted in tokens,
so tokens per word is the exchange rate between your text and your bill. Gemma and Qwen
cut the same text differently because each learned its own vocabulary. Pass a second
server with --qwen-base-url to compare; without it, only Gemma is measured. In --mock the
tokenizer is a deterministic approximation, so only the shape of the result means anything.

    python experiments/04_tokenizer_comparison.py [--mock] [--qwen-base-url URL]
"""

from __future__ import annotations

import re

from vllm_lab import cli, served_model, tokenize

TEXTS = {
    "es": (
        "El cliente compró unos audífonos el lunes. Llegaron el jueves con la caja abierta "
        "y el lado izquierdo no suena. Pide un cambio por otro par igual, sin reembolso."
    ),
    "en": (
        "The customer bought some headphones on Monday. They arrived on Thursday with the "
        "box open and the left side makes no sound. They want an exchange for an identical "
        "pair, no refund."
    ),
    "code": (
        "def handle_ticket(ticket: dict) -> dict:\n"
        "    if ticket['arrived_open'] and not ticket['left_side_works']:\n"
        "        return {'action': 'exchange', 'refund': False}\n"
        "    return {'action': 'review'}\n"
    ),
}
CONTEXT = 8192  # tokens, the max_model_len used in this series


def words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def measure(base_url: str) -> dict:
    model = served_model(base_url)
    out = {"weights": model.get("root")}
    for lang, text in TEXTS.items():
        n = tokenize(base_url, model["id"], text)
        w = words(text)
        out[lang] = {"words": w, "tokens": n, "tokens_per_word": round(n / w, 2),
                     "words_in_8k_context": int(CONTEXT / (n / w)),
                     "repeatable": n == tokenize(base_url, model["id"], text)}
    out["es_vs_en_tokens_x"] = round(
        out["es"]["tokens_per_word"] / out["en"]["tokens_per_word"], 2)
    return out


def run(base_url: str, qwen_base_url: str | None = None) -> dict:
    result = {"gemma": measure(base_url), "qwen": None}
    if qwen_base_url:
        result["qwen"] = measure(qwen_base_url)
        result["qwen_vs_gemma_tokens_x"] = {
            lang: round(result["qwen"][lang]["tokens"] / result["gemma"][lang]["tokens"], 2)
            for lang in TEXTS
        }
    return result


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0], qwen=True)
    base = cli.resolve(args)
    qwen = cli.server(args, "qwen", args.qwen_base_url, required=False) if (
        args.mock or args.qwen_base_url) else None
    result = run(base, qwen)
    print(f"\n{'':6} {'words':>6} {'Gemma tokens':>13} {'tok/word':>9} "
          f"{'Qwen tokens':>12} {'tok/word':>9}")
    for lang in TEXTS:
        g = result["gemma"][lang]
        q = (result["qwen"] or {}).get(lang)
        qcols = f"{q['tokens']:>12} {q['tokens_per_word']:>9}" if q else f"{'-':>12} {'-':>9}"
        print(f"{lang:6} {g['words']:>6} {g['tokens']:>13} {g['tokens_per_word']:>9} {qcols}")
    print(f"\nSpanish costs {result['gemma']['es_vs_en_tokens_x']}x the tokens per word of "
          "English on Gemma: same story, bigger bill, less room in the context.")


if __name__ == "__main__":
    main()

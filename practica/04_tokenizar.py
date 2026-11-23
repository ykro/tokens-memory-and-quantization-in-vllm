"""Cuántos tokens cuesta tu texto por POST /tokenize y cuántas palabras caben en 8,192."""
import json
import re
import urllib.request
from pathlib import Path

TEXTOS = {
    "en": "The customer bought some headphones on Monday and the left side makes no sound.",
    "es": "El cliente compró unos audífonos el lunes y el lado izquierdo no suena.",
    "code": "def ticket(d): return {'cambio': True, 'reembolso': False}",
}
for clave, texto in TEXTOS.items():
    datos = json.dumps({"model": "gemma", "prompt": texto}).encode()
    req = urllib.request.Request("http://localhost:8000/tokenize", datos,
                                 {"Content-Type": "application/json"})
    n = json.loads(urllib.request.urlopen(req).read())["count"]
    palabras = len(re.findall(r"\w+", texto))
    print(f"{clave:4} -> {n:2} tokens ({n / palabras:.2f}/palabra, "
          f"{int(8192 / (n / palabras)):,} palabras en 8,192)")
g = json.loads(Path("benchmarks/latest_results.json").read_text())["experiments"]["04_tokenizer_comparison"]["gemma"]
print(f"Gemma real -> en {g['en']['tokens_per_word']:.2f} ({g['en']['words_in_8k_context']:,}), "
      f"es {g['es']['tokens_per_word']:.2f} ({g['es']['words_in_8k_context']:,}), code {g['code']['tokens_per_word']:.2f}")

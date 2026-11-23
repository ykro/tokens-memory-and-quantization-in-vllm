"""Cuántas conversaciones caben a la vez en la bolsa real de KV-cache de vLLM."""
import json
from pathlib import Path

corrida = json.loads(Path("benchmarks/latest_results.json").read_text())
q = corrida["experiments"]["05_quantization_quality"]
bf16, int4 = q["bf16"]["kv_capacity_tokens"], q["quantized"]["kv_capacity_tokens"]
largos = (2048, 8192, 32768)

print(f"KV-cache real en L4 -> bf16: {bf16:,} tokens | int4: {int4:,} tokens")
for largo in largos:
    print(f"{largo:>6,} tokens/petición -> bf16: {bf16 // largo:>2} a la vez | "
          f"int4: {int4 // largo:>2} a la vez")

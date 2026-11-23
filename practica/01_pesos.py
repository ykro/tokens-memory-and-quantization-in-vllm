"""Cuánto pesan los pesos en GiB reales frente a los 22.03 GiB utilizables de una L4."""
import json
from pathlib import Path

GIB, UTILIZABLE_L4 = 2**30, 22.03
BYTES = {"bf16": 2.0, "fp8": 1.0, "int4": 0.5}
dtype = "bf16"
modelos = {m["id"]: m for m in json.loads(Path("benchmarks/models.json").read_text())["models"]}

for nombre in ("google/gemma-4-E4B-it", "google/gemma-4-12B-it"):
    params = modelos[nombre]["params_total"]
    gib = params * BYTES[dtype] / GIB
    estado = "cabe" if gib < UTILIZABLE_L4 else "no cabe"
    print(f"{nombre.split('/')[1]:14} {params / 1e9:5.2f}B en {dtype}: {gib:5.2f} GiB ({estado})")

qat_gib = modelos["google/gemma-4-E4B-it-qat-w4a16-ct"]["safetensors_bytes"] / GIB
print(f"checkpoint int4 real de E4B: {qat_gib:.2f} GiB (L4 utilizable: {UTILIZABLE_L4} GiB)")
